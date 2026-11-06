import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295.0

    # Precompute clause data as numpy arrays for vectorized operations
    # clause_vars[i] = (var_indices, is_positive_flags)
    clause_var_indices = np.zeros((m, 3), dtype=np.int32)
    clause_is_positive = np.zeros((m, 3), dtype=bool)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            clause_var_indices[i, j] = abs(l) - 1
            clause_is_positive[i, j] = (l > 0)

    # Build lookup: for each variable, which clauses and positions it appears in
    var_clause_info = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var = clause_var_indices[i, j]
            var_clause_info[var].append((i, j))

    # Precompute for each clause: the 3 variable indices and polarity
    # Also precompute "flip masks" - for each variable, the clauses it affects
    var_affected_clauses = np.zeros(n, dtype=np.int32)
    for v in range(n):
        var_affected_clauses[v] = len(var_clause_info[v])

    def compute_unsat_count(v_int):
        """Count satisfied clauses using vectorized numpy operations."""
        # Convert int assignment to numpy bool array
        assign = np.array([(v_int >> i) & 1 for i in range(n)], dtype=np.int8)
        
        # For each clause, check if any literal is satisfied
        # Literal value: if positive, assign[var]; if negative, 1 - assign[var]
        var_vals = assign[clause_var_indices]  # shape (m, 3)
        lit_vals = np.where(clause_is_positive, var_vals, 1 - var_vals)
        
        # Clause is satisfied if any literal is True
        sat_mask = np.any(lit_vals == 1, axis=1)
        return int(np.sum(sat_mask)), sat_mask

    def update_unsat_after_flip(v_int, var, unsat_mask):
        """
        Given current assignment (as int), a variable to flip, and current unsat mask,
        return (new_unsat_mask, gain).
        """
        assign = np.array([(v_int >> i) & 1 for i in range(n)], dtype=np.int8)
        
        # Get clauses affected by this variable
        affected = var_clause_info[var]
        if not affected:
            return unsat_mask, 0
        
        # For each affected clause, check if flipping var changes satisfaction
        gain = 0
        new_unsat = list(unsat_mask)
        
        for (ci, pos) in affected:
            # Current value of this variable in assignment
            cur_val = assign[var]
            # Value of this literal in the clause
            is_pos = clause_is_positive[ci, pos]
            lit_val = cur_val if is_pos else (1 - cur_val)
            
            # After flip
            new_lit_val = (1 - cur_val) if is_pos else cur_val
            
            if lit_val == 0:
                # Literal was False, becomes True -> clause becomes satisfied
                if new_unsat[ci]:
                    new_unsat[ci] = False
                    gain += 1
            else:
                # Literal was True, becomes False -> clause may become unsatisfied
                # Check if any OTHER literal in this clause is True
                other_true = False
                for j in range(3):
                    if j == pos:
                        continue
                    var2 = clause_var_indices[ci, j]
                    is_pos2 = clause_is_positive[ci, j]
                    val2 = assign[var2] if is_pos2 else (1 - assign[var2])
                    if val2 == 1:
                        other_true = True
                        break
                
                if not other_true:
                    # No other true literal, clause becomes unsatisfied
                    if not new_unsat[ci]:
                        new_unsat[ci] = True
                        gain -= 1
        
        return new_unsat, gain

    def get_candidate_vars(unsat_mask):
        """Get variables appearing in unsatisfied clauses."""
        candidates = set()
        for ci, is_unsat in enumerate(unsat_mask):
            if is_unsat:
                for j in range(3):
                    candidates.add(clause_var_indices[ci, j])
        return candidates

    best_assign = None
    best_sat = -1

    # Phase 0: Random initializations
    for attempt_seed in range(16):
        r = random.Random(attempt_seed * 31 + 7)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)
        
        sat_count, sat_mask = compute_unsat_count(v)
        unsat_mask = [not s for s in sat_mask]
        
        if sat_count > best_sat:
            best_sat = sat_count
            best_assign = assign
        if sat_count == m:
            return assign
        if time.time() > deadline:
            return best_assign if best_assign else [False] * n

    # Main search: WalkSAT with adaptive parameters
    max_restarts = 2000
    steps_per_restart = 3000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        r = random.Random(restart * 7919 + 13)
        assign = [r.random() < 0.5 for _ in range(n)]
        v = 0
        for i, b in enumerate(assign):
            if b:
                v |= (1 << i)

        sat_count, sat_mask = compute_unsat_count(v)
        unsat_mask = [not s for s in sat_mask]
        cur_sat = sat_count

        if cur_sat > best_sat:
            best_sat = cur_sat
            best_assign = [bool((v >> i) & 1) for i in range(n)]
        if cur_sat == m:
            return best_assign

        local_best_sat = cur_sat
        local_best_v = v

        for step in range(steps_per_restart):
            if time.time() > deadline:
                return best_assign if best_assign else [False] * n

            if not any(unsat_mask):
                best_assign = [bool((v >> i) & 1) for i in range(n)]
                best_sat = m
                return best_assign

            # Get candidate variables from unsatisfied clauses
            candidate_vars = get_candidate_vars(unsat_mask)
            if not candidate_vars:
                break

            # Find best flip
            best_gain = -100
            best_var = -1
            best_new_unsat = unsat_mask

            for var in candidate_vars:
                new_unsat, gain = update_unsat_after_flip(v, var, unsat_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_var = var
                    best_new_unsat = new_unsat
                if best_gain >= 3:
                    break

            # Decide move
            if best_gain > 0:
                v ^= (1 << best_var)
                unsat_mask = best_new_unsat
                cur_sat = m - sum(unsat_mask)

                if cur_sat > local_best_sat:
                    local_best_sat = cur_sat
                    local_best_v = v
                    if cur_sat > best_sat:
                        best_sat = cur_sat
                        best_assign = [bool((v >> i) & 1) for i in range(n)]

            elif best_gain == 0:
                # Neutral move: accept with 50% probability
                if r.random() < 0.5:
                    v ^= (1 << best_var)
                    unsat_mask = best_new_unsat
                    cur_sat = m - sum(unsat_mask)
                else:
                    var = r.choice(list(candidate_vars))
                    v ^= (1 << var)
                    _, sat_mask = compute_unsat_count(v)
                    unsat_mask = [not s for s in sat_mask]
                    cur_sat = m - sum(unsat_mask)

            else:
                # Local minimum: random escape
                var = r.choice(list(candidate_vars))
                v ^= (1 << var)
                _, sat_mask = compute_unsat_count(v)
                unsat_mask = [not s for s in sat_mask]
                cur_sat = m - sum(unsat_mask)

                if cur_sat > local_best_sat:
                    local_best_sat = cur_sat
                    local_best_v = v
                    if cur_sat > best_sat:
                        best_sat = cur_sat
                        best_assign = [bool((v >> i) & 1) for i in range(n)]

        if local_best_sat > best_sat:
            best_sat = local_best_sat
            best_assign = [bool((local_best_v >> i) & 1) for i in range(n)]

        if best_sat == m:
            return best_assign

    if best_assign is None:
        return [False] * n
    return best_assign
