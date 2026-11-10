import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Precompute clause literal arrays
    clause_vars = np.empty((m, 3), dtype=np.int32)
    clause_signs = np.empty((m, 3), dtype=np.int8) # 1 for pos, -1 for neg
    
    # Also build reverse index: var -> list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]

    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            clause_vars[i, j] = v
            clause_signs[i, j] = s
            var_clauses[v].append((i, s))

    # Build M_pos and M_neg for gain calculation
    # M_pos[v, c] = 1 if clause c contains +v
    # M_neg[v, c] = 1 if clause c contains -v
    M_pos = np.zeros((n, m), dtype=np.float32)
    M_neg = np.zeros((n, m), dtype=np.float32)
    
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            if clause_signs[i, j] == 1:
                M_pos[v, i] = 1
            else:
                M_neg[v, i] = 1

    rng = np.random.RandomState(42)
    assignment = rng.rand(n) < 0.5

    # Compute initial clause_sat
    # lit_sat[i, j] = (assignment[var] == (sign==1))
    a = assignment[clause_vars] # (m, 3)
    s_bool = (clause_signs == 1) # (m, 3)
    lit_sat = (a == s_bool)
    clause_sat = lit_sat.any(axis=1)
    
    unsat_count = np.sum(~clause_sat)
    if unsat_count == 0:
        return assignment.tolist()

    max_iters = 200000
    p_walk = 0.1 # Probability to do random walk if stuck? Or always use best gain?
    
    # We will use a hybrid: 
    # 1. Calculate gains.
    # 2. If max_gain > 0, pick best.
    # 3. Else, pick random unsat clause, random literal, flip.

    for _ in range(max_iters):
        if unsat_count == 0:
            break
            
        # Calculate gains
        unsat_mask = ~clause_sat
        sat_mask = clause_sat
        
        # unsat_counts_pos[v] = sum over c of M_pos[v,c] * unsat_mask[c]
        unsat_counts_pos = M_pos @ unsat_mask
        unsat_counts_neg = M_neg @ unsat_mask
        
        sat_counts_pos = M_pos @ sat_mask
        sat_counts_neg = M_neg @ sat_mask
        
        gain_true = unsat_counts_pos - sat_counts_neg
        gain_false = unsat_counts_neg - sat_counts_pos
        
        # Find best gain
        max_gain_true = np.max(gain_true)
        max_gain_false = np.max(gain_false)
        
        best_gain = -np.inf
        best_var = -1
        best_val = False
        
        if max_gain_true > max_gain_false:
            best_gain = max_gain_true
            candidates = np.where(gain_true == max_gain_true)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = True
        elif max_gain_false > max_gain_true:
            best_gain = max_gain_false
            candidates = np.where(gain_false == max_gain_false)[0]
            best_var = candidates[rng.randint(len(candidates))]
            best_val = False
        else:
            # Tie
            if rng.rand() < 0.5:
                best_gain = max_gain_true
                candidates = np.where(gain_true == max_gain_true)[0]
                best_var = candidates[rng.randint(len(candidates))]
                best_val = True
            else:
                best_gain = max_gain_false
                candidates = np.where(gain_false == max_gain_false)[0]
                best_var = candidates[rng.randint(len(candidates))]
                best_val = False

        # Escape mechanism
        if best_gain <= 0:
            # Pick random unsat clause
            unsat_indices = np.where(unsat_mask)[0]
            if len(unsat_indices) == 0:
                break
            ci = unsat_indices[rng.randint(len(unsat_indices))]
            # Pick random literal in this clause
            j = rng.randint(3)
            v = int(clause_vars[ci, j])
            best_var = v
            # Flip to satisfy this clause? Or random?
            # WalkSAT usually flips with prob p to satisfy, 1-p to flip.
            # Here, since we are stuck, let's just flip to satisfy this specific literal.
            # This ensures we reduce unsat count by at least 1 (unless it breaks others).
            # Actually, flipping a literal in an unsat clause to true makes that clause sat.
            # It might break other clauses, but it's a standard move.
            best_val = (clause_signs[ci, j] == 1)

        # Apply flip
        if best_val != assignment[best_var]:
            assignment[best_var] = best_val
            
            # Update clause_sat for affected clauses
            affected = var_clauses[best_var]
            for ci, s in affected:
                # Check if clause ci is satisfied
                # We need to check all 3 literals
                is_sat = False
                for j in range(3):
                    v_c = int(clause_vars[ci, j])
                    s_c = (clause_signs[ci, j] == 1)
                    if assignment[v_c] == s_c:
                        is_sat = True
                        break
                old_sat = clause_sat[ci]
                clause_sat[ci] = is_sat
                if is_sat and not old_sat:
                    unsat_count -= 1
                elif not is_sat and old_sat:
                    unsat_count += 1

    return assignment.tolist()
