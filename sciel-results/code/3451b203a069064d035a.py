import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    timeout = 315.0  # safety margin before 330s kill

    m = len(clauses)
    if m == 0:
        return [False] * n

    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for clause in clauses:
        vars_list = []
        signs_list = []
        for lit in clause:
            var_idx = abs(lit) - 1
            vars_list.append(var_idx)
            signs_list.append(lit > 0)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)

    # Build var_to_clauses: for each variable, list of (clause_idx, sign_in_clause)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j, v in enumerate(clause_vars[i]):
            var_to_clauses[v].append((i, clause_signs[i][j]))

    # Precompute for each variable: array of clause indices it appears in
    # and corresponding signs, for faster damage calculation
    var_clause_idx = [np.array([ci for ci, s in var_to_clauses[v]], dtype=np.int32) for v in range(n)]
    var_clause_sign = [np.array([s for ci, s in var_to_clauses[v]], dtype=np.bool_) for v in range(n)]
    var_clause_count = [len(var_clause_idx[v]) for v in range(n)]

    def init_state(assignment):
        """Compute clause satisfaction counts using vectorized operations."""
        clause_sat_count = np.zeros(m, dtype=np.int32)
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_signs[i]
            sat = sum(1 for j, v in enumerate(cv) if assignment[v] == cs[j])
            clause_sat_count[i] = sat
        return clause_sat_count

    def is_satisfied(clause_sat_count):
        return bool(np.all(clause_sat_count > 0))

    def flip_var(v, assignment, clause_sat_count):
        """Flip variable v and update clause_sat_count efficiently."""
        old_val = assignment[v]
        assignment[v] = not old_val
        new_val = assignment[v]
        cidx = var_clause_idx[v]
        csign = var_clause_sign[v]
        # Vectorized update: where new_val == sign, increment; else decrement
        mask = (csign == new_val)
        clause_sat_count[cidx[mask]] += 1
        clause_sat_count[cidx[~mask]] -= 1

    def unit_propagate(assignment, clause_sat_count):
        """Set variables in unsatisfied clauses to satisfy them. Returns True if no contradiction."""
        changed = True
        max_rounds = 50
        rounds = 0
        while changed and rounds < max_rounds:
            changed = False
            rounds += 1
            unsat_mask = (clause_sat_count == 0)
            if not np.any(unsat_mask):
                break
            unsat_indices = np.where(unsat_mask)[0]
            for i in unsat_indices:
                if clause_sat_count[i] != 0:
                    continue
                # Pick variable with least damage
                best_v = clause_vars[i][0]
                best_damage = float('inf')
                for v in clause_vars[i]:
                    cidx = var_clause_idx[v]
                    other_mask = cidx != i
                    other_cidx = cidx[other_mask]
                    if len(other_cidx) == 0:
                        damage = 0
                    else:
                        damage = int(np.sum(clause_sat_count[other_cidx] > 0))
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                sign = None
                for j, v in enumerate(clause_vars[i]):
                    if v == best_v:
                        sign = clause_signs[i][j]
                        break
                if assignment[best_v] != sign:
                    flip_var(best_v, assignment, clause_sat_count)
                    changed = True
        return True

    def pure_literal(assignment, clause_sat_count):
        """Find and assign pure literals."""
        pos_count = np.zeros(n, dtype=np.int32)
        neg_count = np.zeros(n, dtype=np.int32)
        for i in range(m):
            for j, v in enumerate(clause_vars[i]):
                if clause_signs[i][j]:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1

        changed = False
        for v in range(n):
            if pos_count[v] > 0 and neg_count[v] == 0:
                if not assignment[v]:
                    flip_var(v, assignment, clause_sat_count)
                    changed = True
            elif neg_count[v] > 0 and pos_count[v] == 0:
                if assignment[v]:
                    flip_var(v, assignment, clause_sat_count)
                    changed = True
        return changed

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    def walk_sat(assignment, clause_sat_count, max_flips, noise, time_limit):
        """Run WalkSAT with given noise parameter. Returns (assignment, clause_sat_count, solved)."""
        for flip_num in range(max_flips):
            if time.time() - start_time > time_limit:
                return assignment, clause_sat_count, False

            # Find unsatisfied clauses
            unsat_count = int(np.sum(clause_sat_count == 0))
            if unsat_count == 0:
                return assignment, clause_sat_count, True

            # Pick a random unsatisfied clause
            unsat_indices = np.where(clause_sat_count == 0)[0]
            ci = int(unsat_indices[rng.randrange(len(unsat_indices))])

            # Get variables in this clause
            vars_in_clause = clause_vars[ci]

            # Decide: random flip or greedy flip
            if rng.random() < noise:
                # Random variable
                v = vars_in_clause[rng.randrange(len(vars_in_clause))]
            else:
                # Greedy: pick variable with minimum damage
                best_v = vars_in_clause[0]
                best_damage = float('inf')
                for v in vars_in_clause:
                    cidx = var_clause_idx[v]
                    other_mask = cidx != ci
                    other_cidx = cidx[other_mask]
                    if len(other_cidx) == 0:
                        damage = 0
                    else:
                        damage = int(np.sum(clause_sat_count[other_cidx] > 0))
                    if damage < best_damage:
                        best_damage = damage
                        best_v = v
                v = best_v

            flip_var(v, assignment, clause_sat_count)

        return assignment, clause_sat_count, False

    # Track best unsat count for adaptive parameter tuning
    min_unsat_seen = m + 1
    restarts = 0
    max_restarts = 500

    while time.time() - start_time < timeout - 5.0 and restarts < max_restarts:
        restarts += 1

        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat_count = init_state(assignment)

        if is_satisfied(clause_sat_count):
            return assignment

        # Pure literal elimination
        pure_literal(assignment, clause_sat_count)

        if is_satisfied(clause_sat_count):
            return assignment

        # Unit propagation
        unit_propagate(assignment, clause_sat_count)

        if is_satisfied(clause_sat_count):
            return assignment

        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            min_unsat_seen = current_unsat

        # Adaptive parameters based on progress
        # Use min_unsat_seen to adaptively tune noise and flip limits
        if min_unsat_seen <= 1:
            max_flips = 20000
            noise = 0.05
        elif min_unsat_seen <= 3:
            max_flips = 15000
            noise = 0.1
        elif min_unsat_seen <= 8:
            max_flips = 10000
            noise = 0.15
        elif min_unsat_seen <= 20:
            max_flips = 8000
            noise = 0.25
        else:
            max_flips = 6000
            noise = 0.35

        # Also adapt based on current_unsat for this restart
        if current_unsat <= 2:
            max_flips = max(max_flips, 15000)
            noise = min(noise, 0.1)
        elif current_unsat <= 5:
            max_flips = max(max_flips, 10000)
            noise = min(noise, 0.15)

        # Run WalkSAT
        assignment, clause_sat_count, solved = walk_sat(
            assignment, clause_sat_count, max_flips, noise, timeout - 5.0
        )

        if solved:
            return assignment

        current_unsat = int(np.sum(clause_sat_count == 0))
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment.copy()
            min_unsat_seen = current_unsat

    # Final polish: if we have a good candidate, try to refine it
    if best_assignment is not None and best_unsat_count <= 8:
        remaining = timeout - (time.time() - start_time)
        if remaining > 10.0:
            assignment = best_assignment.copy()
            clause_sat_count = init_state(assignment)
            # Try multiple short walks with very low noise
            for attempt in range(10):
                if time.time() - start_time > timeout - 2.0:
                    break
                assignment, clause_sat_count, solved = walk_sat(
                    assignment, clause_sat_count, 5000, 0.05, timeout - 2.0
                )
                if solved:
                    return assignment

    if best_assignment is not None:
        return best_assignment
    else:
        return [False] * n
