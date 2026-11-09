"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]

    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start = time.time()
    deadline = start + 295.0  # 35s margin

    rng = random.Random(0)

    m = len(clauses)
    
    # Precompute literal arrays
    lit_pos = np.empty((m, 3), dtype=np.int64)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, cl in enumerate(clauses):
        for j, l in enumerate(cl):
            lit_pos[i, j] = abs(l) - 1
            lit_sign[i, j] = 1 if l > 0 else -1

    # Precompute variable -> clause indices and signs for vectorized updates
    # var_to_clauses[v] = list of clause indices
    # var_to_clause_signs[v] = list of signs of v in those clauses
    var_to_clauses = [[] for _ in range(n)]
    var_to_clause_signs = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_pos[i, j]
            var_to_clauses[v].append(i)
            var_to_clause_signs[v].append(lit_sign[i, j])
    
    # Convert lists to numpy arrays for faster indexing
    var_to_clauses_arr = [np.array(lst, dtype=np.int64) for lst in var_to_clauses]
    var_to_clause_signs_arr = [np.array(lst, dtype=np.int8) for lst in var_to_clause_signs]

    def build_value(assignment):
        return np.where(np.asarray(assignment, dtype=bool), 1, -1).astype(np.int8)

    def calc_sat_counts(value):
        """Calculate number of satisfied literals in each clause."""
        # (m, 3) array of literal values: sign * value[var]
        lit_vals = lit_sign * value[lit_pos]
        # Count how many are 1 (satisfied) in each clause
        return np.sum(lit_vals == 1, axis=1).astype(np.int32)

    # Fast initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    value = build_value(assignment)
    sat_counts = calc_sat_counts(value)
    unsat_mask = (sat_counts == 0)
    cur_unsat = int(np.count_nonzero(unsat_mask))

    if cur_unsat == 0:
        return assignment

    best_assignment = None
    best_unsat = m + 1

    max_total_steps = 2000000
    noise = 0.35

    while time.time() < deadline:
        # Random restart assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        value = build_value(assignment)
        sat_counts = calc_sat_counts(value)
        unsat_mask = (sat_counts == 0)
        cur_unsat = int(np.count_nonzero(unsat_mask))

        if cur_unsat == 0:
            return assignment

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Light annealing: occasionally restart from best
        if best_unsat <= 2 and rng.random() < 0.3:
            assignment = best_assignment[:]
            value = build_value(assignment)
            sat_counts = calc_sat_counts(value)
            unsat_mask = (sat_counts == 0)
            cur_unsat = best_unsat

        for _ in range(max_total_steps):
            if time.time() >= deadline:
                break

            if cur_unsat == 0:
                return assignment

            # Pick a random unsatisfied clause
            unsat_indices = np.nonzero(unsat_mask)[0]
            if len(unsat_indices) == 0:
                return assignment
            ci = rng.choice(unsat_indices.tolist())

            # Compute gains for the 3 variables in clause ci
            gains = np.zeros(3, dtype=np.int64)
            for j in range(3):
                v = lit_pos[ci, j]
                # Count unsat clauses containing v, excluding ci
                # Vectorized sum over the subset of unsat_mask
                indices = var_to_clauses_arr[v]
                # Mask out ci to exclude it
                # Since ci is unsat, and we are summing unsat_mask, 
                # we just subtract 1 if ci is in indices (which it is)
                # np.sum(unsat_mask[indices]) includes ci.
                # We want count of OTHER unsat clauses.
                # So sum - 1.
                count = np.sum(unsat_mask[indices]) - 1
                gains[j] = count

            # Choose literal with max gain + noise
            if rng.random() < noise:
                j = rng.randrange(3)
            else:
                maxg = gains.max()
                candidates = np.nonzero(gains == maxg)[0]
                j = rng.choice(candidates.tolist())

            v = lit_pos[ci, j]
            
            # Update value and assignment
            old_val = value[v]
            new_val = -old_val
            value[v] = new_val
            assignment[v] = new_val > 0

            # Vectorized update of sat_counts and unsat_mask
            indices = var_to_clauses_arr[v]
            signs = var_to_clause_signs_arr[v]
            
            # Current contribution of v to these clauses
            # If sign * old_val == 1, it was satisfied (contrib 1)
            # If sign * old_val == -1, it was unsatisfied (contrib 0)
            old_contribs = (signs * old_val == 1).astype(np.int32)
            
            # New contribution
            # If sign * new_val == 1, it is satisfied (contrib 1)
            new_contribs = (signs * new_val == 1).astype(np.int32)
            
            deltas = new_contribs - old_contribs
            
            # Get old sat_counts for these clauses
            old_counts = sat_counts[indices].copy()
            
            # Update sat_counts
            sat_counts[indices] += deltas
            new_counts = sat_counts[indices]
            
            # Update cur_unsat
            became_unsat = (old_counts > 0) & (new_counts == 0)
            became_sat = (old_counts == 0) & (new_counts > 0)
            cur_unsat += int(np.sum(became_unsat)) - int(np.sum(became_sat))
            
            # Update unsat_mask
            unsat_mask[indices] = (new_counts == 0)

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
