"""Structural change: bitset-based SLS with frequency-weighted clause selection and long random walks."""
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill

    # Precompute literal arrays
    lit_idx = np.empty((m, 3), dtype=np.int32)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lit_idx[i, j] = abs(lit) - 1
            lit_sign[i, j] = 1 if lit > 0 else -1

    # Build variable-to-clause mapping
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_idx[i, j]
            var_clauses[v].append((i, j, lit_sign[i, j]))

    # Precompute arrays for each variable
    var_clause_idx = [np.array([c[0] for c in var_clauses[v]], dtype=np.int32) if var_clauses[v] else np.array([], dtype=np.int32) for v in range(n)]
    var_pos = [np.array([c[1] for c in var_clauses[v]], dtype=np.int8) if var_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]
    var_sign = [np.array([c[2] for c in var_clauses[v]], dtype=np.int8) if var_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]

    rng = np.random.default_rng(42)

    best_assignment = None
    best_unsat_count = m + 1

    def compute_sat_count(assignment):
        """Vectorized computation of sat_count for all clauses."""
        is_true = (assignment[lit_idx] == (lit_sign > 0))
        return is_true.sum(axis=1)

    def update_sat_count(v, old_val, sat_count):
        """Update sat_count after flipping variable v from old_val."""
        vc_idx = var_clause_idx[v]
        if len(vc_idx) == 0:
            return
        vs = var_sign[v]
        # Old literal truth: old_val == (vs > 0)
        old_lit = (old_val == (vs > 0))
        # delta = -1 where old_lit is True (literal becomes false), +1 where False
        delta = np.where(old_lit, -1, 1)
        sat_count[vc_idx] += delta
        sat_count = np.clip(sat_count, 0, 3)
        return sat_count

    # Bitset-based approach for fast unsat clause tracking
    # Use numpy bit operations for efficiency

    max_restarts = 200

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize assignment - random
        assignment = rng.random(n) < 0.5

        # Compute initial sat_count
        sat_count = compute_sat_count(assignment)
        unsat_count = int((sat_count == 0).sum())

        if unsat_count == 0:
            return assignment.tolist()

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()

        # Frequency-based clause selection: count how often each variable appears in unsat clauses
        # This allows prioritizing variables that are "stuck" in many unsat clauses

        # Adaptive parameters
        difficulty = unsat_count / m
        base_noise = 0.25 + 0.35 * difficulty
        noise = base_noise + 0.15 * (restart / max_restarts)
        noise = min(noise, 0.8)

        # Flip budget: scale with remaining time
        time_per_restart = (deadline - time.time()) / max(1, (max_restarts - restart))
        flip_budget = int(min(time_per_restart * 150000, 2000000))
        flip_budget = max(flip_budget, 3000)

        # Stagnation tracking
        stagnant_steps = 0
        local_best = unsat_count
        stagnant_limit = 1500

        # Track variable frequency in unsat clauses for weighted selection
        var_unsat_freq = np.zeros(n, dtype=np.int32)

        for step in range(flip_budget):
            if time.time() > deadline:
                break

            if unsat_count == 0:
                return assignment.tolist()

            # Track best and stagnation
            if unsat_count < local_best:
                local_best = unsat_count
                stagnant_steps = 0
            else:
                stagnant_steps += 1

            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()

            # Early restart if stuck too long
            if stagnant_steps > stagnant_limit:
                break

            # Find unsatisfied clauses
            unsat_clauses = np.where(sat_count == 0)[0]
            if len(unsat_clauses) == 0:
                break

            # Compute variable frequency in unsat clauses for weighted selection
            # Flatten all literals in unsat clauses
            unsat_lits = lit_idx[unsat_clauses].flatten()
            # Count frequency of each variable
            var_unsat_freq = np.zeros(n, dtype=np.int32)
            np.add.at(var_unsat_freq, unsat_lits, 1)

            # Select clause with probability inversely proportional to frequency of its variables
            # Variables with lower frequency are "less stuck" and more likely to help
            clause_weights = np.ones(len(unsat_clauses), dtype=np.float64)
            for ci_idx, ci in enumerate(unsat_clauses):
                # Weight = product of (1 + freq[v]) for each variable in clause, inverted
                w = 1.0
                for j in range(3):
                    v = lit_idx[ci, j]
                    w *= (1.0 + var_unsat_freq[v])
                clause_weights[ci_idx] = 1.0 / w

            # Normalize and select
            total_weight = clause_weights.sum()
            if total_weight > 0:
                probs = clause_weights / total_weight
                clause_idx = unsat_clauses[rng.choice(len(unsat_clauses), p=probs)]
            else:
                clause_idx = unsat_clauses[rng.integers(len(unsat_clauses))]

            c_idx = lit_idx[clause_idx]
            c_sign = lit_sign[clause_idx]

            # Compute scores for each of 3 variables
            scores = np.zeros(3, dtype=np.float64)
            for j in range(3):
                v = c_idx[j]
                vc_idx = var_clause_idx[v]
                if len(vc_idx) == 0:
                    scores[j] = 0
                    continue
                vs = var_sign[v]
                lit_true_arr = (assignment[v] == (vs > 0))
                if np.any(lit_true_arr):
                    scores[j] = float((sat_count[vc_idx[lit_true_arr]] == 1).sum())
                else:
                    scores[j] = 0

            # Apply noise
            if rng.random() < noise:
                flip_pos = rng.integers(3)
            else:
                min_score = scores.min()
                candidates = np.where(scores == min_score)[0]
                flip_pos = candidates[rng.integers(len(candidates))]

            v = c_idx[flip_pos]

            # Flip the variable
            old_val = assignment[v]
            assignment[v] = not old_val

            # Update sat_count
            sat_count = update_sat_count(v, old_val, sat_count)
            unsat_count = int((sat_count == 0).sum())

        # End of restart
        if unsat_count == 0:
            return assignment.tolist()

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
