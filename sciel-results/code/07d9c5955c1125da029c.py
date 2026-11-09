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
    m = len(clauses)

    start_time = time.time()
    deadline = start_time + 290.0

    # Build literal arrays
    lit_var = np.zeros(3 * m, dtype=np.int32)
    lit_sign = np.zeros(3 * m, dtype=np.int8)
    for ci, clause in enumerate(clauses):
        for li, lit in enumerate(clause):
            idx = ci * 3 + li
            lit_var[idx] = abs(lit) - 1
            lit_sign[idx] = 1 if lit > 0 else -1

    # For each variable, precompute the list of (clause_idx, sign) pairs
    var_info = [[] for _ in range(n)]
    for ci in range(m):
        for li in range(3):
            idx = ci * 3 + li
            v = lit_var[idx]
            var_info[v].append((ci, lit_sign[idx]))

    # Convert to numpy arrays per variable for speed
    var_clause_arrs = []
    var_sign_arrs = []
    for v in range(n):
        if var_info[v]:
            cidx = np.array([x[0] for x in var_info[v]], dtype=np.int32)
            sgn = np.array([x[1] for x in var_info[v]], dtype=np.int8)
        else:
            cidx = np.array([], dtype=np.int32)
            sgn = np.array([], dtype=np.int8)
        var_clause_arrs.append(cidx)
        var_sign_arrs.append(sgn)

    def check_all_sat(assignment):
        """Vectorized check: returns True if all clauses satisfied."""
        av = assignment[lit_var]
        desired = (lit_sign > 0).astype(np.int8)
        sat = (av == desired).reshape(m, 3).any(axis=1)
        return bool(np.all(sat))

    def get_unsat_clauses(assignment):
        av = assignment[lit_var]
        desired = (lit_sign > 0).astype(np.int8)
        sat = (av == desired).reshape(m, 3).any(axis=1)
        return np.where(~sat)[0]

    def walksat(assignment, max_steps, rng, random_prob=0.2):
        """WalkSAT with mixed greedy/random variable selection."""
        unsat = get_unsat_clauses(assignment)
        unsat_count = len(unsat)
        if unsat_count == 0:
            return assignment, True

        for step in range(max_steps):
            if unsat_count == 0:
                return assignment, True
            if time.time() > deadline - 3.0:
                return assignment, False

            # Pick random unsat clause
            ci = unsat[rng.randint(0, unsat_count - 1)]
            clause = clauses[ci]

            # Compute scores for each variable in clause
            # Score = (# unsat clauses that would be satisfied) - (# sat clauses that would break)
            best_v = -1
            best_score = -10**9
            best_val = None
            candidates = []

            for lit in clause:
                v = abs(lit) - 1
                desired_val = 1 if lit > 0 else 0
                cidx = var_clause_arrs[v]
                sgn = var_sign_arrs[v]
                if len(cidx) == 0:
                    candidates.append((0, v, desired_val))
                    continue

                # For setting v = desired_val:
                # A clause j containing v is satisfied by this setting if:
                #   (sgn[j] == 1 and desired_val == 1) or (sgn[j] == -1 and desired_val == 0)
                # i.e., sgn[j] * desired_val_sign > 0 where desired_val_sign = +1 if True, -1 if False
                dv_sign = 1 if desired_val else -1
                matches = (sgn * dv_sign) > 0  # bool array: which of v's clauses are satisfied by this setting

                # Gain: unsat clauses among v's clauses that match
                gain_mask = matches & (np.isin(cidx, unsat))
                gain = int(np.sum(gain_mask))

                # Loss: currently sat clauses among v's clauses that do NOT match
                # A clause is currently sat if any of its 3 lits is sat
                # This is complex; approximate: loss = # of v's clauses that are currently sat AND don't match
                # For speed, just use gain as primary signal
                score = gain
                candidates.append((score, v, desired_val))
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_val = desired_val

            # Select: with probability random_prob, pick random; else pick best
            if rng.random() < random_prob:
                idx = rng.randint(0, len(candidates) - 1)
            else:
                # Pick from top candidates with some randomness
                max_s = max(c[0] for c in candidates)
                top = [c for c in candidates if c[0] >= max_s - 1]
                idx = rng.randint(0, len(top) - 1)
                # Need to map back... just pick from candidates directly
                # Simpler: weighted by score
                weights = [c[0] + 0.5 for c in candidates]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                idx = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        idx = i
                        break
                idx = min(idx, len(candidates) - 1)

            _, v, desired_val = candidates[idx]
            assignment[v] = desired_val

            # Update unsat set incrementally
            # Only clauses containing v can change status
            affected = var_clause_arrs[v]
            if len(affected) == 0:
                continue

            old_unsat_set = set(unsat.tolist())
            new_unsat = set()
            for aci in affected:
                c = clauses[aci]
                sat = False
                for lit in c:
                    av = abs(lit) - 1
                    if assignment[av] == (1 if lit > 0 else 0):
                        sat = True
                        break
                if not sat:
                    new_unsat.add(int(aci))

            # Update count
            for aci in old_unsat_set:
                if int(aci) not in new_unsat:
                    unsat_count -= 1
            for aci in new_unsat:
                if int(aci) not in old_unsat_set:
                    unsat_count += 1

            unsat = np.array(list(new_unsat), dtype=np.int32) if new_unsat else np.array([], dtype=np.int32)

        return assignment, (unsat_count == 0)

    best_assignment = None
    best_unsat = float('inf')

    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 2718, 1618, 9999, 12345, 54321, 11111, 22222, 33333]
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(seed)
        assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=np.int8)

        # Phase 1: greedy-ish walk
        assignment, solved = walksat(assignment, 8000, rng, random_prob=0.15)
        if solved:
            return [bool(x) for x in assignment]

        cur_unsat = len(get_unsat_clauses(assignment))
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment.copy()

        # Phase 2: perturb and walk with more randomness
        if time.time() < deadline - 10.0:
            rng2 = random.Random(seed + 777)
            for _ in range(n // 4):
                v = rng2.randint(0, n - 1)
                assignment[v] = 1 - assignment[v]
            assignment, solved2 = walksat(assignment, 5000, rng2, random_prob=0.35)
            if solved2:
                return [bool(x) for x in assignment]
            cur_unsat2 = len(get_unsat_clauses(assignment))
            if cur_unsat2 < best_unsat:
                best_unsat = cur_unsat2
                best_assignment = assignment.copy()

    if best_assignment is not None:
        return [bool(x) for x in best_assignment]

    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
