"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290  # 300s limit, 10s safety margin

    # Precompute: for each variable, list of (clause_idx, literal_value) where
    # literal_value is True if the clause contains +var, False if -var.
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    def dpll_with_unit_prop(seed, time_budget=10):
        """DPLL with unit propagation and randomized restarts."""
        rng = random.Random(seed)
        start_time = time.time()

        def propagate(clauses_list, assignment):
            """Unit propagation. Returns (new_assignment, new_clauses_list, contradiction)"""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses_list:
                    sat = False
                    unit_lit = None
                    open_lits = []
                    for lit in clause:
                        var = abs(lit) - 1
                        if var in assignment:
                            val = assignment[var]
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                        else:
                            open_lits.append(lit)

                    if sat:
                        continue
                    elif len(open_lits) == 0:
                        return assignment, clauses_list, True
                    elif len(open_lits) == 1:
                        unit_lit = open_lits[0]
                        var = abs(unit_lit) - 1
                        val = unit_lit > 0
                        if var in assignment:
                            if assignment[var] != val:
                                return assignment, clauses_list, True
                        else:
                            assignment[var] = val
                            changed = True
                        continue
                    else:
                        new_clauses.append(open_lits)

            return assignment, new_clauses, False

        best_result = None
        best_unsat = float('inf')

        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break

            assignment = {}
            num_initial = min(n // 5, 30)
            for var in rng.sample(range(n), num_initial):
                assignment[var] = rng.random() < 0.5

            clauses_copy = [list(c) for c in clauses]
            assignment, clauses_copy, contradiction = propagate(clauses_copy, assignment)

            if contradiction:
                continue

            if not clauses_copy:
                result = [False] * n
                for var in range(n):
                    result[var] = assignment.get(var, rng.random() < 0.5)
                return result

            result = [assignment.get(var, rng.random() < 0.5) for var in range(n)]
            unsat_count = 0
            for clause in clauses_copy:
                if not any(result[abs(l)-1] == (l > 0) for l in clause):
                    unsat_count += 1

            if unsat_count == 0:
                return result

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_result = result[:]

        return best_result if best_result else [False] * n

    def random_walksat(seed, init_assignment=None, noise=0.3, max_steps=500000):
        """Vectorized WalkSAT with adaptive noise and clause-frequency-weighted init."""
        rng = random.Random(seed)
        if init_assignment is not None:
            assignment = np.array(init_assignment, dtype=bool)
        else:
            # Weighted initialization: variables appearing in more unsatisfied clauses
            # get biased toward values that satisfy more of them
            assignment = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)

        # Precompute numpy arrays for fast updates
        # clause_lits: (m, 3) array of literal indices (0-based var, sign encoded)
        # For each clause position, store (var_idx, is_positive)
        clause_vars = np.zeros((m, 3), dtype=np.int32)
        clause_pos = np.zeros((m, 3), dtype=np.int8)  # 1 if positive, 0 if negative
        for ci, clause in enumerate(clauses):
            for pi, lit in enumerate(clause):
                clause_vars[ci, pi] = abs(lit) - 1
                clause_pos[ci, pi] = 1 if lit > 0 else 0

        # sat_count[ci] = number of satisfied literals in clause ci
        # A literal is satisfied if (clause_pos[ci,pi] == 1 and assignment[var]) or (clause_pos[ci,pi] == 0 and not assignment[var])
        # Equivalent: assignment[var] == (clause_pos[ci,pi] == 1)
        sat_count = np.zeros(m, dtype=np.int32)
        for ci in range(m):
            for pi in range(3):
                v = clause_vars[ci, pi]
                p = clause_pos[ci, pi]
                if assignment[v] == (p == 1):
                    sat_count[ci] += 1

        unsat_mask = (sat_count == 0)
        unsat_count = int(np.sum(unsat_mask))

        if unsat_count == 0:
            return assignment.tolist()

        start_time = time.time()
        prev_unsat = unsat_count
        stall_counter = 0

        # Precompute for each variable: list of (clause_idx, position_in_clause)
        # Already have var_occurrences as (ci, lit_val) where lit_val is bool
        # We need (ci, pi) for numpy updates
        var_clause_pos = [[] for _ in range(n)]
        for ci, clause in enumerate(clauses):
            for pi, lit in enumerate(clause):
                v = abs(lit) - 1
                var_clause_pos[v].append((ci, pi))

        # Convert to numpy-friendly format: for each var, arrays of clause indices and positions
        var_ci_arr = [np.array([x[0] for x in vcp], dtype=np.int32) if vcp else np.array([], dtype=np.int32) for vcp in var_clause_pos]
        var_pi_arr = [np.array([x[1] for x in vcp], dtype=np.int32) if vcp else np.array([], dtype=np.int32) for vcp in var_clause_pos]
        var_p_arr = [np.array([clause_pos[ci, pi] for ci, pi in vcp], dtype=np.int8) if vcp else np.array([], dtype=np.int8) for vcp in var_clause_pos]

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if unsat_count == 0:
                break

            # Adaptive noise
            if unsat_count >= prev_unsat:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_unsat = unsat_count

            adaptive_noise = noise + min(0.4, stall_counter * 0.005)

            # Pick random unsatisfied clause
            unsat_indices = np.where(unsat_mask)[0]
            ci = int(rng.choice(unsat_indices))

            # Get the 3 literals in this clause
            v0, p0 = int(clause_vars[ci, 0]), int(clause_pos[ci, 0])
            v1, p1 = int(clause_vars[ci, 1]), int(clause_pos[ci, 1])
            v2, p2 = int(clause_vars[ci, 2]), int(clause_pos[ci, 2])

            # Compute gain/loss for flipping each variable
            # If we flip var v to value want_val:
            #   For other clauses containing v: if lit_val matches want_val, it becomes satisfied (gain if was unsat)
            #   If lit_val doesn't match want_val, it becomes unsatisfied (loss if was sat with count 1)
            def flip_score(v, want_val):
                ci_arr = var_ci_arr[v]
                pi_arr = var_pi_arr[v]
                p_arr = var_p_arr[v]
                if len(ci_arr) == 0:
                    return 0
                # Exclude current clause
                mask = ci_arr != ci
                if not np.any(mask):
                    return 0
                ci_f = ci_arr[mask]
                p_f = p_arr[mask]
                # want_val is the value we want var to take
                # A literal in clause oc is satisfied by var if p_f == want_val
                # Gain: clauses that are currently unsat (sat_count==0) and this literal would satisfy them
                # Loss: clauses that are currently sat with count 1 and this literal is the only one satisfying them
                # For gain: p_f == want_val and sat_count[ci_f] == 0
                # For loss: p_f != want_val and sat_count[ci_f] == 1
                gain = int(np.sum((p_f == want_val) & (sat_count[ci_f] == 0)))
                loss = int(np.sum((p_f != want_val) & (sat_count[ci_f] == 1)))
                return gain - loss

            s0 = flip_score(v0, p0 == 1)
            s1 = flip_score(v1, p1 == 1)
            s2 = flip_score(v2, p2 == 1)

            scores = np.array([s0, s1, s2])
            if rng.random() < adaptive_noise:
                idx = rng.randint(0, 2)
            else:
                idx = int(np.argmax(scores))

            if idx == 0:
                var, want_val = v0, (p0 == 1)
            elif idx == 1:
                var, want_val = v1, (p1 == 1)
            else:
                var, want_val = v2, (p2 == 1)

            old_val = assignment[var]
            if old_val == want_val:
                continue
            assignment[var] = want_val

            # Update sat_count for all clauses containing var
            ci_arr = var_ci_arr[var]
            p_arr = var_p_arr[var]
            if len(ci_arr) > 0:
                # Clauses where this literal is satisfied: p_arr == want_val
                # Clauses where this literal is not satisfied: p_arr != want_val
                sat_by_var = (p_arr == want_val)
                # For clauses where var now satisfies: sat_count increases
                # For clauses where var no longer satisfies: sat_count decreases
                inc = sat_by_var
                dec = ~sat_by_var
                if np.any(inc):
                    sat_count[ci_arr[inc]] += 1
                    # Check if newly satisfied
                    newly_sat = (sat_count[ci_arr[inc]] == 1)
                    if np.any(newly_sat):
                        unsat_mask[ci_arr[inc][newly_sat]] = False
                        unsat_count -= int(np.sum(newly_sat))
                if np.any(dec):
                    sat_count[ci_arr[dec]] -= 1
                    # Check if newly unsatisfied
                    newly_unsat = (sat_count[ci_arr[dec]] == 0)
                    if np.any(newly_unsat):
                        unsat_mask[ci_arr[dec][newly_unsat]] = True
                        unsat_count += int(np.sum(newly_unsat))

            if unsat_count == 0:
                break

        return assignment.tolist()

    # Phase 1: DPLL with unit propagation
    result = dpll_with_unit_prop(0, time_budget=30)
    unsat_count = sum(1 for c in clauses if not any(
        result[abs(l)-1] == (l > 0) for l in c))
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 2: WalkSAT from DPLL result with adaptive noise
    if time.time() < deadline - 10:
        result = random_walksat(0, init_assignment=best_assignment, noise=0.25, max_steps=1000000)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 3: More DPLL restarts
    for seed in range(1, 20):
        if time.time() > deadline - 5:
            break
        result = dpll_with_unit_prop(seed, time_budget=5)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 4: Multiple WalkSAT restarts from random and best assignments
    for seed in range(1, 100):
        if time.time() > deadline - 3:
            break
        # Alternate between random init and best assignment init
        if seed % 3 == 0:
            result = random_walksat(seed, init_assignment=best_assignment, noise=0.15, max_steps=400000)
        elif seed % 3 == 1:
            result = random_walksat(seed, init_assignment=None, noise=0.3, max_steps=400000)
        else:
            result = random_walksat(seed, init_assignment=best_assignment, noise=0.35, max_steps=400000)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5: Final polish - local search with low noise from best assignment
    if time.time() < deadline - 1 and best_unsat > 0:
        # Try several low-noise polishes with different seeds
        for polish_seed in [999, 1000, 1001, 1002, 1003]:
            if time.time() > deadline - 1:
                break
            result = random_walksat(polish_seed, init_assignment=best_assignment, noise=0.05, max_steps=300000)
            unsat_count = sum(1 for c in clauses if not any(
                result[abs(l)-1] == (l > 0) for l in c))
            if unsat_count == 0:
                return result
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = result

    return best_assignment
# EVOLVE-BLOCK-END
