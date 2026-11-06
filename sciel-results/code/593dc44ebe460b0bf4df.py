"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


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
        """WalkSAT with optional initial assignment and adaptive noise."""
        rng = random.Random(seed)
        if init_assignment is not None:
            assignment = init_assignment[:]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        if not unsat_set:
            return assignment

        start_time = time.time()
        prev_unsat = len(unsat_set)
        stall_counter = 0

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not unsat_set:
                break

            # Adaptive noise: increase noise when stalled
            current_unsat = len(unsat_set)
            if current_unsat >= prev_unsat:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_unsat = current_unsat

            # Increase noise when stalled to escape local minima
            adaptive_noise = noise + min(0.4, stall_counter * 0.01)

            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]

            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if oc == ci:
                        continue
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        if sat_count[oc] == 1:
                            loss += 1
                score = gain - loss
                candidates.append((score, var, want_val))

            if rng.random() < adaptive_noise:
                _, var, want_val = rng.choice(candidates)
            else:
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]

            old_val = assignment[var]
            if old_val == want_val:
                continue
            assignment[var] = want_val

            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == want_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                break

        return assignment

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
        result = random_walksat(0, init_assignment=best_assignment, noise=0.3, max_steps=500000)
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
    for seed in range(1, 50):
        if time.time() > deadline - 3:
            break
        # Alternate between random init and best assignment init
        if seed % 2 == 0:
            result = random_walksat(seed, init_assignment=best_assignment, noise=0.2, max_steps=300000)
        else:
            result = random_walksat(seed, init_assignment=None, noise=0.3, max_steps=300000)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5: Final polish - local search with low noise from best assignment
    if time.time() < deadline - 1 and best_unsat > 0:
        result = random_walksat(999, init_assignment=best_assignment, noise=0.1, max_steps=200000)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
# EVOLVE-BLOCK-END
