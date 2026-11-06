import random
import time
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    # 300s limit, keep 10s margin for safety
    deadline = time.time() + 290 

    if m == 0:
        return [False] * n

    # Precompute variable occurrences as two parallel lists for speed
    # var_occ_clauses[var] = list of clause indices
    # var_occ_pos[var] = list of booleans (True if positive literal)
    var_occ_clauses = [[] for _ in range(n)]
    var_occ_pos = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occ_clauses[var].append(ci)
            var_occ_pos[var].append(lit > 0)

    def count_unsatisfied(assignment):
        count = 0
        for c in clauses:
            sat = False
            for lit in c:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

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

        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break

            assignment = {}
            # Randomize initial assignment slightly to break symmetry
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

            # If DPLL doesn't finish, return a partial assignment filled with randoms
            result = [assignment.get(var, rng.random() < 0.5) for var in range(n)]
            unsat_count = count_unsatisfied(result)
            if unsat_count == 0:
                return result
            return result

        return [False] * n

    def walksat(seed, init_assignment=None, noise=0.3, max_steps=500000):
        """
        Optimized WalkSAT.
        Maintains sat_count for each clause.
        Flips variable in a random unsatisfied clause.
        Variable selection: with prob 'noise' random, else max gain.
        """
        rng = random.Random(seed)
        
        if init_assignment is not None:
            assignment = init_assignment[:]
        else:
            assignment = [rng.random() < 0.5 for _ in range(n)]

        # Initialize satisfaction counts
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
        best_unsat = len(unsat_set)
        best_assignment = assignment[:]

        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if not unsat_set:
                break

            current_unsat = len(unsat_set)
            if current_unsat >= prev_unsat:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_unsat = current_unsat

            # Adaptive noise: increase when stalled
            adaptive_noise = noise + min(0.4, stall_counter * 0.01)

            # Select a random unsatisfied clause using list conversion
            unsat_list = list(unsat_set)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[ci]

            # Evaluate flips for variables in this clause
            candidates = []
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0
                
                # Check impact on other clauses using precomputed parallel lists
                occ_clauses = var_occ_clauses[var]
                occ_pos = var_occ_pos[var]
                for idx in range(len(occ_clauses)):
                    oc = occ_clauses[idx]
                    if oc == ci:
                        continue
                    lit_val = occ_pos[idx]
                    if lit_val == want_val:
                        # Flipping var to want_val satisfies this occurrence in clause oc
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        # Flipping var to want_val unsatisfies this occurrence in clause oc
                        if sat_count[oc] == 1:
                            loss += 1
                
                score = gain - loss
                candidates.append((score, var, want_val))

            # Choose flip
            if rng.random() < adaptive_noise:
                _, var, want_val = rng.choice(candidates)
            else:
                # Pick max score
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]

            old_val = assignment[var]
            if old_val == want_val:
                continue

            assignment[var] = want_val

            # Update sat_counts using precomputed parallel lists
            occ_clauses = var_occ_clauses[var]
            occ_pos = var_occ_pos[var]
            for idx in range(len(occ_clauses)):
                oc = occ_clauses[idx]
                lit_val = occ_pos[idx]
                if lit_val == want_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                return assignment
            
            # Track best
            if len(unsat_set) < best_unsat:
                best_unsat = len(unsat_set)
                best_assignment = assignment[:]

        return best_assignment

    def targeted_repair(assignment, max_rounds=1000):
        """
        Targeted repair: only flip variables involved in currently unsatisfied clauses.
        Uses greedy min-conflict approach with limited restarts.
        """
        rng = random.Random(42)
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1
        
        unsat_set = set(i for i in range(m) if sat_count[i] == 0)
        
        if not unsat_set:
            return assignment

        best_unsat = len(unsat_set)
        best_assignment = assignment[:]

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            unsat_list = list(unsat_set)
            ci = unsat_list[rng.randrange(len(unsat_list))]
            clause = clauses[ci]

            # Find best flip among variables in this clause
            best_score = -999
            best_var = -1
            best_val = False

            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                gain = 0
                loss = 0

                occ_clauses = var_occ_clauses[var]
                occ_pos = var_occ_pos[var]
                for idx in range(len(occ_clauses)):
                    oc = occ_clauses[idx]
                    if oc == ci:
                        continue
                    lit_val = occ_pos[idx]
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            gain += 1
                    else:
                        if sat_count[oc] == 1:
                            loss += 1

                score = gain - loss
                if score > best_score:
                    best_score = score
                    best_var = var
                    best_val = want_val

            if best_var < 0:
                break

            old_val = assignment[best_var]
            if old_val == best_val:
                continue

            assignment[best_var] = best_val

            # Update sat_counts
            occ_clauses = var_occ_clauses[best_var]
            occ_pos = var_occ_pos[best_var]
            for idx in range(len(occ_clauses)):
                oc = occ_clauses[idx]
                lit_val = occ_pos[idx]
                if lit_val == best_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)

            if not unsat_set:
                return assignment

            if len(unsat_set) < best_unsat:
                best_unsat = len(unsat_set)
                best_assignment = assignment[:]

        return best_assignment

    # Phase 1: DPLL
    result = dpll_with_unit_prop(0, time_budget=30)
    unsat_count = count_unsatisfied(result)
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 2: WalkSAT from DPLL result
    if time.time() < deadline - 15:
        result = walksat(0, init_assignment=best_assignment, noise=0.3, max_steps=500000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 2b: Targeted repair on best assignment
    if time.time() < deadline - 10 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=5000)
        unsat_count = count_unsatisfied(result)
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
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 4: Multiple WalkSAT restarts
    for seed in range(1, 50):
        if time.time() > deadline - 3:
            break
        if seed % 2 == 0:
            result = walksat(seed, init_assignment=best_assignment, noise=0.2, max_steps=300000)
        else:
            result = walksat(seed, init_assignment=None, noise=0.3, max_steps=300000)
        
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 5: Final targeted repair polish
    if time.time() < deadline - 1 and best_unsat > 0:
        result = targeted_repair(best_assignment, max_rounds=10000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 6: Final WalkSAT polish
    if time.time() < deadline - 1 and best_unsat > 0:
        result = walksat(999, init_assignment=best_assignment, noise=0.1, max_steps=200000)
        unsat_count = count_unsatisfied(result)
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    return best_assignment
