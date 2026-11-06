"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 280  # 300s limit, 20s safety margin

    # Precompute: for each variable, list of (clause_idx, literal_value) where
    # literal_value is True if the clause contains +var, False if -var.
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    def unit_propagation(assignment):
        """Fast unit propagation. Returns (assignment, contradiction)."""
        changed = True
        while changed:
            changed = False
            for ci in range(m):
                clause = clauses[ci]
                open_lits = []
                sat = False
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
                if len(open_lits) == 0:
                    return assignment, True
                if len(open_lits) == 1:
                    lit = open_lits[0]
                    var = abs(lit) - 1
                    val = lit > 0
                    if var in assignment:
                        if assignment[var] != val:
                            return assignment, True
                    else:
                        assignment[var] = val
                        changed = True
        return assignment, False

    def quick_dpll(seed, time_budget=5):
        """Quick DPLL with unit propagation and a few random assignments."""
        rng = random.Random(seed)
        start_time = time.time()
        
        best_result = None
        best_unsat = float('inf')
        
        for restart in range(10):
            if time.time() - start_time > time_budget:
                break
            
            assignment = {}
            num_initial = min(n // 10, 15)
            for var in rng.sample(range(n), num_initial):
                assignment[var] = rng.random() < 0.5
            
            assignment, contradiction = unit_propagation(assignment)
            
            if contradiction:
                continue
            
            result = [assignment.get(var, rng.random() < 0.5) for var in range(n)]
            unsat_count = sum(1 for c in clauses if not any(
                result[abs(l)-1] == (l > 0) for l in c))
            
            if unsat_count == 0:
                return result
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_result = result[:]
        
        return best_result if best_result else [False] * n

    def walksat_with_weights(seed, time_budget):
        """WalkSAT with clause weighting and adaptive noise."""
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_list = [i for i in range(m) if sat_count[i] == 0]
        unsat_set = set(unsat_list)

        if not unsat_set:
            return assignment

        # Clause weights for weighted clause selection
        clause_weight = [1.0] * m
        weight_decay = 0.95
        weight_boost = 2.0

        # Adaptive noise: start higher, decrease over time
        base_noise = 0.4
        min_noise = 0.1
        noise_decay = 0.99995

        current_noise = base_noise
        no_improve_steps = 0
        best_unsat_so_far = len(unsat_set)
        best_assignment_so_far = assignment[:]

        for step in range(1000000):
            if time.time() > deadline - 2:
                break
            if not unsat_set:
                break

            # Adaptive noise
            current_noise = max(min_noise, base_noise * (noise_decay ** step))

            # Weighted random selection of unsatisfied clause
            # Use weighted sampling: pick from unsat_list with weights
            if len(unsat_set) <= 10:
                ci = rng.choice(list(unsat_set))
            else:
                # Weighted selection using cumulative weights
                unsat_items = list(unsat_set)
                weights = [clause_weight[c] for c in unsat_items]
                total_w = sum(weights)
                if total_w <= 0:
                    ci = rng.choice(unsat_items)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    ci = unsat_items[-1]
                    for idx, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            ci = unsat_items[idx]
                            break

            clause = clauses[ci]

            # Score each literal in the clause
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

            # With probability noise, pick random; else pick best
            if rng.random() < current_noise:
                _, var, want_val = rng.choice(candidates)
            else:
                best = max(candidates, key=lambda x: x[0])
                var, want_val = best[1], best[2]

            # Apply flip
            old_val = assignment[var]
            if old_val == want_val:
                continue
            assignment[var] = want_val

            # Update sat_counts for all clauses containing var
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == want_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                        # Decrease weight for now-satisfied clause
                        clause_weight[oc] *= weight_decay
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)
                        # Increase weight for newly-unsatisfied clause
                        clause_weight[oc] = min(clause_weight[oc] * weight_boost, 10.0)

            # Track best
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_so_far:
                best_unsat_so_far = current_unsat
                best_assignment_so_far = assignment[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # If stuck for too long, restart with fresh randomization
            if no_improve_steps > 5000:
                # Partially randomize
                for _ in range(n // 5):
                    v = rng.randint(0, n - 1)
                    assignment[v] = rng.random() < 0.5
                # Recompute sat_count
                sat_count = [0] * m
                for ci2, clause2 in enumerate(clauses):
                    for lit2 in clause2:
                        var2 = abs(lit2) - 1
                        if assignment[var2] == (lit2 > 0):
                            sat_count[ci2] += 1
                unsat_set = set(i for i in range(m) if sat_count[i] == 0)
                best_unsat_so_far = len(unsat_set)
                best_assignment_so_far = assignment[:]
                no_improve_steps = 0
                if not unsat_set:
                    break

        return best_assignment_so_far if best_unsat_so_far < len(unsat_set) else assignment

    # Phase 1: Quick DPLL check
    result = quick_dpll(0, time_budget=5)
    unsat_count = sum(1 for c in clauses if not any(
        result[abs(l)-1] == (l > 0) for l in c))
    if unsat_count == 0:
        return result

    best_assignment = result
    best_unsat = unsat_count

    # Phase 2: WalkSAT with weights - main solver
    # Run multiple walks with different seeds
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        result = walksat_with_weights(seed, deadline)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result

    # Phase 3: Final polish - try to fix remaining unsat clauses with targeted flips
    if best_unsat > 0 and best_unsat <= 20 and time.time() < deadline - 3:
        rng = random.Random(999)
        assignment = best_assignment[:]
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1
        unsat_set = set(i for i in range(m) if sat_count[i] == 0)
        
        for step in range(100000):
            if time.time() > deadline - 2:
                break
            if not unsat_set:
                break
            ci = rng.choice(list(unsat_set))
            clause = clauses[ci]
            # Try each literal, pick one that minimizes new unsat
            best_var, best_val, best_delta = -1, False, 100
            for lit in clause:
                var = abs(lit) - 1
                want_val = lit > 0
                if assignment[var] == want_val:
                    continue
                # Simulate flip
                delta = 0
                for (oc, lit_val) in var_occurrences[var]:
                    if lit_val == want_val:
                        if sat_count[oc] == 0:
                            delta -= 1
                    else:
                        if sat_count[oc] == 1:
                            delta += 1
                if delta < best_delta:
                    best_delta = delta
                    best_var = var
                    best_val = want_val
            if best_var == -1:
                break
            assignment[best_var] = best_val
            for (oc, lit_val) in var_occurrences[best_var]:
                if lit_val == best_val:
                    sat_count[oc] += 1
                    if sat_count[oc] == 1:
                        unsat_set.discard(oc)
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        unsat_set.add(oc)
        
        final_unsat = len(unsat_set)
        if final_unsat == 0:
            return assignment
        if final_unsat < best_unsat:
            best_unsat = final_unsat
            best_assignment = assignment

    return best_assignment
# EVOLVE-BLOCK-END
