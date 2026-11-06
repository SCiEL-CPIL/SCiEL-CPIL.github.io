"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import deque

sys.setrecursionlimit(10000)


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 285  # 300s limit, 15s safety margin

    # Precompute: for each variable, list of (clause_idx, literal_value) where
    # literal_value is True if the clause contains +var, False if -var.
    var_occurrences = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            var_occurrences[var].append((ci, lit > 0))

    def dpll_solve():
        """DPLL with watched literals and unit propagation."""
        # assignment[v] = -1 (unassigned), 0 (false), 1 (true)
        assignment = [-1] * n
        
        # sat_count[ci] = number of satisfied literals in clause ci
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (1 if lit > 0 else 0):
                    sat_count[ci] += 1
        
        # Unit propagation queue
        up_queue = deque()
        
        # Check for initially unit clauses
        for ci in range(m):
            if sat_count[ci] == 0:
                clause = clauses[ci]
                unassigned = []
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] == -1:
                        unassigned.append(lit)
                if len(unassigned) == 1:
                    up_queue.append(unassigned[0])
                elif len(unassigned) == 0:
                    return None  # Conflict
        
        # Process unit propagation
        while up_queue:
            if time.time() > deadline:
                return None
            
            lit = up_queue.popleft()
            var = abs(lit) - 1
            want_val = 1 if lit > 0 else 0
            
            if assignment[var] != -1:
                if assignment[var] != want_val:
                    return None  # Conflict
                continue
            
            assignment[var] = want_val
            
            # Update sat_counts
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == (want_val == 1):
                    sat_count[oc] += 1
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        clause = clauses[oc]
                        unassigned = []
                        for l in clause:
                            v = abs(l) - 1
                            if assignment[v] == -1:
                                unassigned.append(l)
                        if len(unassigned) == 1:
                            up_queue.append(unassigned[0])
                        elif len(unassigned) == 0:
                            return None  # Conflict
        
        # Check if all clauses are satisfied
        if all(sat_count[ci] > 0 for ci in range(m)):
            result = []
            for v in range(n):
                if assignment[v] == -1:
                    result.append(random.random() < 0.5)
                else:
                    result.append(assignment[v] == 1)
            return result
        
        # Backtracking search
        # Find an unsatisfied clause to branch on
        unsat_clauses = [ci for ci in range(m) if sat_count[ci] == 0]
        if not unsat_clauses:
            result = []
            for v in range(n):
                if assignment[v] == -1:
                    result.append(random.random() < 0.5)
                else:
                    result.append(assignment[v] == 1)
            return result
        
        # Try each literal in the first unsatisfied clause
        ci = unsat_clauses[0]
        clause = clauses[ci]
        
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] != -1:
                continue
            
            want_val = 1 if lit > 0 else 0
            
            # Save state
            saved_assignment = assignment[:]
            saved_sat_count = sat_count[:]
            
            # Assign and propagate
            assignment[var] = want_val
            up_queue2 = deque()
            
            conflict = False
            for (oc, lit_val) in var_occurrences[var]:
                if lit_val == (want_val == 1):
                    sat_count[oc] += 1
                else:
                    sat_count[oc] -= 1
                    if sat_count[oc] == 0:
                        clause_oc = clauses[oc]
                        unassigned = []
                        for l in clause_oc:
                            v = abs(l) - 1
                            if assignment[v] == -1:
                                unassigned.append(l)
                        if len(unassigned) == 1:
                            up_queue2.append(unassigned[0])
                        elif len(unassigned) == 0:
                            conflict = True
                            break
            
            if conflict:
                assignment = saved_assignment
                sat_count = saved_sat_count
                continue
            
            # Continue propagation
            while up_queue2:
                lit2 = up_queue2.popleft()
                var2 = abs(lit2) - 1
                want_val2 = 1 if lit2 > 0 else 0
                
                if assignment[var2] != -1:
                    if assignment[var2] != want_val2:
                        conflict = True
                        break
                    continue
                
                assignment[var2] = want_val2
                
                for (oc, lit_val) in var_occurrences[var2]:
                    if lit_val == (want_val2 == 1):
                        sat_count[oc] += 1
                    else:
                        sat_count[oc] -= 1
                        if sat_count[oc] == 0:
                            clause_oc = clauses[oc]
                            unassigned = []
                            for l in clause_oc:
                                v = abs(l) - 1
                                if assignment[v] == -1:
                                    unassigned.append(l)
                            if len(unassigned) == 1:
                                up_queue2.append(unassigned[0])
                            elif len(unassigned) == 0:
                                conflict = True
                                break
                    if conflict:
                        break
                
                if conflict:
                    break
            
            if not conflict:
                if all(sat_count[oc] > 0 for oc in range(m)):
                    result = []
                    for v in range(n):
                        if assignment[v] == -1:
                            result.append(random.random() < 0.5)
                        else:
                            result.append(assignment[v] == 1)
                    return result
        
        return None

    def random_walksat(seed, noise=0.3, max_steps=500000):
        """WalkSAT with clause-count tracking."""
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # sat_count[c] = number of satisfied literals in clause c
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        if not unsat_set:
            return assignment

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))
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

            if rng.random() < noise:
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

    def repair_phase(assignment, max_iters=10000):
        """Targeted repair: flip variables in unsatisfied clauses."""
        sat_count = [0] * m
        for ci, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat_count[ci] += 1

        unsat_set = set(i for i in range(m) if sat_count[i] == 0)

        if not unsat_set:
            return assignment

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if not unsat_set:
                break

            # Pick a random unsatisfied clause
            ci = random.choice(list(unsat_set))
            clause = clauses[ci]

            # Find the variable in this clause that, when flipped, maximizes gain
            best_var = -1
            best_gain = -1
            best_val = None

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
                net = gain - loss
                if net > best_gain:
                    best_gain = net
                    best_var = var
                    best_val = want_val

            if best_var == -1:
                break

            if assignment[best_var] == best_val:
                continue

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

            if not unsat_set:
                break

        return assignment

    # First try DPLL
    result = dpll_solve()
    if result is not None:
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return result

    # Fall back to WalkSAT with multiple seeds and adaptive noise
    best_assignment = None
    best_unsat = float('inf')

    if result is not None:
        best_assignment = result
        best_unsat = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))

    # Try WalkSAT with varying noise levels
    noise_levels = [0.2, 0.3, 0.4, 0.5, 0.35, 0.25]
    
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        noise = noise_levels[seed % len(noise_levels)]
        result = random_walksat(seed, noise=noise)
        unsat_count = sum(1 for c in clauses if not any(
            result[abs(l)-1] == (l > 0) for l in c))
        
        if unsat_count == 0:
            return result
        
        if best_assignment is None or unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = result
    
    # Final repair phase on the best assignment found
    if best_assignment is not None and best_unsat > 0:
        repaired = repair_phase(best_assignment[:], max_iters=50000)
        unsat_count = sum(1 for c in clauses if not any(
            repaired[abs(l)-1] == (l > 0) for l in c))
        if unsat_count == 0:
            return repaired
        if unsat_count < best_unsat:
            best_assignment = repaired
            best_unsat = unsat_count

    return best_assignment
# EVOLVE-BLOCK-END
