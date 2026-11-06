import time
import random
import math
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin

    # 1. Preprocessing: Convert to 0-based and signed
    # lit > 0 means variable is True, lit < 0 means variable is False
    # We store literals as (var_idx, sign) where sign is 1 for pos, -1 for neg
    clauses_0 = []
    for c in clauses:
        clause_0 = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_0.append((var, sign))
        clauses_0.append(clause_0)
    
    m = len(clauses_0)
    
    # 2. Preprocessing: Pure Literal Elimination + Unit Propagation
    # This can significantly reduce the problem size.
    
    var_assigned = [False] * n
    var_value = [False] * n
    
    # Count literal occurrences for pure literal elimination
    # (var, sign) -> count
    lit_count = {}
    for clause in clauses_0:
        for var, sign in clause:
            key = (var, sign)
            lit_count[key] = lit_count.get(key, 0) + 1

    # Pure Literal Elimination
    changed = True
    while changed:
        changed = False
        for var in range(n):
            if var_assigned[var]:
                continue
            pos_count = lit_count.get((var, 1), 0)
            neg_count = lit_count.get((var, -1), 0)
            if pos_count > 0 and neg_count == 0:
                var_assigned[var] = True
                var_value[var] = True
                changed = True
            elif neg_count > 0 and pos_count == 0:
                var_assigned[var] = True
                var_value[var] = False
                changed = True

    # Simplify clauses based on assignments
    simplified_clauses = []
    for clause in clauses_0:
        sat = False
        remaining = []
        for var, sign in clause:
            if var_assigned[var]:
                if var_value[var] == (sign > 0):
                    sat = True
                    break
                # else: literal is false, remove it
            else:
                remaining.append((var, sign))
        
        if sat:
            continue
        if len(remaining) == 0:
            # Empty clause -> UNSAT (should not happen for satisfiable planted)
            # If it does, we return a random assignment hoping for the best, 
            # but typically we just continue and let search fail.
            continue 
        simplified_clauses.append(remaining)

    # Unit Propagation
    changed = True
    while changed:
        changed = False
        new_clauses = []
        for clause in simplified_clauses:
            # Check for unit clause
            unassigned_lits = [(v, s) for v, s in clause if not var_assigned[v]]
            if len(unassigned_lits) == 0:
                # All assigned and not satisfied -> contradiction
                # In satisfiable instances this shouldn't happen if preprocessing is correct
                continue
            if len(unassigned_lits) == 1:
                v, s = unassigned_lits[0]
                if not var_assigned[v]:
                    var_assigned[v] = True
                    var_value[v] = (s > 0)
                    changed = True
                    # This clause is now satisfied, so we don't add it
                    continue
            
            # Filter out assigned literals
            filtered = []
            sat = False
            for v, s in clause:
                if var_assigned[v]:
                    if var_value[v] == (s > 0):
                        sat = True
                        break
                else:
                    filtered.append((v, s))
            
            if sat:
                continue
            if len(filtered) > 0:
                new_clauses.append(filtered)
            # if len(filtered) == 0 and not sat, it's an empty clause (UNSAT)
        
        simplified_clauses = new_clauses

    # If no clauses left, we have a solution
    if len(simplified_clauses) == 0:
        return var_value[:]

    # 3. Build Search Structures
    # Map unassigned variables to compact indices
    unassigned_vars = [v for v in range(n) if not var_assigned[v]]
    var_map = {v: i for i, v in enumerate(unassigned_vars)}
    n_search = len(unassigned_vars)
    
    if n_search == 0:
        return var_value[:]

    # Convert simplified clauses to search space
    search_clauses = []
    for clause in simplified_clauses:
        sc = []
        for v, s in clause:
            if not var_assigned[v]:
                sc.append((var_map[v], s))
        if len(sc) > 0:
            search_clauses.append(sc)
            
    m_search = len(search_clauses)
    if m_search == 0:
        return var_value[:]

    # Precompute variable to clause index for fast updates
    var_clauses = [[] for _ in range(n_search)]
    for ci, clause in enumerate(search_clauses):
        for v_idx, s in clause:
            var_clauses[v_idx].append(ci)

    # Helper to evaluate a clause
    def is_satisfied(clause_idx, assignment):
        clause = search_clauses[clause_idx]
        for v, s in clause:
            if assignment[v] == (s > 0):
                return True
        return False

    # Helper to compute initial unsat count
    def init_unsat(assignment):
        unsat = [False] * m_search
        count = 0
        for ci in range(m_search):
            if not is_satisfied(ci, assignment):
                unsat[ci] = True
                count += 1
        return unsat, count

    # 4. Search Strategies
    
    def run_walksat(rng, time_budget):
        """WalkSAT with adaptive noise."""
        end_time = time.time() + time_budget
        best_assign = None
        best_unsat = m_search + 1
        
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Random restart
            assignment = [rng.random() < 0.5 for _ in range(n_search)]
            unsat, unsat_count = init_unsat(assignment)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment[:]
            
            # Parameters
            noise = 0.5
            max_steps = 2000
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assignment[:]
                
                # Pick random unsatisfied clause
                unsat_indices = [i for i in range(m_search) if unsat[i]]
                if not unsat_indices:
                    break
                ci = unsat_indices[rng.randrange(len(unsat_indices))]
                clause = search_clauses[ci]
                
                # Find best variable to flip in this clause
                best_var = -1
                best_score = -float('inf')
                
                for v, s in clause:
                    # Calculate delta in unsat count if we flip v
                    delta = 0
                    for other_ci in var_clauses[v]:
                        if other_ci == ci:
                            continue
                        other_clause = search_clauses[other_ci]
                        if unsat[other_ci]:
                            # Flipping v might satisfy this clause
                            # Check if v is the only false literal? No, if clause is unsat, all are false.
                            # If we flip v, it becomes true, so clause becomes sat.
                            # So unsat count decreases by 1.
                            delta -= 1
                        else:
                            # Clause is sat. Check if v is the only satisfying literal.
                            # If yes, flipping v makes it unsat.
                            only_sat = True
                            for v2, s2 in other_clause:
                                if v2 == v:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                delta += 1
                    # Score: minimize delta, add noise
                    score = -delta + rng.random() * noise
                    if score > best_score:
                        best_score = score
                        best_var = v
                        
                if best_var == -1:
                    break
                    
                # Flip
                assignment[best_var] = not assignment[best_var]
                
                # Update unsat status for clauses containing best_var
                for other_ci in var_clauses[best_var]:
                    was_unsat = unsat[other_ci]
                    is_sat = is_satisfied(other_ci, assignment)
                    is_unsat = not is_sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    unsat[other_ci] = is_unsat
                    
                # Decay noise
                noise *= 0.995

        return best_assign

    def run_simulated_annealing(rng, time_budget):
        """Simulated Annealing."""
        end_time = time.time() + time_budget
        best_assign = None
        best_unsat = m_search + 1
        
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > end_time:
                break
            
            assignment = [rng.random() < 0.5 for _ in range(n_search)]
            unsat, unsat_count = init_unsat(assignment)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment[:]
            
            T = 1.0
            cooling = 0.99
            max_steps = 5000
            stagnation_limit = 500
            stagnation_count = 0
            
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assignment[:]
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > stagnation_limit:
                        break # Restart SA with new random state
            
                # Pick random variable
                v = rng.randrange(n_search)
                
                # Calculate delta
                delta = 0
                for ci in var_clauses[v]:
                    if unsat[ci]:
                        # If clause is unsat, flipping v makes one literal true.
                        # Clause becomes sat. Delta = -1
                        delta -= 1
                    else:
                        # Clause is sat. Check if v is the only sat literal.
                        clause = search_clauses[ci]
                        only_sat = True
                        for v2, s2 in clause:
                            if v2 == v:
                                continue
                            if assignment[v2] == (s2 > 0):
                                only_sat = False
                                break
                        if only_sat:
                            delta += 1
                
                # Acceptance
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    assignment[v] = not assignment[v]
                    
                    # Update unsat
                    for ci in var_clauses[v]:
                        was_unsat = unsat[ci]
                        is_sat = is_satisfied(ci, assignment)
                        is_unsat = not is_sat
                        if was_unsat and not is_unsat:
                            unsat_count -= 1
                        elif not was_unsat and is_unsat:
                            unsat_count += 1
                        unsat[ci] = is_unsat
                    
                    T *= cooling
                    if T < 0.01:
                        T = 0.01
                        
        return best_assign

    def run_min_conflicts(rng, time_budget):
        """Min-Conflicts heuristic."""
        end_time = time.time() + time_budget
        best_assign = None
        best_unsat = m_search + 1
        
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > end_time:
                break
            
            assignment = [rng.random() < 0.5 for _ in range(n_search)]
            unsat, unsat_count = init_unsat(assignment)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assign = assignment[:]
            
            max_steps = 10000
            for step in range(max_steps):
                if time.time() > end_time:
                    break
                
                if unsat_count == 0:
                    return assignment
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assign = assignment[:]
                
                # Find variable with max conflicts
                # We can approximate by picking a random unsat clause and picking a var in it,
                # but true min-conflicts counts conflicts per variable.
                # For speed, we'll pick a random unsat clause and then pick the var in it
                # that minimizes the delta (similar to WalkSAT but without noise).
                
                unsat_indices = [i for i in range(m_search) if unsat[i]]
                if not unsat_indices:
                    break
                ci = unsat_indices[rng.randrange(len(unsat_indices))]
                clause = search_clauses[ci]
                
                best_var = -1
                best_delta = float('inf')
                
                for v, s in clause:
                    delta = 0
                    for other_ci in var_clauses[v]:
                        if other_ci == ci:
                            continue
                        if unsat[other_ci]:
                            delta -= 1
                        else:
                            other_clause = search_clauses[other_ci]
                            only_sat = True
                            for v2, s2 in other_clause:
                                if v2 == v:
                                    continue
                                if assignment[v2] == (s2 > 0):
                                    only_sat = False
                                    break
                            if only_sat:
                                delta += 1
                    if delta < best_delta:
                        best_delta = delta
                        best_var = v
                        
                if best_var == -1:
                    break
                    
                assignment[best_var] = not assignment[best_var]
                
                for other_ci in var_clauses[best_var]:
                    was_unsat = unsat[other_ci]
                    is_sat = is_satisfied(other_ci, assignment)
                    is_unsat = not is_sat
                    if was_unsat and not is_unsat:
                        unsat_count -= 1
                    elif not was_unsat and is_unsat:
                        unsat_count += 1
                    unsat[other_ci] = is_unsat
                        
        return best_assign

    # 5. Execution Plan
    # Allocate time to different strategies.
    # 1. WalkSAT (fast, good for easy instances)
    # 2. Simulated Annealing (robust, good for hard instances)
    # 3. Min-Conflicts (alternative local search)
    
    # Check time
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining <= 0:
        return var_value[:]

    # Strategy 1: WalkSAT
    t_walk = min(10.0, remaining * 0.3)
    rng1 = random.Random(42)
    res1 = run_walksat(rng1, t_walk)
    
    # Verify res1
    if res1:
        final1 = var_value[:]
        for i, v in enumerate(unassigned_vars):
            final1[v] = res1[i]
        # Quick check
        is_sol = True
        for ci in range(m_search):
            clause = search_clauses[ci]
            sat = False
            for v, s in clause:
                if final1[unassigned_vars[v]] == (s > 0):
                    sat = True
                    break
            if not sat:
                is_sol = False
                break
        if is_sol:
            return final1

    # Strategy 2: Simulated Annealing
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 5:
        t_sa = min(remaining * 0.5, remaining - 5)
        rng2 = random.Random(123)
        res2 = run_simulated_annealing(rng2, t_sa)
        
        if res2:
            final2 = var_value[:]
            for i, v in enumerate(unassigned_vars):
                final2[v] = res2[i]
            is_sol = True
            for ci in range(m_search):
                clause = search_clauses[ci]
                sat = False
                for v, s in clause:
                    if final2[unassigned_vars[v]] == (s > 0):
                        sat = True
                        break
                if not sat:
                    is_sol = False
                    break
            if is_sol:
                return final2
            # If SA found a better unsat count, keep it as candidate
            # We'll just return the best found so far, which is res1 or res2
            # For now, if res2 is not a solution, we proceed to next strategy
            # but we should probably keep track of the best unsat assignment
            # However, the task asks for a satisfying assignment. If we don't find one,
            # we return the best effort (which might be unsat, scoring 0).
            # So we just try to find a sat one.

    # Strategy 3: Min-Conflicts
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining > 2:
        t_mc = remaining - 2
        rng3 = random.Random(999)
        res3 = run_min_conflicts(rng3, t_mc)
        
        if res3:
            final3 = var_value[:]
            for i, v in enumerate(unassigned_vars):
                final3[v] = res3[i]
            is_sol = True
            for ci in range(m_search):
                clause = search_clauses[ci]
                sat = False
                for v, s in clause:
                    if final3[unassigned_vars[v]] == (s > 0):
                        sat = True
                        break
                if not sat:
                    is_sol = False
                    break
            if is_sol:
                return final3

    # If no strategy found a solution, return the result of the first strategy (WalkSAT)
    # as it is the most standard fallback.
    if res1:
        final1 = var_value[:]
        for i, v in enumerate(unassigned_vars):
            final1[v] = res1[i]
        return final1
    
    return var_value[:]

# EVOLVE-BLOCK-END
