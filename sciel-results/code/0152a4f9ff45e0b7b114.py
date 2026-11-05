"""Improved 3-SAT solver with portfolio search: WalkSAT with restarts, clause weighting, and time budget."""
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Explicit deadline guard: 300s wall-clock limit, use 290s as safety margin
    start_time = time.time()
    TIME_LIMIT = 290.0
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    if time_remaining() <= 0:
        return [True] * n
    
    # Precompute clause literals as (var_index, is_positive) for speed
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    m = len(clause_data)
    
    # Precompute variable to clause index mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clause_data):
        for var, pos in clause:
            var_to_clauses[var].append(i)
    
    def is_satisfied(assignment, clause_idx):
        clause = clause_data[clause_idx]
        for var, pos in clause:
            if assignment[var] == pos:
                return True
        return False
        
    def count_unsat(assignment):
        cnt = 0
        for i in range(m):
            if not is_satisfied(assignment, i):
                cnt += 1
        return cnt
        
    def full_satisfied(assignment):
        for i in range(m):
            if not is_satisfied(assignment, i):
                return False
        return True
        
    def walksat_restarts(seed, max_restarts, steps_per_restart, p_min_conflicts=0.5):
        """WalkSAT with restarts, clause weighting, and phase saving."""
        rng = random.Random(seed)
        
        best_assignment = None
        best_unsat_count = float('inf')
        
        # Initialize with random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        last_assignment = assignment[:]
        
        # Track unsatisfied clauses
        unsat_clauses = set()
        for i in range(m):
            if not is_satisfied(assignment, i):
                unsat_clauses.add(i)
        
        current_unsat = len(unsat_clauses)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        if current_unsat == 0:
            return assignment[:]
        
        # Clause weights for weighted random selection
        clause_weights = [1.0] * m
        
        for restart in range(max_restarts):
            if time_remaining() < 1.0:
                break
            
            # Run WalkSAT for max_steps
            for step in range(steps_per_restart):
                if time_remaining() < 1.0:
                    break
                
                if not unsat_clauses:
                    return assignment[:]
                
                # Pick a random unsatisfied clause using weights
                unsat_list = list(unsat_clauses)
                weights_list = [clause_weights[i] for i in unsat_list]
                chosen_unsat_idx = rng.choices(unsat_list, weights=weights_list)[0]
                clause = clause_data[chosen_unsat_idx]
                
                # Decide flip strategy
                if rng.random() < p_min_conflicts:
                    # Min-Conflicts: Flip the variable in the clause that makes the fewest other clauses unsatisfied
                    best_var = -1
                    min_new_unsat = float('inf')
                    
                    for var, pos in clause:
                        newly_unsat = 0
                        for other_idx in var_to_clauses[var]:
                            if other_idx == chosen_unsat_idx:
                                continue
                            
                            # Check if other_clause is currently satisfied
                            if other_idx not in unsat_clauses:
                                # It is satisfied. Check if var is the only satisfying literal.
                                only_sat = True
                                for v2, p2 in clause_data[other_idx]:
                                    if v2 == var:
                                        continue
                                    if assignment[v2] == p2:
                                        only_sat = False
                                        break
                                if only_sat:
                                    newly_unsat += 1
                            
                        if newly_unsat < min_new_unsat:
                            min_new_unsat = newly_unsat
                            best_var = var
                            
                    if best_var != -1:
                        assignment[best_var] = not assignment[best_var]
                        flipped_var = best_var
                    else:
                        var, pos = rng.choice(clause)
                        assignment[var] = not assignment[var]
                        flipped_var = var
                else:
                    # Random flip from the clause
                    var, pos = rng.choice(clause)
                    assignment[var] = not assignment[var]
                    flipped_var = var
                
                # Update unsat_clauses incrementally
                prev_unsat_count = current_unsat
                for idx in var_to_clauses[flipped_var]:
                    if is_satisfied(assignment, idx):
                        unsat_clauses.discard(idx)
                    else:
                        unsat_clauses.add(idx)
                
                current_unsat = len(unsat_clauses)
                
                # Update clause weights
                # If the flip increased the number of unsatisfied clauses, increase weights of newly unsatisfied clauses
                if current_unsat > prev_unsat_count:
                    # Increase weight of the chosen clause and any clause that became unsatisfied
                    clause_weights[chosen_unsat_idx] *= 1.1
                    for idx in var_to_clauses[flipped_var]:
                        if idx in unsat_clauses and idx != chosen_unsat_idx:
                            clause_weights[idx] *= 1.1
                    # Normalize weights occasionally to prevent overflow
                    if step % 1000 == 0:
                        max_w = max(clause_weights)
                        if max_w > 1000:
                            clause_weights = [w / max_w for w in clause_weights]
                else:
                    # Decay weights slowly
                    # This is expensive if done every step, so maybe only do it on restart or periodically
                    pass

                # Update best
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                
                if current_unsat == 0:
                    return assignment[:]
            
            # End of this restart
            # Phase saving: Use last assignment as starting point, with some randomization
            if rng.random() < 0.7:
                assignment = last_assignment[:]
                # Flip a few random variables to diversify
                num_flips = rng.randint(0, 5)
                for _ in range(num_flips):
                    v = rng.randint(0, n - 1)
                    assignment[v] = not assignment[v]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]
            
            unsat_clauses = set()
            for i in range(m):
                if not is_satisfied(assignment, i):
                    unsat_clauses.add(i)
            current_unsat = len(unsat_clauses)
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            last_assignment = assignment[:]
            # Reset weights on restart to prevent bias from previous restart dominating
            clause_weights = [1.0] * m
        
        return best_assignment
    
    # Strategy 1: WalkSAT with restarts and clause weighting
    # Use multiple seeds to diversify
    best_result = None
    best_score = -1
    
    seeds_to_try = [42, 123, 777, 2024, 555, 999, 31337, 12345, 67890, 24680]
    
    for seed in seeds_to_try:
        if time_remaining() < 5.0:
            break
        
        # Adaptive parameters based on problem size
        steps_per_restart = max(1000, n * 20)
        max_restarts = max(5, min(50, int(100000 / steps_per_restart)))
        
        result = walksat_restarts(seed, max_restarts, steps_per_restart, p_min_conflicts=0.5)
        
        if result is not None:
            if full_satisfied(result):
                return result
            else:
                unsat_cnt = count_unsat(result)
                if unsat_cnt < best_score:
                    best_score = unsat_cnt
                    best_result = result
    
    # If we still have time, try a different strategy: more aggressive restarts
    if time_remaining() > 10.0:
        for seed in [111, 222, 333, 444, 555]:
            if time_remaining() < 5.0:
                break
            
            steps_per_restart = max(500, n * 10)
            max_restarts = max(10, min(100, int(200000 / steps_per_restart)))
            
            result = walksat_restarts(seed, max_restarts, steps_per_restart, p_min_conflicts=0.5)
            
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    unsat_cnt = count_unsat(result)
                    if unsat_cnt < best_score:
                        best_score = unsat_cnt
                        best_result = result
    
    # Final fallback: try one more set of seeds with even shorter steps
    if time_remaining() > 5.0:
        for seed in [77, 88, 99, 100]:
            if time_remaining() < 2.0:
                break
            
            steps_per_restart = max(200, n * 5)
            max_restarts = max(20, min(200, int(300000 / steps_per_restart)))
            
            result = walksat_restarts(seed, max_restarts, steps_per_restart, p_min_conflicts=0.5)
            
            if result is not None:
                if full_satisfied(result):
                    return result
                else:
                    unsat_cnt = count_unsat(result)
                    if unsat_cnt < best_score:
                        best_score = unsat_cnt
                        best_result = result
    
    # Return best found
    if best_result is not None:
        return best_result
    
    # Absolute fallback
    return [True] * n
