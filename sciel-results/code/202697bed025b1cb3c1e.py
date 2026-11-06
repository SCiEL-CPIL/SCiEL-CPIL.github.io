# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Precompute variable to clause mapping and literal signs
    var_to_clauses = [[] for _ in range(n)]
    clause_var_info = []
    # var_signs[v][c_idx] = sign of variable v in clause c_idx (0 if not present)
    var_signs = [{} for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        info = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            info.append((v, s))
            var_to_clauses[v].append(c_idx)
            var_signs[v][c_idx] = s
        clause_var_info.append(info)
    
    m = len(clauses)
    
    # Deadline: 325 seconds to be safe against the 330s kill
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    noise = 0.1  # Lower noise for more greedy steps
    restart_threshold = 10000
    weight_decay = 0.99  # Decay factor for clause weights
    weight_boost = 1.5   # Boost factor for unsatisfied clauses
    
    while time.time() < deadline:
        # Initialize random assignment with some bias
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial clause weights and track unsat clauses in a set
        clause_weights = [0] * m
        # Dynamic priority weights for weighted Max-SAT
        clause_priority = [1.0] * m
        unsat_set = set()
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_set.add(i)
                clause_priority[i] = weight_boost
        
        total_unsat = len(unsat_set)
        
        if total_unsat == 0:
            return assignment
        
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        step_count = 0
        
        for _ in range(100000): # Inner loop limit per restart
            step_count += 1
            # Check time every 100 steps to reduce overhead
            if step_count % 100 == 0 and time.time() >= deadline:
                break
            
            if total_unsat == 0:
                return assignment
            
            # Track best solution
            if total_unsat < best_unsat_count:
                best_unsat_count = total_unsat
                best_assignment = assignment[:]
            
            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > restart_threshold:
                break # Restart with new random assignment
            
            if not unsat_set:
                return assignment
            
            # Select clause with highest priority (weighted random)
            unsat_list = list(unsat_set)
            if len(unsat_list) > 1:
                # Weighted selection based on clause_priority
                weights = [clause_priority[c] for c in unsat_list]
                total_weight = sum(weights)
                if total_weight > 0:
                    r = rng.random() * total_weight
                    cumsum = 0
                    selected_idx = unsat_list[-1]
                    for idx, c in enumerate(unsat_list):
                        cumsum += weights[idx]
                        if r <= cumsum:
                            selected_idx = c
                            break
                    clause_idx = selected_idx
                else:
                    clause_idx = rng.choice(unsat_list)
            else:
                clause_idx = unsat_list[0]
            
            clause_info = clause_var_info[clause_idx]
            
            if rng.random() < noise:
                # Random walk step
                v, s = rng.choice(clause_info)
                var_idx = v
            else:
                # Weighted greedy step: Find variable that gives best weighted gain
                best_weighted_gain = float('inf')
                best_vars = []
                
                for v, s in clause_info:
                    var_idx = v
                    current_val = assignment[var_idx]
                    
                    # Calculate weighted delta: prioritize fixing high-priority clauses
                    weighted_delta = 0.0
                    for c_idx in var_to_clauses[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_signs[var_idx][c_idx]
                        
                        # Is the literal satisfied currently?
                        is_sat = (assignment[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            if old_w == 1:
                                # This clause would become unsat
                                weighted_delta += clause_priority[c_idx]
                        else:
                            if old_w == 0:
                                # This clause would become sat
                                weighted_delta -= clause_priority[c_idx]
                    
                    if weighted_delta < best_weighted_gain:
                        best_weighted_gain = weighted_delta
                        best_vars = [var_idx]
                        # Early termination if we can fix an unsat clause without breaking any
                        if weighted_delta <= -max(clause_priority[c] for c in unsat_set if c in var_to_clauses[var_idx]) if unsat_set else False:
                            break
                    elif weighted_delta == best_weighted_gain:
                        best_vars.append(var_idx)
                            
                var_idx = rng.choice(best_vars)
            
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause weights and unsat set
            for c_idx in var_to_clauses[var_idx]:
                old_w = clause_weights[c_idx]
                c_sign = var_signs[var_idx][c_idx]
                
                # Is the literal satisfied NOW (after flip)?
                is_sat = (assignment[var_idx] == (c_sign == 1))
                
                if is_sat:
                    clause_weights[c_idx] = old_w + 1
                else:
                    clause_weights[c_idx] = old_w - 1
                    
                # Update unsat set and priorities
                if old_w == 0 and clause_weights[c_idx] > 0:
                    unsat_set.discard(c_idx)
                    total_unsat -= 1
                    # Satisfying a clause: reduce its priority
                    clause_priority[c_idx] = max(1.0, clause_priority[c_idx] * weight_decay)
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    unsat_set.add(c_idx)
                    total_unsat += 1
                    # Breaking a clause: increase its priority
                    clause_priority[c_idx] = clause_priority[c_idx] * weight_boost
        
        # Decay all priorities at end of restart
        for i in range(m):
            clause_priority[i] = max(1.0, clause_priority[i] * weight_decay)

    return best_assignment
# EVOLVE-BLOCK-END
