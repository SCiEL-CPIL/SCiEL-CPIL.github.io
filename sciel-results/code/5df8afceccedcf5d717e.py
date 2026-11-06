"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocess: convert to 0-indexed, store polarity
    clause_vars = []
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(i)
        clause_vars.append(cv)
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    # Clause weights for weighted selection
    clause_weights = [1.0] * m
    
    def weighted_choice(unsat_list):
        """Select an unsat clause with probability proportional to weight."""
        if len(unsat_list) == 1:
            return unsat_list[0]
        total_w = sum(clause_weights[i] for i in unsat_list)
        r = random.random() * total_w
        cumulative = 0
        for i in unsat_list:
            cumulative += clause_weights[i]
            if r <= cumulative:
                return i
        return unsat_list[-1]
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Adaptive restart strategy with increasing step budget
    base_steps = 300
    max_restarts = 8000
    restarts = 0
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(m):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive step budget: grow with restarts up to a cap
        steps_budget = min(base_steps + restarts * 20, 5000)
        steps = 0
        stall_counter = 0
        prev_unsat = current_unsat
        
        while steps < steps_budget and time.time() - start_time < time_limit:
            if not unsat_set:
                return assignment
            
            # Pick weighted random unsat clause
            unsat_list = list(unsat_set)
            idx = weighted_choice(unsat_list)
            clause = clause_vars[idx]
            
            # Damage-minimizing literal selection
            best_var_idx = -1
            best_damage = float('inf')
            best_is_pos = False
            best_gain = 0
            
            for var_idx, is_pos in clause:
                # Simulate flip of var_idx
                # Current: assignment[var_idx] != is_pos (clause is unsat)
                # After flip: assignment[var_idx] == is_pos
                
                damage = 0
                gain = 0  # Number of currently-unsat clauses that become sat
                
                for c_idx in var_to_clauses[var_idx]:
                    if c_idx == idx:
                        # This clause becomes satisfied
                        gain += 1
                        continue
                    
                    # Find polarity of var_idx in c_idx
                    p_c = None
                    for v, p in clause_vars[c_idx]:
                        if v == var_idx:
                            p_c = p
                            break
                    
                    if p_c is None:
                        continue
                    
                    was_sat = (assignment[var_idx] == p_c)
                    
                    if was_sat:
                        # Clause was satisfied; check if it remains satisfied
                        # After flip, this literal becomes false
                        remains_sat = False
                        for v, p in clause_vars[c_idx]:
                            if v == var_idx:
                                continue
                            if assignment[v] == p:
                                remains_sat = True
                                break
                        if not remains_sat:
                            damage += 1
                    else:
                        # Clause was unsat; after flip this literal becomes true
                        # So this clause becomes satisfied
                        gain += 1
                
                # Net change: gain - damage
                net = gain - damage
                if net > best_gain or (net == best_gain and damage < best_damage):
                    best_gain = net
                    best_damage = damage
                    best_var_idx = var_idx
                    best_is_pos = is_pos
            
            if best_var_idx == -1:
                var_idx, is_pos = random.choice(clause)
            else:
                var_idx = best_var_idx
                is_pos = best_is_pos
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update unsat_set
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                
                if was_unsat and not is_now_unsat:
                    unsat_set.remove(c_idx)
                    # Decrease weight for satisfied clause
                    clause_weights[c_idx] = max(0.1, clause_weights[c_idx] * 0.5)
                elif not was_unsat and is_now_unsat:
                    unsat_set.add(c_idx)
                    # Increase weight for newly unsat clause
                    clause_weights[c_idx] *= 2.0
            
            steps += 1
            
            # Check for solution
            if not unsat_set:
                return assignment
            
            # Track best
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Stall detection
            if current_unsat <= prev_unsat:
                stall_counter = 0
            else:
                stall_counter += 1
            
            prev_unsat = current_unsat
            
            # Early restart if stalled too long
            if stall_counter > 100:
                break
        
        restarts += 1
        
        # Decay weights back toward 1 between restarts
        for i in range(m):
            clause_weights[i] += (1.0 - clause_weights[i]) * 0.3
    
    if best_assignment:
        return best_assignment
    return assignment  # Fallback to last attempt

# EVOLVE-BLOCK-END
