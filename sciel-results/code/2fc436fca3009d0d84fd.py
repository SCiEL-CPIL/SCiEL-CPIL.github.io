# EVOLVE-BLOCK-START
import random
import time


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
    
    # Deadline: 325 seconds to be safe against the 330s kill
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Parameters
    noise = 0.3  # Lower noise for more greedy steps
    restart_threshold = 3000
    
    while time.time() < deadline:
        # Initialize random assignment with some bias
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial clause weights and track unsat clauses in a set
        clause_weights = [0] * len(clauses)
        unsat_set = set()
        for i, clause_info in enumerate(clause_var_info):
            w = 0
            for v, s in clause_info:
                if assignment[v] == (s == 1):
                    w += 1
            clause_weights[i] = w
            if w == 0:
                unsat_set.add(i)
        
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
                
            clause_idx = rng.choice(list(unsat_set))
            clause_info = clause_var_info[clause_idx]
            
            if rng.random() < noise:
                # Random walk step
                v, s = rng.choice(clause_info)
                var_idx = v
            else:
                # Greedy step: Find variable in clause that gives best gain
                best_gain = float('inf')
                best_var_idx = -1
                
                for v, s in clause_info:
                    var_idx = v
                    current_val = assignment[var_idx]
                    
                    # Calculate delta in total_unsat
                    delta = 0
                    for c_idx in var_to_clauses[var_idx]:
                        old_w = clause_weights[c_idx]
                        c_sign = var_signs[var_idx][c_idx]
                        
                        # Is the literal satisfied currently?
                        is_sat = (assignment[var_idx] == (c_sign == 1))
                        
                        if is_sat:
                            if old_w == 1:
                                delta += 1
                        else:
                            if old_w == 0:
                                delta -= 1
                    
                    if delta < best_gain:
                        best_gain = delta
                        best_var_idx = var_idx
                        if best_gain == -1:
                            break
                            
                var_idx = best_var_idx
            
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
                    
                # Update unsat set
                if old_w == 0 and clause_weights[c_idx] > 0:
                    unsat_set.discard(c_idx)
                    total_unsat -= 1
                elif old_w > 0 and clause_weights[c_idx] == 0:
                    unsat_set.add(c_idx)
                    total_unsat += 1

    return best_assignment
# EVOLVE-BLOCK-END
