"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s kill
    
    # Precompute clause literals as (var_idx, polarity)
    # var_idx is 0-based, polarity is True for positive, False for negative
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_opt)
    
    # Build watchers: for each literal (var, pol), list of clause indices
    # Key: var * 2 + (1 if pol else 0)
    watchers = [[] for _ in range(2 * n)]
    
    for i, c in enumerate(clauses_opt):
        for var_idx, pol in c:
            key = var_idx * 2 + (1 if pol else 0)
            watchers[key].append(i)
            
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 100000 # High number, controlled by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Restart strategy
        if restart > 0 and best_assignment is not None:
             # 10% chance to restart from best known assignment (exploitation)
             if random.random() < 0.1:
                 assignment = best_assignment[:]
             else:
                 assignment = [random.random() < 0.5 for _ in range(n)]
        else:
            assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Initialize clause satisfaction counts and unsat list
        clause_sat_count = [0] * m
        unsat_list = []
        unsat_pos = [-1] * m # Position of clause in unsat_list, -1 if not unsat
        
        for i, c in enumerate(clauses_opt):
            count = 0
            for var_idx, pol in c:
                if assignment[var_idx] == pol:
                    count += 1
            clause_sat_count[i] = count
            if count == 0:
                unsat_list.append(i)
                unsat_pos[i] = len(unsat_list) - 1
                
        if not unsat_list:
            return assignment
            
        if len(unsat_list) < best_unsat_count:
            best_unsat_count = len(unsat_list)
            best_assignment = assignment[:]
            
        # WalkSAT loop
        max_steps = 100000 
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if not unsat_list:
                return assignment
                
            if len(unsat_list) < best_unsat_count:
                best_unsat_count = len(unsat_list)
                best_assignment = assignment[:]
                
            # Pick a random unsatisfied clause
            clause_idx = unsat_list[random.randrange(len(unsat_list))]
            clause = clauses_opt[clause_idx]
            
            # Heuristic: Min-Conflict
            best_var = None
            best_score = -1
            
            for var_idx, pol in clause:
                # We want to flip var_idx.
                # The literal (var_idx, pol) will become true.
                # Count how many *other* unsat clauses contain (var_idx, pol).
                key = var_idx * 2 + (1 if pol else 0)
                watchers_list = watchers[key]
                
                score = 0
                for other_clause_idx in watchers_list:
                    if other_clause_idx != clause_idx and clause_sat_count[other_clause_idx] == 0:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_var = var_idx
                    
            # Apply noise
            if random.random() < 0.25:
                var_idx, _ = clause[random.randrange(len(clause))]
            else:
                var_idx = best_var
                
            if var_idx is None:
                continue
                
            # Flip the variable
            assignment[var_idx] = not assignment[var_idx]
            
            # Update clause satisfaction counts for all clauses containing var_idx
            # 1. Clauses with (var_idx, True)
            key_pos = var_idx * 2 + 1
            for c_idx in watchers[key_pos]:
                if assignment[var_idx]:
                    # Literal became True
                    clause_sat_count[c_idx] += 1
                    if clause_sat_count[c_idx] == 1:
                        # Became satisfied (was 0)
                        # Remove from unsat_list
                        pos = unsat_pos[c_idx]
                        if pos != -1:
                            last_idx = unsat_list[-1]
                            unsat_list[pos] = last_idx
                            unsat_pos[last_idx] = pos
                            unsat_list.pop()
                            unsat_pos[c_idx] = -1
                else:
                    # Literal became False
                    clause_sat_count[c_idx] -= 1
                    if clause_sat_count[c_idx] == 0:
                        # Became unsatisfied (was 1 or more, now 0)
                        # Add to unsat_list
                        unsat_list.append(c_idx)
                        unsat_pos[c_idx] = len(unsat_list) - 1
                        
            # 2. Clauses with (var_idx, False)
            key_neg = var_idx * 2
            for c_idx in watchers[key_neg]:
                if assignment[var_idx]:
                    # Literal (var_idx, False) is now False
                    # Old assignment was False, so literal was True.
                    # Delta -1
                    clause_sat_count[c_idx] -= 1
                    if clause_sat_count[c_idx] == 0:
                        # Became unsatisfied
                        unsat_list.append(c_idx)
                        unsat_pos[c_idx] = len(unsat_list) - 1
                else:
                    # Literal (var_idx, False) is now True
                    # Old assignment was True, so literal was False.
                    # Delta +1
                    clause_sat_count[c_idx] += 1
                    if clause_sat_count[c_idx] == 1:
                        # Became satisfied
                        pos = unsat_pos[c_idx]
                        if pos != -1:
                            last_idx = unsat_list[-1]
                            unsat_list[pos] = last_idx
                            unsat_pos[last_idx] = pos
                            unsat_list.pop()
                            unsat_pos[c_idx] = -1
                        
        if time.time() > deadline:
            break
            
    if best_assignment is not None:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
