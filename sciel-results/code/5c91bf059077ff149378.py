import random
import time
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Time limit with safety margin
    deadline = time.time() + 285.0
    
    m = len(clauses)
    
    # Build clause-to-variable index and var-to-clause index
    # var_to_clauses[v] = list of (clause_idx, sign) where sign=1 for positive, 0 for negative
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else 0
            var_to_clauses[v].append((i, sign))
    
    rng = random.Random(42)
    best_assignment = None
    best_unsat = float('inf')
    
    # Parameters
    max_restarts = 800
    steps_per_restart = 5000
    noise_scale = 0.3
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization with slight bias
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute clause satisfaction counts
        clause_sat_count = [0] * m
        for i, c in enumerate(clauses):
            for l in c:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    clause_sat_count[i] += 1
        
        current_unsat = sum(1 for cnt in clause_sat_count if cnt == 0)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
        
        local_best_unsat = current_unsat
        no_improve = 0
        
        # Build unsat clause list
        unsat_clauses = [i for i in range(m) if clause_sat_count[i] == 0]
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            c_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
            c = clauses[c_idx]
            
            # Find best variable to flip in this clause
            best_var = -1
            best_score = float('-inf')
            
            for l in c:
                v = abs(l) - 1
                lit_true = assignment[v] == (l > 0)
                
                # Calculate delta for flipping v
                break_count = 0
                sat_count = 0
                
                for (ci, sign) in var_to_clauses[v]:
                    if ci == c_idx:
                        continue
                    ci_lit_true = assignment[v] if sign == 1 else (not assignment[v])
                    
                    old_cnt = clause_sat_count[ci]
                    if ci_lit_true:
                        new_cnt = old_cnt - 1
                    else:
                        new_cnt = old_cnt + 1
                    
                    if old_cnt > 0 and new_cnt == 0:
                        break_count += 1
                    elif old_cnt == 0 and new_cnt > 0:
                        sat_count += 1
                
                # c_idx: currently unsat (count=0), flipping v's literal makes it True
                # So c_idx goes from 0 to 1 -> becomes satisfied
                score = (1 + sat_count) - break_count
                # Add noise for diversification
                score += rng.random() * noise_scale
                
                if score > best_score:
                    best_score = score
                    best_var = v
            
            # Flip best_var and update state incrementally
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update clause_sat_count for all clauses containing best_var
            for (ci, sign) in var_to_clauses[best_var]:
                if sign == 1:
                    if old_val:
                        clause_sat_count[ci] -= 1
                        if clause_sat_count[ci] == 0:
                            unsat_clauses.append(ci)
                    else:
                        clause_sat_count[ci] += 1
                        if clause_sat_count[ci] == 1:
                            if ci in unsat_clauses:
                                unsat_clauses.remove(ci)
                else:
                    if not old_val:
                        clause_sat_count[ci] -= 1
                        if clause_sat_count[ci] == 0:
                            unsat_clauses.append(ci)
                    else:
                        clause_sat_count[ci] += 1
                        if clause_sat_count[ci] == 1:
                            if ci in unsat_clauses:
                                unsat_clauses.remove(ci)
            
            current_unsat = len(unsat_clauses)
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improve = 0
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
            else:
                no_improve += 1
                
            if no_improve > 200:
                break  # Restart early if stuck
    
    return best_assignment
