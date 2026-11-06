import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Time limit with safety margin
    deadline = time.time() + 290.0
    
    # Pre-process
    # var_to_clauses: list of lists of (clause_idx, sign)
    # sign: 1 if positive, 0 if negative
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else 0
            var_to_clauses[v].append((i, sign))
            
    m = len(clauses)
    
    rng = random.Random(0)
    best_assignment = None
    best_unsat = float('inf')
    
    # Parameters
    max_restarts = 500
    steps_per_restart = 3000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Compute clause satisfaction counts
        # clause_sat_count[i] = number of true literals in clause i
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
            
            c_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
            c = clauses[c_idx]
            
            # Find best variable to flip in this clause
            best_var = -1
            best_score = float('-inf')
            
            for l in c:
                v = abs(l) - 1
                # Calculate delta for flipping v
                # Flipping v: if l was true, it becomes false; if false, becomes true
                # Current literal value:
                lit_true = assignment[v] == (l > 0)
                
                # For each clause containing v, compute change in sat count
                # and determine if it changes sat/unsat status
                break_count = 0
                sat_count = 0
                
                for (ci, sign) in var_to_clauses[v]:
                    if ci == c_idx:
                        continue
                    # Determine if literal in ci is currently true
                    # The literal in ci for variable v has sign 'sign'
                    # If sign==1 (positive), literal is true when assignment[v] is True
                    # If sign==0 (negative), literal is true when assignment[v] is False
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
                
                # c_idx: currently unsat (count=0), v's literal in c_idx
                # If lit_true was False (which it must be since c_idx is unsat), flipping makes it True
                # So c_idx goes from 0 to 1 -> becomes satisfied
                # This is already accounted: we gain 1 from c_idx
                score = (1 + sat_count) - break_count
                # Add noise for diversification
                score += rng.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_var = v
            
            # Flip best_var and update state incrementally
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update clause_sat_count for all clauses containing best_var
            for (ci, sign) in var_to_clauses[best_var]:
                # Determine if the literal was true before flip
                if sign == 1:
                    # Positive literal: true when assignment is True
                    if old_val:
                        clause_sat_count[ci] -= 1
                        if clause_sat_count[ci] == 0:
                            # Became unsat
                            unsat_clauses.append(ci)
                    else:
                        clause_sat_count[ci] += 1
                        if clause_sat_count[ci] == 1:
                            # Became sat, remove from unsat
                            if ci in unsat_clauses:
                                unsat_clauses.remove(ci)
                else:
                    # Negative literal: true when assignment is False
                    if not old_val:
                        clause_sat_count[ci] -= 1
                        if clause_sat_count[ci] == 0:
                            # Became unsat
                            unsat_clauses.append(ci)
                    else:
                        clause_sat_count[ci] += 1
                        if clause_sat_count[ci] == 1:
                            # Became sat, remove from unsat
                            if ci in unsat_clauses:
                                unsat_clauses.remove(ci)
            
            # Recompute current_unsat from unsat_clauses length
            current_unsat = len(unsat_clauses)
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improve = 0
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
            else:
                no_improve += 1
                
            if no_improve > 150:
                break  # Restart early if stuck

    return best_assignment
