import random
import time
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    m = len(clauses)
    
    # Precompute clause info
    clause_lits = []
    var_to_clauses = [[] for _ in range(n)]
    # var_clause_sign[v][c_idx] = sign of v in clause c_idx
    var_clause_sign = [{} for _ in range(n)]
    
    for c_idx, clause in enumerate(clauses):
        lits = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            lits.append((v, s))
            var_to_clauses[v].append(c_idx)
            var_clause_sign[v][c_idx] = s
        clause_lits.append(lits)
        
    def calculate_initial_clause_sat(assignment):
        sat_counts = [0] * m
        unsat = []
        for i, lits in enumerate(clause_lits):
            count = 0
            for v, s in lits:
                if (s == 1 and assignment[v]) or (s == -1 and not assignment[v]):
                    count += 1
            sat_counts[i] = count
            if count == 0:
                unsat.append(i)
        return sat_counts, unsat

    deadline = time.time() + 290.0 # Safety margin for 300s limit
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Phase saving
    phase = [rng.random() < 0.5 for _ in range(n)]
    
    restart_count = 0
    max_restarts = 1000000
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # Initialize assignment
        assignment = []
        for i in range(n):
            if rng.random() < 0.5:
                assignment.append(phase[i])
            else:
                assignment.append(rng.random() < 0.5)
        
        sat_counts, unsat_clauses = calculate_initial_clause_sat(assignment)
        total_unsat = len(unsat_clauses)
        
        if total_unsat == 0:
            return assignment
        
        if total_unsat < best_unsat_count:
            best_unsat_count = total_unsat
            best_assignment = assignment[:]
            phase = assignment[:]
            
        # Local Search
        steps_since_improvement = 0
        min_unsat_seen = total_unsat
        base_noise = 0.3
        
        for step in range(10000):
            if time.time() > deadline:
                break
            
            if total_unsat == 0:
                return assignment
            
            if total_unsat < min_unsat_seen:
                min_unsat_seen = total_unsat
                steps_since_improvement = 0
                if total_unsat < best_unsat_count:
                    best_unsat_count = total_unsat
                    best_assignment = assignment[:]
                    phase = assignment[:]
            else:
                steps_since_improvement += 1
                
            if steps_since_improvement > 500:
                break
            
            current_noise = min(0.8, base_noise + steps_since_improvement * 0.005)
            
            if not unsat_clauses:
                break
            
            c_idx = unsat_clauses[rng.randrange(len(unsat_clauses))]
            lits = clause_lits[c_idx]
            
            if rng.random() < current_noise:
                v, s = lits[rng.randrange(3)]
                var_idx = v
            else:
                # MaxWalkSAT
                max_sat = -1
                best_var = lits[0][0]
                for v, s in lits:
                    new_sat_count = 0
                    for vv, ss in lits:
                        if vv == v:
                            if (s == 1 and not assignment[v]) or (s == -1 and assignment[v]):
                                new_sat_count += 1
                        else:
                            if (ss == 1 and assignment[vv]) or (ss == -1 and not assignment[vv]):
                                new_sat_count += 1
                    if new_sat_count > max_sat:
                        max_sat = new_sat_count
                        best_var = v
                var_idx = best_var
                
            # Flip
            assignment[var_idx] = not assignment[var_idx]
            
            # Update sat_counts and unsat_clauses
            for oc_idx in var_to_clauses[var_idx]:
                old_sat = sat_counts[oc_idx]
                sign_in_oc = var_clause_sign[var_idx][oc_idx]
                
                is_sat_now = (sign_in_oc == 1 and assignment[var_idx]) or (sign_in_oc == -1 and not assignment[var_idx])
                
                if is_sat_now:
                    new_sat = old_sat + 1
                else:
                    new_sat = old_sat - 1
                    
                sat_counts[oc_idx] = new_sat
                
                if old_sat == 0 and new_sat > 0:
                    # Remove from unsat
                    for j in range(len(unsat_clauses)):
                        if unsat_clauses[j] == oc_idx:
                            unsat_clauses[j] = unsat_clauses[-1]
                            unsat_clauses.pop()
                            break
                    total_unsat -= 1
                elif old_sat > 0 and new_sat == 0:
                    unsat_clauses.append(oc_idx)
                    total_unsat += 1
    
    return best_assignment
