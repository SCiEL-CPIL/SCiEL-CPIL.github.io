import random
import math
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 295  # 5s safety margin before 300s
    
    # Precompute: clause_data as list of (var_idx, sign) tuples
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])
    
    # var_to_clauses: for each variable, list of (clause_idx, sign_in_that_clause)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            var_to_clauses[var].append((ci, sign))
    
    # Helper: check if literal (var, sign) is satisfied given assignment
    def lit_sat(var, sign, assignment):
        if sign == 1:
            return assignment[var]
        else:
            return not assignment[var]
    
    # Helper: compute initial clause satisfaction counts
    def init_counts(assignment):
        counts = [0] * m
        for ci, clause in enumerate(clause_data):
            cnt = 0
            for var, sign in clause:
                if lit_sat(var, sign, assignment):
                    cnt += 1
            counts[ci] = cnt
        return counts
    
    # SA Parameters
    T0 = 1.0
    alpha = 0.995
    T_min = 0.001
    max_iter_per_restart = 50000
    
    best_energy = float('inf')
    best_assignment = None
    
    while time.time() < deadline:
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_counts = init_counts(assignment)
        
        # Build initial unsat_clauses list
        unsat_clauses = [i for i in range(m) if clause_counts[i] == 0]
        energy = len(unsat_clauses)
        
        if energy == 0:
            return assignment
        
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment[:]
        
        T = T0
        iters = 0
        no_improve = 0
        best_in_run = energy
        
        while T > T_min and iters < max_iter_per_restart:
            if time.time() > deadline:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Pick random unsatisfied clause
            c_idx = random.choice(unsat_clauses)
            clause = clause_data[c_idx]
            
            # Pick random literal/variable from this clause
            var, sign = random.choice(clause)
            
            # Calculate delta E for flipping 'var'
            delta_e = 0
            for c2_idx, s2 in var_to_clauses[var]:
                current_count = clause_counts[c2_idx]
                # Check current state of this literal
                if lit_sat(var, s2, assignment):
                    # Currently satisfied -> flipping makes it unsat
                    if current_count == 1:
                        delta_e += 1
                else:
                    # Currently unsatisfied -> flipping makes it sat
                    if current_count == 0:
                        delta_e -= 1
            
            # Accept?
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                # Flip
                assignment[var] = not assignment[var]
                
                # Update clause_counts and unsat_clauses
                new_unsat = []
                for c2_idx, s2 in var_to_clauses[var]:
                    # After flip, the literal state is inverted
                    # If now satisfied, count increases by 1
                    # If now unsatisfied, count decreases by 1
                    if lit_sat(var, s2, assignment):
                        clause_counts[c2_idx] += 1
                    else:
                        clause_counts[c2_idx] -= 1
                
                # Rebuild unsat_clauses list (only for affected clauses + check others)
                # Actually, we need to rebuild the full list since multiple clauses changed
                # But we can be smarter: only rebuild if needed
                # For now, rebuild from scratch - O(m) but m is ~4-5n=600-750
                # This is acceptable since we do it per accepted move, not per iteration
                
                unsat_clauses = [i for i in range(m) if clause_counts[i] == 0]
                energy = len(unsat_clauses)
                
                if energy == 0:
                    return assignment
                
                if energy < best_in_run:
                    best_in_run = energy
                    no_improve = 0
                else:
                    no_improve += 1
                
                if energy < best_energy:
                    best_energy = energy
                    best_assignment = assignment[:]
                
                # Early restart if stuck too long
                if no_improve > 5000:
                    break
            
            # Cool down
            T *= alpha
            iters += 1
        
        # End of inner loop
        if time.time() > deadline:
            break

    if best_assignment:
        return best_assignment
    else:
        return [False] * n
