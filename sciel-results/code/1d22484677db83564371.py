import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    time_limit = 290.0
    
    # Preprocessing
    clause_vars = []
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        vars_in_clause = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            vars_in_clause.append((var_idx, sign))
            var_clauses[var_idx].append((i, sign))
        clause_vars.append(vars_in_clause)
        
    m = len(clauses)
    
    # State
    assignment = np.random.randint(0, 2, n)
    clause_unsat = np.zeros(m, dtype=int)
    
    # Initialize clause_unsat
    for i in range(m):
        cnt = 0
        for var_idx, sign in clause_vars[i]:
            val = assignment[var_idx]
            req_val = 1 if sign == 1 else 0
            if val != req_val:
                cnt += 1
        clause_unsat[i] = cnt
        
    unsat_clauses = [i for i in range(m) if clause_unsat[i] > 0]
    
    if not unsat_clauses:
        return assignment.tolist()

    last_assignment = assignment.copy()
    
    p = 0.5
    restart_limit = 10000
    max_restarts = 10000 # High enough to be time-limited
    
    flips_in_restart = 0
    rng = random.Random(42)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart > 0:
            if rng.random() < 0.5:
                assignment = last_assignment.copy()
            else:
                assignment = np.random.randint(0, 2, n)
            
            clause_unsat = np.zeros(m, dtype=int)
            unsat_clauses = []
            for i in range(m):
                cnt = 0
                for var_idx, sign in clause_vars[i]:
                    val = assignment[var_idx]
                    req_val = 1 if sign == 1 else 0
                    if val != req_val:
                        cnt += 1
                clause_unsat[i] = cnt
                if cnt > 0:
                    unsat_clauses.append(i)
                    
            if not unsat_clauses:
                return assignment.tolist()
            flips_in_restart = 0
        
        while unsat_clauses:
            if time.time() - start_time > time_limit:
                return assignment.tolist()
                
            c_idx = rng.choice(unsat_clauses)
            vars_in_clause = clause_vars[c_idx]
            
            if rng.random() < p:
                var_idx, _ = rng.choice(vars_in_clause)
            else:
                best_gain = -100
                best_var = vars_in_clause[0][0]
                for v_idx, _ in vars_in_clause:
                    gain = 0
                    for c_i, sgn_in_c in var_clauses[v_idx]:
                        if clause_unsat[c_i] == 0:
                            gain -= 1
                        else:
                            val = assignment[v_idx]
                            req_val = 1 if sgn_in_c == 1 else 0
                            is_unsat = (val != req_val)
                            if clause_unsat[c_i] == 1 and is_unsat:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_var = v_idx
                var_idx = best_var
                
            # Flip
            assignment[var_idx] = 1 - assignment[var_idx]
            
            # Update counts
            for c_i, sgn_in_c in var_clauses[var_idx]:
                old_val = 1 - assignment[var_idx]
                new_val = assignment[var_idx]
                req_val = 1 if sgn_in_c == 1 else 0
                old_sat = (old_val == req_val)
                new_sat = (new_val == req_val)
                
                if old_sat and not new_sat:
                    clause_unsat[c_i] += 1
                    if clause_unsat[c_i] == 1:
                        unsat_clauses.append(c_i)
                elif not old_sat and new_sat:
                    clause_unsat[c_i] -= 1
                    if clause_unsat[c_i] == 0:
                        # Remove c_i from unsat_clauses
                        # Find index
                        try:
                            idx = unsat_clauses.index(c_i)
                            unsat_clauses[idx] = unsat_clauses[-1]
                            unsat_clauses.pop()
                        except ValueError:
                            pass
                            
            flips_in_restart += 1
            if flips_in_restart > restart_limit:
                last_assignment = assignment.copy()
                break
                
    return assignment.tolist()
