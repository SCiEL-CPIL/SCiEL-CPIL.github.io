import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    C = len(clauses)
    
    # Preprocess
    clause_vars = [[0]*3 for _ in range(C)]
    clause_signs = [[1]*3 for _ in range(C)]
    var_to_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            clause_vars[i][j] = v
            clause_signs[i][j] = s
            var_to_clauses[v].append((i, j))
            
    def is_lit_sat(assignment, v, s):
        if s == 1:
            return assignment[v] == 1
        else:
            return assignment[v] == 0

    def init_state():
        assignment = [random.randint(0, 1) for _ in range(n)]
        is_sat = [[False]*3 for _ in range(C)]
        clause_sat_count = [0]*C
        unsat_clauses = []
        
        for c in range(C):
            count = 0
            for j in range(3):
                v = clause_vars[c][j]
                s = clause_signs[c][j]
                if is_lit_sat(assignment, v, s):
                    is_sat[c][j] = True
                    count += 1
            clause_sat_count[c] = count
            if count == 0:
                unsat_clauses.append(c)
        return assignment, is_sat, clause_sat_count, unsat_clauses

    start_time = time.time()
    deadline = start_time + 295.0
    
    best_assignment = None
    best_unsat_count = C + 1
    
    while time.time() < deadline:
        assignment, is_sat, clause_sat_count, unsat_clauses = init_state()
        current_unsat = len(unsat_clauses)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
            
        max_steps = 100000
        steps = 0
        
        while steps < max_steps and time.time() < deadline:
            if not unsat_clauses:
                return assignment
                
            # Pick random unsat clause
            c = random.choice(unsat_clauses)
            
            best_v = -1
            best_delta = -1000
            
            for p in range(3):
                v = clause_vars[c][p]
                
                delta = 0
                for (cc, pp) in var_to_clauses[v]:
                    if is_sat[cc][pp]:
                        if clause_sat_count[cc] == 1:
                            delta -= 1
                    else:
                        if clause_sat_count[cc] == 0:
                            delta += 1
                            
                noise = random.random() * 0.5
                score = delta + noise
                
                if score > best_delta:
                    best_delta = score
                    best_v = v
                    
            if best_v != -1:
                for (cc, pp) in var_to_clauses[best_v]:
                    old_sat = is_sat[cc][pp]
                    old_count = clause_sat_count[cc]
                    
                    if old_sat:
                        is_sat[cc][pp] = False
                        clause_sat_count[cc] = old_count - 1
                        if old_count == 1:
                            unsat_clauses.append(cc)
                    else:
                        is_sat[cc][pp] = True
                        clause_sat_count[cc] = old_count + 1
                        if old_count == 0:
                            unsat_clauses.remove(cc)
                            
                assignment[best_v] = 1 - assignment[best_v]
                steps += 1

    if best_assignment:
        return best_assignment
    return [random.randint(0, 1) for _ in range(n)]
