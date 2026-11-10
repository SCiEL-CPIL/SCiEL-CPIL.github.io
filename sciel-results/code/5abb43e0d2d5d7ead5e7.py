import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n
        
    # Precompute variable to clause index mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for lit in c:
            var = abs(lit) - 1
            var_to_clauses[var].append(i)
            
    def is_satisfied(clause, assign):
        for lit in clause:
            var = abs(lit) - 1
            if assign[var] == -1:
                continue
            is_true = (lit > 0)
            if (assign[var] == 1 and is_true) or (assign[var] == 0 and not is_true):
                return True
        return False

    def get_unsat_indices(assign):
        unsat = []
        for i, c in enumerate(clauses):
            if not is_satisfied(c, assign):
                unsat.append(i)
        return unsat

    def flip_gain(var, new_val, assign):
        gain = 0
        for ci in var_to_clauses[var]:
            c = clauses[ci]
            sat_by_other = False
            for lit in c:
                v = abs(lit) - 1
                if v == var:
                    continue
                if assign[v] != -1:
                    is_true = (lit > 0)
                    if (assign[v] == 1 and is_true) or (assign[v] == 0 and not is_true):
                        sat_by_other = True
                        break
            if sat_by_other:
                continue
            for lit in c:
                if abs(lit) - 1 == var:
                    is_true = (lit > 0)
                    if (new_val == 1 and is_true) or (new_val == 0 and not is_true):
                        gain += 1
                        break
        return gain

    rng = random.Random(42)
    
    best_assign = None
    best_unsat_count = len(clauses)
    
    max_restarts = 200
    max_steps_per_restart = 2000
    noise = 0.25
    
    for restart in range(max_restarts):
        assign = [1 if rng.random() < 0.5 else 0 for _ in range(n)]
        
        unsat_indices = get_unsat_indices(assign)
        if not unsat_indices:
            return assign
            
        current_unsat_count = len(unsat_indices)
        if current_unsat_count < best_unsat_count:
            best_unsat_count = current_unsat_count
            best_assign = assign[:]
            
        for step in range(max_steps_per_restart):
            if not unsat_indices:
                return assign
                
            ci = rng.choice(unsat_indices)
            clause = clauses[ci]
            
            if rng.random() < noise:
                lit = rng.choice(clause)
                var = abs(lit) - 1
                new_val = 1 if lit > 0 else 0
            else:
                best_var = -1
                best_gain = -1
                best_new_val = -1
                
                for lit in clause:
                    var = abs(lit) - 1
                    val_satisfying = 1 if lit > 0 else 0
                    gain = flip_gain(var, val_satisfying, assign)
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                        best_new_val = val_satisfying
                        
                if best_var == -1:
                    lit = rng.choice(clause)
                    var = abs(lit) - 1
                    new_val = 1 if lit > 0 else 0
                else:
                    var = best_var
                    new_val = best_new_val
            
            assign[var] = new_val
            
            unsat_indices = get_unsat_indices(assign)
            current_unsat_count = len(unsat_indices)
            
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assign = assign[:]
                
            if not unsat_indices:
                return assign

    return best_assign
