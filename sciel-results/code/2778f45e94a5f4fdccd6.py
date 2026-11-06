import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    time_limit = 290.0
    
    assignment = [0] * (n + 1)
    phase = [1] * (n + 1)
    trail = []
    
    def val(lit):
        if lit > 0:
            return assignment[lit]
        else:
            return -assignment[-lit]
            
    def is_sat():
        for c in clauses:
            if not any(val(l) == 1 for l in c):
                return False
        return True
        
    def propagate():
        # Returns True if no conflict, False if conflict
        changed = True
        while changed:
            changed = False
            for c in clauses:
                unassigned = []
                satisfied = False
                all_false = True
                for l in c:
                    v = val(l)
                    if v == 1:
                        satisfied = True
                        break
                    elif v == 0:
                        unassigned.append(l)
                        all_false = False
                if satisfied:
                    continue
                if all_false:
                    return False
                if len(unassigned) == 1:
                    l = unassigned[0]
                    v = abs(l)
                    if assignment[v] == 0:
                        assignment[v] = 1 if l > 0 else -1
                        phase[v] = 1 if l > 0 else -1
                        trail.append(v)
                        changed = True
        return True
        
    def dpll():
        if not propagate():
            return False
            
        if all(assignment[i] != 0 for i in range(1, n + 1)):
            return is_sat()
            
        # Decision
        var = 0
        for i in range(1, n + 1):
            if assignment[i] == 0:
                var = i
                break
        if var == 0:
            return False
            
        # Try phase
        v1 = phase[var]
        assignment[var] = v1
        trail.append(var)
        
        if dpll():
            return True
            
        # Backtrack
        assignment[var] = 0
        trail.pop()
        
        # Try other phase
        v2 = -v1
        assignment[var] = v2
        trail.append(var)
        
        if dpll():
            return True
            
        # Backtrack
        assignment[var] = 0
        trail.pop()
        
        return False

    # Run with restarts
    for r in range(20):
        if time.time() - start_time > time_limit:
            break
            
        assignment = [0] * (n + 1)
        trail = []
        
        if dpll():
            return [assignment[i] == 1 for i in range(1, n + 1)]
            
        # Update phase randomly?
        for i in range(1, n + 1):
            if random.random() < 0.1:
                phase[i] = -phase[i]

    # Fallback: WalkSAT
    rng = random.Random(0)
    walk_assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def count_unsat(assignment):
        count = 0
        for c in clauses:
            if not any(assignment[abs(l) - 1] == (l > 0) for l in c):
                count += 1
        return count
        
    current_unsat = count_unsat(walk_assignment)
    
    for _ in range(50000):
        if time.time() - start_time > time_limit:
            break
        if current_unsat == 0:
            break
            
        unsat_clauses = [c for c in clauses if not any(walk_assignment[abs(l) - 1] == (l > 0) for l in c)]
        if not unsat_clauses:
            break
        c = rng.choice(unsat_clauses)
        l = rng.choice(c)
        v = abs(l) - 1
        walk_assignment[v] = not walk_assignment[v]
        
        new_unsat = count_unsat(walk_assignment)
        if new_unsat <= current_unsat or rng.random() < 0.05:
            current_unsat = new_unsat
        else:
            walk_assignment[v] = not walk_assignment[v]
            
    return walk_assignment
