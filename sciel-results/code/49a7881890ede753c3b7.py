import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # State
    values = [0] * n
    assigned = [0] * n
    levels = [0] * n
    reasons = [None] * n
    activity = [1.0] * n
    trail = []
    trail_lim = []
    
    # Clauses DB (Original + Learned)
    # We will not implement full clause learning to keep it simple,
    # but we can add the conflict clause if we implement non-chronological.
    # For now, chronological backtracking.
    
    clauses_db = [list(c) for c in clauses]
    
    def get_val(l):
        idx = abs(l) - 1
        if not assigned[idx]:
            return None
        return values[idx] == 1
        
    def is_sat(c):
        for l in c:
            if get_val(l) == True:
                return True
        return False
        
    def is_unsat(c):
        for l in c:
            v = get_val(l)
            if v is None:
                return False
            if v == True:
                return False
        return True
        
    def propagate():
        changed = True
        conflict_idx = None
        while changed:
            changed = False
            for i, c in enumerate(clauses_db):
                if is_sat(c):
                    continue
                if is_unsat(c):
                    return i
                unassigned_lits = [l for l in c if get_val(l) is None]
                if len(unassigned_lits) == 1:
                    lit = unassigned_lits[0]
                    var_idx = abs(lit) - 1
                    if not assigned[var_idx]:
                        values[var_idx] = 1 if lit > 0 else 0
                        assigned[var_idx] = 1
                        levels[var_idx] = len(trail_lim)
                        reasons[var_idx] = i
                        trail.append(var_idx)
                        changed = True
        return None

    def pick_var():
        max_act = -1
        best_var = -1
        for i in range(n):
            if not assigned[i]:
                if activity[i] > max_act:
                    max_act = activity[i]
                    best_var = i
        return best_var

    def decay():
        for i in range(n):
            activity[i] *= 0.95

    rng = random.Random(0)
    
    restarts = 0
    max_restarts = 100
    
    while restarts < max_restarts:
        if time.time() > deadline:
            break
            
        # Reset state
        for i in range(n):
            assigned[i] = 0
            values[i] = 0
            levels[i] = 0
            reasons[i] = None
        trail = []
        trail_lim = []
        
        solved = False
        unsat = False
        
        while not solved and not unsat:
            conflict_idx = propagate()
            
            if conflict_idx is not None:
                if trail_lim:
                    # Backtrack
                    lim = trail_lim.pop()
                    while len(trail) > lim:
                        var = trail.pop()
                        assigned[var] = 0
                        reasons[var] = None
                        levels[var] = 0
                        # Increase activity of var?
                        activity[var] += 1.0
                else:
                    unsat = True
                    break
            else:
                if len(trail) == n:
                    solved = True
                    break
                    
                var = pick_var()
                if var == -1:
                    if len(trail) == n:
                        solved = True
                    break
                    
                is_true = rng.random() < 0.5
                values[var] = 1 if is_true else 0
                assigned[var] = 1
                levels[var] = len(trail_lim)
                reasons[var] = None
                trail.append(var)
                trail_lim.append(len(trail))
                
            if len(trail) % 100 == 0:
                decay()
                
        if solved:
            return values
        if unsat:
            return values
            
        restarts += 1
        
    return values
