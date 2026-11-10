"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 256
    squares = set()
    for i in range(1, int(math.isqrt(2*n)) + 2):
        squares.add(i*i)
    
    # Identify valid elements (no self-conflict)
    self_conflict = set()
    for i in range(1, n+1):
        if 2*i in squares:
            self_conflict.add(i)
            
    valid_elements = [i for i in range(1, n+1) if i not in self_conflict]
    elem_to_idx = {e: idx for idx, e in enumerate(valid_elements)}
    m = len(valid_elements)
    
    # Build conflict list
    conflicts = []
    for i in range(1, n+1):
        if i in self_conflict:
            continue
        for j in range(i+1, n+1):
            if j in self_conflict:
                continue
            if i + j in squares:
                conflicts.append((elem_to_idx[i], elem_to_idx[j]))
                
    # Build adjacency for heuristic
    adj = [set() for _ in range(m)]
    for (u, v) in conflicts:
        adj[u].add(v)
        adj[v].add(u)
        
    # Heuristic initial solution: Greedy by degree
    order = sorted(range(m), key=lambda x: len(adj[x]))
    init_sol = [0] * m
    selected = set()
    for idx in order:
        if not any(nb in selected for nb in adj[idx]):
            selected.add(idx)
            init_sol[idx] = 1
            
    init_sol = np.array(init_sol)
    
    # Try ILP
    best_result = [valid_elements[i] for i in range(m) if init_sol[i] == 1]
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        c = np.full(m, -1.0)
        
        if conflicts:
            A_ub = np.zeros((len(conflicts), m))
            for k, (u, v) in enumerate(conflicts):
                A_ub[k, u] = 1
                A_ub[k, v] = 1
            constraints = [LinearConstraint(A_ub, None, 1)]
        else:
            constraints = []
            
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(m)
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"presolve": True, "time_limit": 70}
        )
        
        if res.status == 0:
            sol = np.round(res.x).astype(int)
            result_indices = np.where(sol == 1)[0]
            if len(result_indices) > len(best_result):
                best_result = [valid_elements[i] for i in result_indices]
                
    except Exception:
        pass
        
    # Validation
    result_set = set(best_result)
    valid = True
    for a in best_result:
        for b in best_result:
            if a <= b:
                if a + b in squares:
                    valid = False
                    break
        if not valid:
            break
            
    if valid:
        return sorted(best_result)
    else:
        # Fallback to heuristic which is guaranteed valid
        return sorted([valid_elements[i] for i in range(m) if init_sol[i] == 1])
# EVOLVE-BLOCK-END
