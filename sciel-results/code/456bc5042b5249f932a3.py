import time
import itertools
import numpy as np
from scipy import optimize

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # The set of interest is [51, 100]
    # Triples x+y+z=200 with 51 <= x < y < z <= 100
    
    # Generate all bad triples
    triples = []
    for z in range(100, 67, -1):
        target_xy = 200 - z
        max_x = (target_xy - 1) // 2
        min_x = max(51, 200 - 2*z + 1)
        
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        return list(range(51, 101))

    values = list(range(51, 101))
    val_to_idx = {v: i for i, v in enumerate(values)}
    n_vars = 50
    
    # Build constraint matrix for ILP
    A = np.zeros((len(triples), n_vars), dtype=int)
    for i, t in enumerate(triples):
        for v in t:
            A[i, val_to_idx[v]] = 1
            
    b = np.ones(len(triples), dtype=int)
    c = np.ones(n_vars, dtype=int)
    
    bounds = [(0, 1)] * n_vars
    integrality = np.ones(n_vars, dtype=int)
    
    H = None
    try:
        # Try ILP
        # Note: scipy.optimize.milp expects constraints in the form Ax >= b or Ax <= b?
        # The tuple (A, b) is interpreted as Ax >= b according to some docs, 
        # but let's be safe and use LinearConstraint if possible, or check result.
        # Actually, LinearConstraint is safer.
        from scipy.optimize import LinearConstraint
        constraints = LinearConstraint(A, 1, None) # Ax >= 1
        
        res = optimize.milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        if res.status == 0:
            # Check if solution is valid
            H_candidate = set()
            for i in range(n_vars):
                if res.x[i] > 0.5:
                    H_candidate.add(values[i])
            
            # Verify H_candidate covers all triples
            valid = True
            for t in triples:
                if not any(v in H_candidate for v in t):
                    valid = False
                    break
            
            if valid:
                H = H_candidate
            else:
                # ILP returned invalid solution (e.g. all zeros)
                H = None
        else:
            H = None
    except Exception:
        H = None

    if H is None:
        # Fallback to Greedy
        freq = {v: 0 for v in values}
        for t in triples:
            for v in t:
                freq[v] += 1
                
        H = set()
        active_triples = set(triples)
        
        while active_triples:
            counts = {}
            for t in active_triples:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
                    
            if not counts:
                break
                
            sorted_elems = sorted(counts.items(), key=lambda item: -item[1])
            best_elem = sorted_elems[0][0]
            
            H.add(best_elem)
            active_triples = {t for t in active_triples if best_elem not in t}
            
            if time.time() > deadline:
                break

    result = [v for v in values if v not in H]
    return result
