import numpy as np
from itertools import product

def construct():
    best_score = float('inf')
    best_set = None
    
    # Initialize with the current 2-smooth set
    candidates_2s = set()
    for i in range(15):
        for j in range(10):
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                candidates_2s.add(val)
    A_init = sorted(list(candidates_2s))[:20]
    if len(A_init) == 20:
        s_init = score(A_init)
        best_score = s_init
        best_set = A_init
    
    # Search AP + GP
    a_range = range(1, 16)
    d_range = [1, 2, 3]
    g_range = range(1, 16)
    r_range = [2, 3, 4]
    n_ap_range = range(4, 17)
    n_gp_range = range(4, 17)
    
    for a in a_range:
        for d in d_range:
            for g in g_range:
                for r in r_range:
                    for n_ap in n_ap_range:
                        for n_gp in n_gp_range:
                            # Generate AP
                            ap = [a + i * d for i in range(n_ap)]
                            # Generate GP
                            gp = [g * (r ** j) for j in range(n_gp)]
                            
                            # Check bounds
                            # Quick check: if max is too big, skip
                            # r^15 for r=4 is huge. 
                            # We check max(ap+gp)
                            try:
                                max_val = max(max(ap), max(gp))
                            except ValueError:
                                continue
                                
                            if max_val > 10000:
                                continue
                            
                            # Union
                            candidates = set(ap + gp)
                            # Filter positive
                            candidates = {x for x in candidates if x > 0}
                            
                            if len(candidates) < 20:
                                continue
                                
                            # Take 20 smallest
                            A = sorted(list(candidates))[:20]
                            
                            # Calculate score
                            s = score(A)
                            
                            if s < best_score:
                                best_score = s
                                best_set = A
    
    if best_set is None:
        return list(range(1, 21))
        
    return best_set

def score(A):
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))
