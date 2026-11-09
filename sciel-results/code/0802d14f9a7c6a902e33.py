import time
import random
from math import gcd

def construct():
    """
    Finds maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Uses randomized greedy with constraint propagation.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
            
    # Precompute conflict masks?
    # For each pair (i, j), and residue r_i, which residues r_j are forbidden?
    # This is too big.
    
    best_result = []
    best_size = 0
    
    def run_greedy(order):
        nonlocal best_result, best_size
        
        available = [set(range(all_moduli[i])) for i in range(n)]
        assignment = {}
        selected = set()
        
        for idx in order:
            if not available[idx]:
                continue
                
            m = all_moduli[idx]
            valid_rs = list(available[idx])
            
            if not valid_rs:
                continue
            
            # Pick random residue
            r = random.choice(valid_rs)
            
            assignment[idx] = r
            selected.add(idx)
            
            # Propagate constraints
            for k in range(n):
                if k in selected or k == idx:
                    continue
                g = gcd_mat[idx][k]
                if g == 1:
                    continue
                # Remove residues from available[k] that conflict with r
                # Conflict if (rk - r) % g == 0
                to_remove = set()
                for rk in available[k]:
                    if (rk - r) % g == 0:
                        to_remove.add(rk)
                if to_remove:
                    available[k] -= to_remove
        
        if len(assignment) > best_size:
            best_size = len(assignment)
            best_result = [[assignment[idx], all_moduli[idx]] for idx in assignment]
            
        return len(assignment)
    
    # Run many restarts
    # Estimate time per run
    t0 = time.time()
    run_greedy(list(range(n)))
    t1 = time.time()
    avg_time = t1 - t0
    if avg_time < 0.001:
        avg_time = 0.001
        
    remaining = deadline - time.time() - 5
    max_restarts = int(remaining / avg_time)
    max_restarts = max(100, min(max_restarts, 5000))
    
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        run_greedy(order)
        
    if not best_result:
        best_result = [[0, 2]]
        
    return best_result
