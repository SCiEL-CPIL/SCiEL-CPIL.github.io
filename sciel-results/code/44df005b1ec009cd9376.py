"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Precompute row sums for initial heuristic
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)

    # 1. Initial Solution: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx

    # Calculate initial cost
    cost = np.sum(a * b[p])
    
    best_cost = cost
    best_p = p.copy()

    # Precompute b[p] for faster access in delta computation
    # We'll maintain B_p = b[p] as a matrix where B_p[i] = b[p[i]]
    # But actually we need b[u, p] and b[v, p] which are rows of b indexed by p
    # Let's just compute delta directly

    def get_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        diff_a = a[i] - a[j]
        diff_b = b[v, p] - b[u, p]
        delta = np.dot(diff_a, diff_b) + 2 * a[i, j] * b[u, v]
        return delta

    # Multi-start local search with 2-opt and perturbation
    # We'll do several restarts if time permits
    
    num_restarts = 0
    max_restarts = 50
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        # Local search: first-improvement 2-opt
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            # Randomize pair order for diversity
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() - start_time > time_limit:
                    break
                
                delta = get_delta(i, j, p, a, b)
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_p = p.copy()
                    
                    break  # Restart scan after improvement
        
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # If we have time for more restarts, perturb and continue
        num_restarts += 1
        if num_restarts >= max_restarts:
            break
        
        # Perturbation: 2-opt move (swap a random pair)
        # Or use Or-opt style: pick 2-3 consecutive facilities and reinsert
        # Simple approach: swap 2 random facilities
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        while i == j:
            j = random.randint(0, n - 1)
        
        p[i], p[j] = p[j], p[i]
        cost = np.sum(a * b[p])
        
        # If perturbed solution is better, update best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
