"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # --- Initial solution: rank-matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Efficient: sum over i of a[i,:] dot b[p[i], p]
        # b_p = b[p, :] -> shape (n, n), row i is b[p[i], :]
        # cost = sum_i sum_j a[i,j] * b_p[i,j] = sum of element-wise product
        b_p = b[p, :]
        return int(np.sum(a * b_p))
    
    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute row sums of b for quick delta estimation? 
    # For 2-opt: moving facility i from location x to location y
    # Change in cost = sum_j a[i,j]*(b[y,j]-b[x,j]) + sum_j a[j,i]*(b[j,y]-b[j,x])
    # = sum_j (a[i,j]+a[j,i]) * (b[y,j]-b[x,j])
    
    # Let's use a 2-opt style local search with simulated annealing or hill climbing
    # Given time constraints, a fast hill-climbing with random restarts might work.
    
    # Precompute A_sym = a + a.T for faster delta calculation
    A_sym = a + a.T
    
    def try_improve(p, max_iters=5000):
        """Hill climbing: try to find a better location for each facility."""
        nonlocal current_cost
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Shuffle order of facilities to avoid bias
            order = np.random.permutation(n)
            for i in order:
                if time.time() > deadline:
                    return False
                x = p[i]
                # Find best y != x
                # Delta for moving i from x to y:
                # sum_j (a[i,j]+a[j,i]) * (b[y,j]-b[x,j])
                # Let v = A_sym[i, :] -> shape (n,)
                # Delta(y) = sum_j v[j]*b[y,j] - sum_j v[j]*b[x,j]
                # = v @ b[y] - v @ b[x]
                # We want to minimize cost, so we want the smallest Delta(y)
                # i.e., smallest v @ b[y]
                
                v = A_sym[i]
                # Compute v @ b[y] for all y
                scores = b @ v  # shape (n,), scores[y] = sum_j b[y,j]*v[j]
                scores[x] = np.inf  # exclude current location
                
                y = np.argmin(scores)
                delta = scores[y] - scores[x]
                
                if delta < 0:
                    # Move i to y
                    p[i] = y
                    current_cost += int(delta)
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p[:] = p
        return improved
    
    # Run hill climbing
    try_improve(p, max_iters=100)
    
    # Try a few random restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed solution
        p_new = best_p.copy()
        # Swap a few random pairs
        num_swaps = 10
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_new[i], p_new[j] = p_new[j], p_new[i]
        
        p = p_new
        current_cost = compute_cost(p)
        
        # Hill climb
        try_improve(p, max_iters=50)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
