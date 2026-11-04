import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Ensure we are working with float64 for stability and speed in some BLAS ops, 
    # though int64 is fine for cost. We will use float64 for delta calculations to avoid overflow 
    # if values are large, though QAPLIB tho150 values are small.
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # 1. Initial Heuristic: Rank Matching
    # Heavy flows (high row sum in A) should go to central locations (low row sum in B)
    a_sums = A.sum(axis=1)
    b_sums = B.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_facs = np.argsort(-a_sums, kind='stable')
    # Sort locations by ascending distance sum
    central_locs = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facs] = central_locs
    
    # Calculate initial cost
    # Cost = sum(A[i, j] * B[p[i], p[j]])
    # We can compute this as trace(A * B[p[:, None], p[None, :]])? No.
    # It's sum over i,j.
    # Efficient way:
    # C = A * B[p[:, None], p[None, :]]
    # cost = C.sum()
    # But B[p[:, None], p[None, :]] creates an n x n matrix.
    # For n=150, this is 22,500 elements. Very fast.
    
    def compute_cost(p):
        # p is facility -> location
        # B_perm[i, j] = B[p[i], p[j]]
        B_perm = B[p[:, None], p[None, :]]
        return np.sum(A * B_perm)
    
    current_cost = compute_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    timeout = 145.0 # Leave 5s margin
    
    # Simulated Annealing Parameters
    # T0 should be high enough to accept most moves initially
    # Estimate average delta
    # For random swaps, delta is roughly related to variance of A and B.
    T = 1000.0
    T_min = 1e-5
    # Decay rate
    # We want to reach T_min in about 80% of the time budget if we assume uniform steps.
    # But step count depends on speed.
    # Let's use a fixed number of iterations or time-based decay.
    # Better: decay T based on time remaining or fixed steps.
    # Let's try fixed steps first, then switch to 2-opt.
    
    # Precompute nothing specific for delta except B columns which are accessed dynamically.
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        # Pick two random distinct indices
        i, j = rng.choice(n, size=2, replace=False)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Calculate Delta
        # Delta = sum_{k != i,j} (A[k,i] - A[k,j]) * (B[p[k], v] - B[p[k], u]) 
        #         + A[i,i]*(B[u,v]-B[u,u]) + A[j,j]*(B[u,v]-B[v,v])
        
        # Vectorized part
        # V[k] = B[p[k], v] - B[p[k], u]
        # We can get B[:, v] and B[:, u]
        col_v = B[:, v]
        col_u = B[:, u]
        
        # We need B[p[k], v] which is col_v[p[k]]
        # And B[p[k], u] which is col_u[p[k]]
        # So V = col_v[p] - col_u[p]
        V = col_v[p] - col_u[p]
        
        # W = A[:, i] - A[:, j]
        W = A[:, i] - A[:, j]
        
        # Sum part = dot(W, V)
        # Subtract terms for k=i and k=j
        # Term i: W[i] * V[i]
        # Term j: W[j] * V[j]
        
        sum_part = np.dot(W, V) - W[i] * V[i] - W[j] * V[j]
        
        # Self terms
        # A[i,i]*(B[u,v]-B[u,u]) + A[j,j]*(B[v,u]-B[v,v])
        # Note B is symmetric so B[u,v] = B[v,u]
        self_part = A[i, i] * (B[u, v] - B[u, u]) + A[j, j] * (B[u, v] - B[v, v])
        
        delta = sum_part + self_part
        
        # Acceptance
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Swap
            p[i], p[j] = v, u
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Decay Temperature
        # Simple exponential decay per iteration is risky if speed varies.
        # Let's use a time-based or iteration-based decay.
        # Assume ~10000 iterations per second.
        # Let's just decay T by a factor each iteration.
        T *= 0.99995
        
        # If T is very low, we can break to 2-opt
        if T < T_min:
            break
            
    # Final 2-opt Polish
    # Iterate until no improvement
    improved = True
    while improved and (time.time() - start_time < timeout):
        improved = False
        for i in range(n):
            if time.time() - start_time > timeout:
                break
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                
                # Recalculate delta
                col_v = B[:, v]
                col_u = B[:, u]
                V = col_v[p] - col_u[p]
                W = A[:, i] - A[:, j]
                sum_part = np.dot(W, V) - W[i] * V[i] - W[j] * V[j]
                self_part = A[i, i] * (B[u, v] - B[u, u]) + A[j, j] * (B[u, v] - B[v, v])
                delta = sum_part + self_part
                
                if delta < 0:
                    p[i], p[j] = v, u
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()
