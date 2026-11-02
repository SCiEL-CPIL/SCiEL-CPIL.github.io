"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank  
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]
    
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s limit
    
    # Schoolbook decomposition as baseline (Rank 80)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Target Tensor T of size (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exist i, j, k such that:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # This is the standard matrix multiplication tensor.
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = get_target_tensor()
    
    # Strategy: Randomized search for low-rank decompositions
    # We try ranks lower than schoolbook.
    # Known bounds: For 4x4x4, rank is 49 (Strassen-like improvements exist but 49 is standard best known for 4x4x4? 
    # Actually, best known for 4x4x4 is 49. For 4x4x5, it's likely between 49 and 80.
    # Let's try ranks from 49 upwards, or random search.
    
    def try_rank(rank, num_iterations=1000, num_restarts=5):
        """Try to find a decomposition of given rank using gradient descent on factors."""
        current_time = time.time()
        if current_time > deadline:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize factors with small random values
            # Using normal distribution to allow exploration
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Scale to help convergence?
            # The tensor T has entries 0 or 1.
            
            # We want to minimize || T - sum_r U[:,r] V[:,r]^T W[:,r]^T ||_F
            # This is non-convex. We use a simple iterative update.
            
            # To speed up, we can use a fixed number of steps with adaptive step size.
            # Or use a more robust method like ALS (Alternating Least Squares) on the tensor.
            # ALS for tensor decomposition:
            # Fix V, W, solve for U.
            # U[:, r] = argmin_U || T - sum_r' U[:,r'] V[:,r']^T W[:,r']^T ||
            # This is a linear least squares problem for each r.
            
            # Let's implement a simplified ALS.
            # For each r:
            # Residual R = T - sum_{r' != r} U[:,r'] V[:,r']^T W[:,r']^T
            # We want to find U_r such that <U_r, V_r> * W_r^T approximates R.
            # This is still coupled.
            
            # Simpler approach: Gradient Descent with momentum.
            # Loss L = 0.5 * || T - sum_r U_r V_r^T W_r^T ||^2
            # dL/dU_r = (T - sum ... ) * (V_r, W_r)
            
            # Given the small size, we can afford full tensor operations.
            
            lr = 0.01
            momentum = 0.9
            v_U = np.zeros_like(U)
            v_V = np.zeros_like(V)
            v_W = np.zeros_like(W)
            
            for step in range(num_iterations):
                if time.time() > deadline:
                    return None
                
                # Compute current tensor estimate
                # S = sum_r U[:,r] V[:,r]^T W[:,r]^T
                # This can be done efficiently:
                # S = np.einsum('ar,br,cr->abc', U, V, W)
                S = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Error
                E = T - S
                
                # Gradients
                # dS/dU_a = sum_r E_{a,b,c} V_{b,r} W_{c,r}
                # This is einsum('abc,br,cr->ar', E, V, W)
                grad_U = np.einsum('abc,br,cr->ar', E, V, W)
                grad_V = np.einsum('abc,ar,cr->br', E, U, W)
                grad_W = np.einsum('abc,ar,br->cr', E, U, V)
                
                # Update with momentum
                v_U = momentum * v_U + lr * grad_U
                v_V = momentum * v_V + lr * grad_V
                v_W = momentum * v_W + lr * grad_W
                
                U -= v_U
                V -= v_V
                W -= v_W
                
                # Check for convergence or improvement
                # We want to check if we can round to integers/half-integers
                # and if that rounded version is exact.
                
                if step % 10 == 0:
                    # Try rounding to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    S_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    err = np.linalg.norm(T - S_r)
                    
                    if err < 1e-9:
                        # Found an exact decomposition!
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
                    # Also try integer rounding
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    S_i = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
                    err_i = np.linalg.norm(T - S_i)
                    if err_i < 1e-9:
                        return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                        
            # If we didn't find exact, return None for this restart
        return None

    # Try ranks in increasing order
    # Start from a reasonable lower bound.
    # For 4x4x4, min rank is 49.
    # For 4x4x5, it should be at least 49.
    # Let's try 49, 50, ... up to 79.
    # If we find one, it's better than 80.
    
    # To save time, we might skip some ranks or use a coarser search first.
    # Let's try a few specific ranks that are likely candidates.
    # 49 is the best for 4x4x4.
    # 50, 51...
    
    # We have ~140 seconds.
    # Each rank attempt takes some time.
    # Let's allocate time per rank.
    
    ranks_to_try = list(range(49, 80))
    
    # Shuffle to explore more broadly? Or sequential?
    # Sequential is better for finding the *minimal* rank.
    # But if 49 is impossible, we waste time.
    
    # Let's try 49 first.
    # If it fails, try 50, etc.
    
    # Optimization: Reduce iterations if time is tight.
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        res = try_rank(r, num_iterations=500, num_restarts=3)
        if res:
            # Verify
            U = np.array(res['U'])
            V = np.array(res['V'])
            W = np.array(res['W'])
            S = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(S, T, atol=1e-9):
                best_result = res
                best_rank = r
                break # Found minimal rank in this search space, stop?
                # Actually, we should check if we can go lower, but we started at 49.
                # If 49 works, it's likely optimal or near-optimal.
                # If we found 50, we should check 49 again? No, we tried 49 first.
                # So if we found r, it is the smallest r in our list that worked.
                
    return best_result

# EVOLVE-BLOCK-END
