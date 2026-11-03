import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 21  # Known or suspected minimum rank
    
    # Dimensions of the tensor
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a decomposition of rank r
    # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
    
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Use a simple iterative refinement or optimization
    # Since we need integer/half-integer, we can try to find a real solution and then round.
    
    # Initialize with random values
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        r = target_rank
        # Random initialization
        U = np.random.randn(dim_u, r)
        V = np.random.randn(dim_v, r)
        W = np.random.randn(dim_w, r)
        
        # Optimize using a few steps of gradient descent or ALS (Alternating Least Squares)
        # ALS is faster and often works well for tensor decomposition.
        
        for step in range(100):
            if time.time() > deadline:
                break
                
            # Update U
            # T = sum_c U_c V_c W_c
            # For fixed V, W, T is linear in U.
            # Residual R = T - sum_c U_c V_c W_c
            # We want to minimize ||R||^2.
            # This is equivalent to solving for each U[:, c]
            
            # Precompute V^T W terms?
            # T reshaped to (dim_u, dim_v * dim_w)
            T_mat = T.reshape(dim_u, dim_v * dim_w)
            
            # For each column c, U[:, c] should be the projection of T onto the subspace spanned by (V[:, c] \otimes W[:, c])?
            # Actually, the term is U[:, c] * (V[:, c] \otimes W[:, c])^T
            # So U[:, c] = (V[:, c] \otimes W[:, c])^T T / ||V[:, c] \otimes W[:, c]||^2
            
            # Let's do this for all c simultaneously using matrix operations if possible, or loop.
            # Loop is fine for small r.
            
            for c in range(r):
                v_w = np.outer(V[:, c], W[:, c])
                norm_sq = np.dot(v_w, v_w)
                if norm_sq < 1e-10:
                    continue
                # Project T onto v_w
                # U[:, c] = sum_{a,b} T[a,b,:] * v_w[a, b] / norm_sq ? No.
                # T[a, b, c_idx]
                # The contribution of column c to T is U[:, c] * v_w
                # We want U[:, c] such that U[:, c] * v_w approximates T - sum_{c' != c} ...
                # This is getting complicated. Let's use a simpler gradient step.
                
            # Simpler: Just use a few steps of random restart and check rank via SVD of the unfolded tensor?
            # No, we need the factors.
            
            # Let's try a different approach:
            # We know the structure. For 2x5x6, the decomposition might be related to 2x5x5.
            # If we can find a rank 16 decomposition for 2x5x5, we can extend it.
            
        # After some iterations, check if we can round to half-integers
        # This is a heuristic.
        
    # Fallback: Return the schoolbook decomposition if optimization fails
    # But we want to beat 1.0.
    
    # Let's try to construct a specific known decomposition.
    # For 2x2x2, rank 7.
    # For 2x5x6, rank 21 is a strong candidate.
    
    # Since I don't have the exact factors hardcoded, I will return a decomposition 
    # that I can verify is correct if I can find one.
    # However, without a verified decomposition, I risk returning an invalid one.
    
    # Safe strategy: Return the schoolbook decomposition (rank 60) which is guaranteed valid.
    # But the score is 1.0.
    
    # Can I find a rank 30 decomposition easily?
    # If I split the 5x6 matrix into two 5x3 matrices, and use rank 15 for each?
    # Rank of 2x5x3 is 11?
    # 11 + 11 = 22.
    
    # Let's try to return a rank 22 decomposition if I can construct it.
    # Or just return the schoolbook one to be safe, but that's not an improvement.
    
    # Wait, the task says "integer/half-integer entries".
    # If I return floats, it might fail.
    
    # I will return the schoolbook decomposition as a baseline, 
    # but I will try to optimize it slightly if possible.
    
    # Actually, let's just return the schoolbook decomposition. 
    # It's the only one I can guarantee is correct and valid.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
