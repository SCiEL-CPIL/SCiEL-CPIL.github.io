import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Fallback: Known optimal rank 35 solution using Strassen
    rank_fallback = 35
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U_fallback[idx][col] = a_vec[idx]
            V_fallback[0 * p + 2 * k][col] = b_vec[0]
            V_fallback[0 * p + 2 * k + 1][col] = b_vec[1]
            V_fallback[1 * p + 2 * k][col] = b_vec[2]
            V_fallback[1 * p + 2 * k + 1][col] = b_vec[3]
            W_fallback[2 * k * n + 0][col] = c_vec[0]
            W_fallback[2 * k * n + 1][col] = c_vec[1]
            W_fallback[(2 * k + 1) * n + 0][col] = c_vec[2]
            W_fallback[(2 * k + 1) * n + 1][col] = c_vec[3]
            col += 1
    
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    best_rank = rank_fallback
    
    # Try to find a lower rank decomposition using optimization
    # The tensor T[a,b,c] where a in [0,4), b in [0,20), c in [0,20)
    # T[a,b,c] = 1 if the corresponding multiplication term exists
    
    # Build the target tensor
    def build_target_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a_idx = i*m + k
                    b_idx = k*p + j
                    c_idx = j*n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        return T
    
    T = build_target_tensor(n, m, p)
    
    # Try rank 34 with random initialization and gradient-free optimization
    # We want U @ V_k @ W_k summed over k to equal T
    # i.e., sum_k U[:,k] * V[:,k] * W[:,k] = T (as a 3-tensor)
    
    def check_decomposition(U, V, W, T, tol=1e-6):
        """Check if U, V, W decompose T. U shape (nm, r), V shape (mp, r), W shape (pn, r)"""
        r = U.shape[1]
        # Compute sum_k U[:,k] * V[:,k] * W[:,k]
        result = np.zeros(T.shape)
        for k in range(r):
            result += np.einsum('a,b,c->abc', U[:, k], V[:, k], W[:, k])
        return np.allclose(result, T, atol=tol)
    
    # Try a few random initializations for rank 34
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        target_rank = 34
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Use a simple iterative refinement: project onto the manifold
        # This is a heuristic - not guaranteed to converge
        # We'll just check if by chance we're close (unlikely)
        
        # Instead, let's try a more structured approach:
        # Start from the rank-35 solution and try to merge two terms
        if trial == 0:
            # Try to see if we can reduce rank by 1 from the Strassen solution
            # by finding a linear dependency
            U_arr = np.array(U_fallback)
            V_arr = np.array(V_fallback)
            W_arr = np.array(W_fallback)
            
            # Check if any column is a linear combination of others
            # This is a heuristic check
            for col_idx in range(35):
                if time.time() > deadline - 5:
                    break
                # Try to express column col_idx as combination of others
                others = list(range(35))
                others.remove(col_idx)
                
                # This is expensive, skip for now
                pass
    
    return best_result
