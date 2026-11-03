import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Structural Block Decomposition
    # Split A (3x5) into A_top (2x5) and A_bottom (1x5)
    # Split B (5x6) into B (5x6)
    # Output C (3x6) = [A_top @ B; A_bottom @ B]
    
    # Block 1: A_top (2x5) @ B (5x6) -> C_top (2x6)
    # We can view this as (2x2) @ (2x3) if we group columns? No.
    # Let's use the Kronecker structure proposed:
    # Rank 7 Strassen for 2x2x2.
    # The "1x5x1" part needs to handle the 5 intermediate dimension.
    # Actually, a simpler structural approach for 2x5x6:
    # We can decompose the 5 intermediate dimension into 2+3 or similar?
    # Or use the fact that 2x5x6 can be done with rank <= 2*5*6 = 60 (schoolbook).
    # But we want to beat 90.
    
    # Let's try a different structural split that is known to be efficient.
    # For 3x5x6, we can split the 3 rows into 2+1.
    # The 2x5x6 block can be computed using a rank-15 decomposition?
    # Wait, the inspiration mentioned "Rank-7 Strassen 2x2x2 and rank-4 Winograd 1x5x1".
    # This implies a tensor product structure.
    # T_{3,5,6} = T_{2,5,6} \oplus T_{1,5,6}
    
    # Let's construct the 2x5x6 decomposition first.
    # We can treat the 5 intermediate dimension as a single dimension 'm=5'.
    # Is there a known low-rank decomposition for 2x5x6?
    # Schoolbook is 60.
    # If we use Strassen for the 2x2 part of the rows, and some optimization for the 5x6?
    
    # Alternative: Use the explicit Strassen decomposition for 2x2x2 and pad/embed.
    # But 5 and 6 are not powers of 2.
    
    # Let's try a simpler, guaranteed improvement:
    # Use the known rank-23 decomposition for 3x3x3 and embed?
    # 3x5x6 contains 3x3x3 sub-blocks.
    
    # Actually, let's look at the "Winograd" idea.
    # Winograd's algorithm for 2x2 convolution uses rank 4 instead of 4 (no gain) or 8->4?
    # For matrix multiplication, Winograd's method for 2x2x2 is Strassen (rank 7).
    
    # Let's try to construct a rank-40 decomposition manually or via a simple heuristic.
    # If we can't find a specific structural one quickly, we fall back to a better greedy.
    # But the directive says "Structural change".
    
    # Let's implement a specific structural decomposition for 2x5x6 using Kronecker products.
    # We can split the 5 columns of A into 2+3 and 6 columns of B into 2+4?
    # This gets complicated.
    
    # Let's try a different structural approach:
    # Decompose 3x5x6 into 3x3x3 + 3x2x3 + 3x3x3 + 3x2x3 ...?
    
    # Given the time constraint and the difficulty of finding a known <90 rank decomposition
    # for 3x5x6 without deep research, I will implement a **Hybrid Strategy**:
    # 1. Try to construct a rank-40 decomposition using a specific block structure (2x5x6 via 2x2 Strassen x 5x6 padding? No).
    # 2. If that fails verification, fall back to a **Randomized Greedy** with better initialization (SVD-based) 
    #    which is a different optimization method than the naive greedy in the current program.
    
    # Actually, the current program's greedy is quite weak. 
    # A structural change could be using **ALS (Alternating Least Squares)** on the full tensor 
    # to find a low-rank approximation, then **Error Repair** to make it exact.
    
    # Let's implement ALS-based search.
    
    def _als_decompose(target_tensor, rank, n_iters=100, n_restarts=5):
        """
        Find a CP decomposition of rank 'rank' that approximates target_tensor.
        Returns U, V, W matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
        """
        n, m, p = DIMENSIONS
        T = target_tensor # shape (n, m, p)
        
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for r in range(n_restarts):
            # Random init
            U = np.random.randn(n, m, rank)
            V = np.random.randn(m, p, rank)
            W = np.random.randn(p, n, rank)
            
            # Normalize
            U = U / np.linalg.norm(U, axis=(0,1), keepdims=True)
            V = V / np.linalg.norm(V, axis=(0,1), keepdims=True)
            W = W / np.linalg.norm(W, axis=(0,1), keepdims=True)
            
            for it in range(n_iters):
                # Update U
                # U[i,j] = sum_k T[i,j,k] V[j,k] W[k,i] / sum_k (V[j,k] W[k,i])^2
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        vec = V[j, :, :] * W[:, i, :] # (p, rank)
                        denom = np.sum(vec * vec, axis=0)
                        numer = np.sum(T[i, j, :, :] * vec, axis=0)
                        mask = denom > 1e-10
                        U_new[i, j, mask] = numer[mask] / denom[mask]
                U = U_new
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        vec = U[:, j, :] * W[k, :, :] # (n, rank)
                        denom = np.sum(vec * vec, axis=0)
                        numer = np.sum(T[:, j, k, :] * vec, axis=0)
                        mask = denom > 1e-10
                        V_new[j, k, mask] = numer[mask] / denom[mask]
                V = V_new
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        vec = U[i, :, :] * V[:, k, :] # (m, rank)
                        denom = np.sum(vec * vec, axis=0)
                        numer = np.sum(T[i, :, k, :] * vec, axis=0)
                        mask = denom > 1e-10
                        W_new[k, i, mask] = numer[mask] / denom[mask]
                W = W_new
                
            # Calculate error
            approx = np.einsum('ijr,jkr,ikr->ijk', U, V, W)
            err = np.linalg.norm(T - approx)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-8:
                break
                
        # Reshape to column vectors
        U_mat = best_U.transpose(2, 0, 1).reshape(rank, n * m).T
        V_mat = best_V.transpose(2, 0, 1).reshape(rank, m * p).T
        W_mat = best_W.transpose(2, 0, 1).reshape(rank, p * n).T
        
        return U_mat, V_mat, W_mat, best_err

    # The target tensor is all 1s
    T = np.ones((n, m, p))
    
    # Try to find a decomposition with rank < 90
    # We know 90 is schoolbook.
    # Let's try ranks: 80, 70, 60, 50, 40
    # If we find one with low error, we attempt to "repair" it to exact integers/half-integers.
    
    best_rank = 90
    best_U, best_V, best_W = None, None, None
    
    # Generate schoolbook as fallback
    U_school = np.zeros((n * m, 90))
    V_school = np.zeros((m * p, 90))
    W_school = np.zeros((p * n, 90))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try ALS for lower ranks
    for rank in [60, 50, 40, 30]:
        if time.time() > deadline:
            break
        
        U_als, V_als, W_als, err = _als_decompose(T, rank, n_iters=50, n_restarts=3)
        
        if err < 1e-5:
            # Attempt to round to half-integers
            U_rounded = np.round(U_als * 2) / 2
            V_rounded = np.round(V_als * 2) / 2
            W_rounded = np.round(W_als * 2) / 2
            
            if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
                best_rank = rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break # Found a valid lower rank, stop searching lower? 
                # Actually, we should check even lower, but time is limited.
                # If we found 40, we can try 30, but 30 is very hard.
                # Let's just return the first valid one found in this loop.
        
        # If error is low but rounding failed, we might try to adjust?
        # For now, skip.

    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = 90

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum"""
    rank = U.shape[1]
    if rank == 0:
        return False
        
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    U_sub = U[u_idx, :]
    V_sub = V[v_idx, :]
    W_sub = W[w_idx, :]
    
    result = np.einsum('ic,ic,ic->i', U_sub, V_sub, W_sub)
    
    return np.allclose(result, 1.0, atol=1e-9)

def _find_rank(rank, n, m, p, max_time=5.0):
    """Placeholder to keep interface if needed"""
    return None
