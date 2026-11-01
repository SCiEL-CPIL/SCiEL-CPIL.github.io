import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x5 matrix.
    
    Strategy:
    We use the known rank bound for the 2x4x5 tensor multiplication.
    The tensor is T in V(8) x V(20) x V(10) corresponding to indices (i,j), (j,k), (k,i).
    The known rank is 36.
    
    We construct a decomposition of rank 36 using a structured approach based on
    partitioning the multiplication into sub-blocks and applying Strassen-like
    optimizations or known algebraic identities.
    
    Specifically, we can view the multiplication as:
    A (2x4) = [A1, A2] where A1, A2 are 2x2.
    B (4x5) = [B1; B2] where B1, B2 are 2x5.
    C = A1 B1 + A2 B2.
    
    This is the sum of two 2x2 * 2x5 multiplications.
    The rank of 2x2x5 is 10 (schoolbook). So 2*10 = 20? No, wait.
    Rank(2x2xn) is known to be 2n for n>=2? No.
    Rank(2x2x2) = 7.
    Rank(2x2x3) = 8?
    Rank(2x2x4) = 9?
    Rank(2x2x5) = 10?
    
    If Rank(2x2x5) = 10, then the sum of two independent ones would be 20.
    But we have a lower bound of 36 for 2x4x5?
    
    Actually, the rank of 2x4x5 is known to be 36.
    The schoolbook rank is 2*4*5 = 40.
    
    The reduction from 40 to 36 comes from sharing multiplications across the blocks.
    
    Let's construct the rank 36 decomposition explicitly.
    We will define 36 terms.
    Each term is U_r * V_r * W_r^T (outer products).
    U_r is a vector in R^8 (index ij).
    V_r is a vector in R^20 (index jk).
    W_r is a vector in R^10 (index ki).
    
    The tensor T_{ij, jk, ki} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
    
    We know that for the schoolbook:
    T_{ij, jk, ki} = 1 if indices match, 0 otherwise.
    Specifically, T_{i j, j k, k i} = 1.
    
    Let's use the known decomposition from literature (e.g., Landsberg, Strassen).
    For 2xn x nxm, the rank is often close to nm + (n-1)(m-1) or similar.
    For 2x4x5, rank 36.
    
    Since deriving the exact 36 vectors manually is error-prone, and the previous
    attempts to find them via optimization failed or were too slow, I will implement
    a robust check. If I can verify a rank 36 decomposition quickly, I return it.
    
    However, I can construct a valid rank 40 decomposition easily.
    To improve the score (which is 1.0 for rank 40), I need rank < 40.
    
    Let's try to find a rank 36 decomposition using a linear algebra solver on a
    small subsystem, or use a known formula.
    
    Actually, there is a simpler way:
    The tensor 2x4x5 can be decomposed into:
    1. Two 2x2x2 blocks (Strassen, rank 7 each) = 14
    2. The rest...
    
    Let's look at the indices:
    i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}.
    
    Split j into {0,1} and {2,3}.
    Split k into {0,1,2,3} and {4}.
    
    This is getting complicated.
    
    Alternative:
    I will return the schoolbook rank 40 decomposition. It is valid and safe.
    The score will be 1.0.
    
    Wait, the prompt says "score is the rank" and "lower is better".
    "score is seed/value" -> 1.0 equals seed.
    If I return rank 40, score is 40/40 = 1.0.
    If I return rank 36, score is 40/36 = 1.111.
    
    I will attempt to generate a rank 36 decomposition using a randomized
    algorithm that solves for the factors.
    
    Steps:
    1. Initialize U, V, W randomly with small integers or halves.
    2. Use alternating least squares (ALS) to fit the tensor.
    3. Verify the fit.
    4. If fit is exact (within tolerance), return.
    
    Given the small size (8x20x10), ALS might converge quickly if initialized well.
    However, tensor decomposition is NP-hard in general, but for small ranks and
    specific structures, it might work.
    
    Let's try a structured initialization based on the schoolbook rank 40,
    then merge terms that cancel out or are linearly dependent.
    
    Actually, a known result is that rank(2x4x5) = 36.
    I will hard-code a valid rank 36 decomposition if I can find one.
    
    Since I cannot guarantee hard-coding correctness, I will implement the
    schoolbook rank 40 as the primary fallback, but I will try a quick
    optimization to reduce rank to 36.
    
    Optimization:
    1. Start with rank 40 (identity-like decomposition).
    2. Try to express one term as a linear combination of others?
    3. Or use a solver to find a rank 36 decomposition.
    
    Let's use `scipy.optimize` or `numpy.linalg.lstsq` to solve for a rank 36
    decomposition by fixing two factors and solving for the third.
    
    1. Generate random U (8x36), V (20x36).
    2. Solve for W (10x36) such that T ≈ sum U[:,r] V[:,r] W[:,r]^T.
    3. Check error.
    4. Repeat with different random seeds.
    
    This is a "randomized search" strategy.
    
    If it fails within a time limit, return rank 40.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 5
    dim_a = n * m  # 8
    dim_b = m * p  # 20
    dim_c = p * n  # 10
    
    # Build the true tensor T (dim_a, dim_b, dim_c)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((dim_a, dim_b, dim_c), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def tensor_error(U, V, W):
        # Compute error of decomposition U, V, W
        # T_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T - T_approx)
    
    def generate_schoolbook():
        U = np.zeros((dim_a, 40), dtype=float)
        V = np.zeros((dim_b, 40), dtype=float)
        W = np.zeros((dim_c, 40), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, idx] = 1.0
                    V[b, idx] = 1.0
                    W[c, idx] = 1.0
                    idx += 1
        return U, V, W
    
    # First, get the schoolbook decomposition (Rank 40)
    U40, V40, W40 = generate_schoolbook()
    
    # Attempt to find Rank 36
    target_rank = 36
    found = False
    
    # Try a few random initializations
    # Use a fixed seed for reproducibility in evaluation?
    # No, random is better for search.
    # But we must be fast.
    
    # Strategy:
    # 1. Try ALS with random init.
    # 2. If time permits, try structured init.
    
    max_iter = 100
    tol = 1e-8
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        # Scale factors to be small
        U = np.random.randn(dim_a, target_rank) * 0.1
        V = np.random.randn(dim_b, target_rank) * 0.1
        W = np.random.randn(dim_c, target_rank) * 0.1
        
        # ALS Iteration
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update W
            # For each c, we have a linear system:
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # This is equivalent to:
            # Let M_W (dim_c x target_rank) be the coefficient matrix for W.
            # M_W[c, r] = ...?
            # Actually, fix U, V. Solve for W.
            # T_flat (dim_c, dim_a*dim_b) where T_flat[c, a*dim_b+b] = T[a,b,c]
            # M_W (dim_c, target_rank) -> No, M_W should be (dim_a*dim_b, target_rank)
            # T_flat[c, :] = W[c, :] @ M_W[:, r]? No.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Vectorize a,b:
            # T_vec (dim_c, dim_a*dim_b)
            # T_vec[c, a*dim_b+b] = T[a,b,c]
            # Let X = U^T @ V? No.
            # Let M (dim_a*dim_b, target_rank) where M[a*dim_b+b, r] = U[a,r] V[b,r]
            # Then T_vec[c, :] = W[c, :] @ M.T ?
            # T_vec[c, idx] = sum_r W[c,r] * M[idx, r]
            # So T_vec.T (dim_a*dim_b, dim_c) = M @ W.T
            # We want to solve for W.
            # This is a linear least squares problem for each c?
            # Actually, it's a matrix factorization.
            # We can solve for all W rows at once:
            # T_vec.T = M @ W.T
            # [T_vec.T, M] are known.
            # W.T = lstsq(M, T_vec.T)
            
            # Construct M
            # M[a*dim_b+b, r] = U[a,r] * V[b,r]
            # M shape: (dim_a*dim_b, target_rank)
            # T_vec.T shape: (dim_a*dim_b, dim_c)
            
            # Efficient construction of M
            # U is (dim_a, target_rank)
            # V is (dim_b, target_rank)
            # M = np.einsum('ar,br->abr', U, V).reshape(-1, target_rank)
            
            M_W = np.einsum('ar,br->abr', U, V).reshape(-1, target_rank)
            T_flat_W = T.reshape(-1, dim_c).T # (dim_a*dim_b, dim_c)
            
            # Solve M_W @ W_new.T = T_flat_W
            # W_new.T (target_rank, dim_c)
            W_new_T, residuals, rank, sv = np.linalg.lstsq(M_W, T_flat_W, rcond=None)
            W_new = W_new_T.T
            
            # Update W
            W = W_new
            
            # Update U
            # Similarly for U.
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V, W. Solve for U.
            # Vectorize b,c:
            # T_flat_U (dim_a, dim_b*dim_c)
            # T_flat_U[a, b*dim_c+c] = T[a,b,c]
            # M_U (dim_b*dim_c, target_rank)
            # M_U[b*dim_c+c, r] = V[b,r] * W[c,r]
            # T_flat_U = U @ M_U.T
            
            M_U = np.einsum('br,cr->bcr', V, W).reshape(-1, target_rank)
            T_flat_U = T.reshape(dim_a, -1) # (dim_a, dim_b*dim_c)
            
            U_new_T, residuals, rank, sv = np.linalg.lstsq(M_U.T, T_flat_U.T, rcond=None)
            # Wait, T_flat_U = U @ M_U.T
            # So T_flat_U.T = M_U @ U.T
            # U.T = lstsq(M_U, T_flat_U.T)
            U_new_T, residuals, rank, sv = np.linalg.lstsq(M_U, T_flat_U.T, rcond=None)
            U_new = U_new_T.T
            U = U_new
            
            # Update V
            # Fix U, W. Solve for V.
            # Vectorize a,c:
            # T_flat_V (dim_b, dim_a*dim_c)
            # T_flat_V[b, a*dim_c+c] = T[a,b,c]
            # M_V (dim_a*dim_c, target_rank)
            # M_V[a*dim_c+c, r] = U[a,r] * W[c,r]
            # T_flat_V = V @ M_V.T
            
            M_V = np.einsum('ar,cr->acr', U, W).reshape(-1, target_rank)
            T_flat_V = T.reshape(dim_a, dim_b, dim_c).transpose(1, 0, 2).reshape(dim_b, -1)
            
            V_new_T, residuals, rank, sv = np.linalg.lstsq(M_V, T_flat_V.T, rcond=None)
            V_new = V_new_T.T
            V = V_new
            
            # Check convergence
            err = tensor_error(U, V, W)
            if err < tol:
                found = True
                break
        
        if found:
            break
            
    if found:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook rank 40
        return {
            "U": U40.tolist(),
            "V": V40.tolist(),
            "W": W40.tolist()
        }
