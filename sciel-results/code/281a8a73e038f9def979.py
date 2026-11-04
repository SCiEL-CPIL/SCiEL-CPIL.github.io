import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-9):
        R = U.shape[1]
        if R > rank_schoolbook:
            return False
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
                return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if not np.isclose(val, 1.0, atol=tol):
                        return False
        return True

    # Construct a rank-120 decomposition using Kronecker embedding
    # Split n=4 into n1=4, n2=0 (effectively, we embed 4x5 as 4x4 and 1x5? No, that changes dimensions)
    # Let's embed 4x5 into 4x4 and 1x5? No, the matrix is 4x5.
    # We can embed A (4x5) into block diag(A1, A2) where A1 is 4x4 and A2 is 1x1? No.
    # Standard embedding: A (4x5) -> A' (5x6) by padding?
    # Let's use the embedding 4x5 -> 4x4 and 1x5 is not a standard square embedding.
    # However, we can embed 4x5 into 5x5 by padding a row of zeros?
    # Or embed 4x5 into 6x6?
    # The current program's inspiration mentioned Strassen 2x2.
    # Let's try to construct a specific rank-120 decomposition.
    # Rank 120 = 4 * 5 * 6 is schoolbook.
    # Can we get rank 119?
    # If we can find a rank-120 decomposition that is "sparse" or has structure, we might reduce it.
    
    # Actually, a known result for 4x5x6 might be rank 119 or lower?
    # Let's try to use the "Kronecker" idea to generate a valid rank-120 decomposition 
    # that is different from schoolbook, then try to reduce rank by 1 using exact linear algebra.
    
    # Embedding:
    # Let A be 4x5, B be 5x6.
    # Embed A into A_emb (5x6) by adding a row of zeros and a column of zeros?
    # If we embed A (4x5) into A' (5x6) and B (5x6) into B' (5x6),
    # then A' B' has the product in the top-left 4x6 block.
    # Rank of 5x5x6 multiplication?
    # Schoolbook rank for 5x5x6 is 150.
    # This doesn't help directly reduce 120.
    
    # Let's try a different strategy:
    # Start with schoolbook rank 120.
    # Try to find a linear combination of the 120 rank-1 tensors that is zero.
    # If we find a non-trivial dependency, we can reduce the rank.
    # The space of tensors is 4*5*6 = 120 dimensional.
    # The space of rank-1 tensors is larger.
    # If the 120 schoolbook tensors are linearly independent (they are, as they form a basis),
    # we cannot reduce the rank by combining them.
    # We need a different set of rank-1 tensors.
    
    # Let's try to use the ALS approach but with a better initialization and exact snapping.
    # The previous failure was likely due to floating point issues or getting stuck in local minima.
    # We will use a "randomized" schoolbook initialization:
    # Mix schoolbook columns with random columns, then ALS.
    
    def attempt_rank_als(R, seed):
        if R >= rank_schoolbook:
            return None
        
        np.random.seed(seed)
        # Initialize with a mix of schoolbook and random
        # To ensure we don't start at a trivial solution, we use random.
        # But random often fails to converge to integer solutions.
        # Let's try to initialize with schoolbook columns, but perturb them.
        
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Use first R-1 schoolbook columns
        for r in range(R - 1):
            i, j, k = r // (m*p), (r // p) % m, r % p
            U[i * m + j, r] = 1
            V[j * p + k, r] = 1
            W[k * n + i, r] = 1
        
        # Last column is random
        U[:, R-1] = np.random.randn(n * m)
        V[:, R-1] = np.random.randn(m * p)
        W[:, R-1] = np.random.randn(p * n)
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0)**2
            
            if err < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                
                if verify(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
            
            if time.time() > deadline:
                return None
        
        return None

    # Try to find a lower rank decomposition
    # We will try ranks from 119 down to 115
    # We will try multiple seeds for each rank to avoid local minima
    
    for R in range(119, 114, -1):
        for seed in range(10):
            res = attempt_rank_als(R, seed)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() > deadline:
            break

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
