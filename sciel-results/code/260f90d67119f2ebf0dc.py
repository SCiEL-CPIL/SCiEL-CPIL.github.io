"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, tol=1e-9):
    """Check if the decomposition exactly decomposes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                uv = u_col[i * m + j] * v_col[j * p:]
                w_idx = np.array([k * n + i for k in range(p)])
                T[i, j, :] += uv * w_col[w_idx]
    return np.allclose(T, 1.0, atol=tol)

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are integer or half-integer."""
    arr = np.array(arr, dtype=float)
    doubled = 2.0 * arr
    nearest = np.round(doubled)
    return np.allclose(doubled, nearest, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank-46 decomposition using Strassen-inspired approach
    # The idea: use 7 Strassen multiplications for 2x2 blocks
    # For 2x5x5, we can split into blocks and apply Strassen
    
    def strassen_based_decomposition():
        """
        Construct a rank-46 decomposition using Strassen's algorithm.
        
        We view the 2x5x5 multiplication as:
        A (2x5) * B (5x5) = C (2x5)
        
        Split B into columns: B = [B_col0, B_col1, ..., B_col4]
        Then C[i,k] = sum_j A[i,j] * B[j,k]
        
        Key insight: Use Strassen on pairs of (i,k) computations.
        For fixed j, we have A[i,j] * B[j,k] for i in {0,1}, k in {0..4}.
        
        Alternative: Use the fact that we can combine computations.
        """
        # Build a rank-46 decomposition explicitly
        # Start from schoolbook (rank 50) and find 4 linear dependencies
        
        # Schoolbook tensors: T_{i,j,k} = e_i ⊗ e_j ⊗ e_k for i in 0..1, j in 0..4, k in 0..4
        # We need to find 4 linear combinations that can be replaced
        
        # Use Strassen: For computing 2 outputs from 2 inputs with 5 intermediate:
        # M1 = (A0 + A1) * (B0 + B1) -> but B is 5x5, not 2x2
        
        # Let's try a different approach: use the known result that
        # rank(2x5x5) <= 46 by finding explicit rank-46 decomposition
        
        # Construct using block Strassen on a 2x2 sub-block:
        # Treat the 5x5 B as having a 2x2 block structure (partially)
        
        # Actually, let's use a simpler approach:
        # We know that if we can find a rank-r decomposition where r < 50,
        # we can verify it. Let's try to construct one.
        
        # Use the following construction:
        # For the 2x5x5 case, we can use 46 terms by exploiting
        # linear dependencies in the tensor space.
        
        # The tensor T = sum_{i,j,k} T_{i,j,k} ⊗ e_i ⊗ e_j ⊗ e_k
        # where T_{i,j,k} is the (i,j,k)-th entry of the multiplication tensor.
        
        # We want to write T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
        
        # Start with schoolbook and try to merge terms.
        
        # Let's try: combine T_{0,j,k} and T_{1,j,k} for some j,k
        # If we can write T_{0,j,k} + T_{1,j,k} = u ⊗ v ⊗ w for some u,v,w,
        # that saves 1 rank.
        
        # Actually, let's just do a search for rank 46, 47, 48, 49
        pass
    
    def try_rank_exact(rank, num_trials=50):
        """Try to find a rank-rank decomposition with exact half-integer entries."""
        best = None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return best
            
            np.random.seed(42 + trial)
            
            # Initialize U, V with random half-integers
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(n * m, rank))
            V = np.random.choice(vals, size=(m * p, rank))
            
            # Solve for W
            W = np.zeros((p * n, rank))
            success = True
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve M @ w = ones(m)
                    try:
                        # Check if M has full row rank
                        rank_m = np.linalg.matrix_rank(M)
                        if rank_m < m:
                            # Use least squares
                            w, residuals, rank2, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        else:
                            w = np.linalg.solve(M, np.ones(m))
                    except:
                        success = False
                        break
                    
                    W[k * n + i, :] = w
                    
                    # Check half-integer
                    if not _is_half_integer(w, tol=1e-6):
                        success = False
                        break
                if not success:
                    break
            
            if success:
                # Verify
                if _check_decomposition(U, V, W, tol=1e-7):
                    # Round
                    U = np.round(2 * U) / 2
                    V = np.round(2 * V) / 2
                    W = np.round(2 * W) / 2
                    
                    # Re-verify after rounding
                    if _check_decomposition(U, V, W, tol=1e-7):
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        return best
        
        return best
    
    # Try ranks from 46 up to 49
    for rank in range(46, 50):
        if time.time() > deadline:
            break
        
        result = try_rank_exact(rank, num_trials=30)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
