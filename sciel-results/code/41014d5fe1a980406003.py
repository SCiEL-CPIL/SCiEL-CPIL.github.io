"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    Uses structured search with fast vectorized verification.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using NumPy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the full tensor from decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape and compute efficiently
        
        # For each column c, the contribution to T is:
        # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Sum over c
        
        # Vectorized approach:
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We need to compute for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        
        # Let's build it step by step
        # First, for each c, form the outer structure
        # This is equivalent to: sum_c (U_c reshaped to (n,m)) * (V_c reshaped to (m,p)) * (W_c reshaped to (p,n))
        # where * is element-wise after appropriate broadcasting
        
        # More efficient: iterate over rank but vectorize over i,j,k
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)  # (n, m)
            Vc = V[:, c].reshape(m, p)  # (m, p)
            Wc = W[:, c].reshape(p, n)  # (p, n)
            # Uc[i,j] * Vc[j,k] * Wc[k,i] for all i,j,k
            # This is a 3-tensor product
            # (n,m) * (m,p) * (p,n) -> sum over j,k? No, we need all i,j,k
            # Actually: for fixed i,j,k: Uc[i,j] * Vc[j,k] * Wc[k,i]
            # = (Uc[:,j] * Vc[j,:] * Wc[:,i]) summed? No.
            # Let's do: outer(Uc, Vc) gives (n,m,m,p), then multiply by Wc
            # Too large. Instead:
            # For each i: Uc[i,:] * Wc[:,i] gives (m,p) which is Uc[i,j]*Wc[k,i] for all j,k
            # Then multiply element-wise by Vc[j,k]
            for i in range(n):
                # Uc[i, j] for all j: shape (m,)
                # Wc[k, i] for all k: shape (p,)
                # Product: Uc[i,:] * Wc[:,i] -> (m, p) via outer
                contrib = np.outer(Uc[i, :], Wc[:, i])  # (m, p), contrib[j,k] = Uc[i,j]*Wc[k,i]
                total[i, :, :] += contrib * Vc  # element-wise multiply by Vc[j,k]
        
        if np.max(np.abs(total - 1.0)) > tol:
            return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition (rank n*m*p)."""
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
        return U, V, W
    
    def try_random_decomposition(rank, seed=0, sparsity=0.3):
        """Try random decomposition with given sparsity."""
        np.random.seed(seed)
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Random non-zero entries
        mask_U = np.random.random((n*m, rank)) < sparsity
        mask_V = np.random.random((m*p, rank)) < sparsity
        mask_W = np.random.random((p*n, rank)) < sparsity
        
        U[mask_U] = np.random.choice([-1, 1], size=mask_U.sum())
        V[mask_V] = np.random.choice([-1, 1], size=mask_V.sum())
        W[mask_W] = np.random.choice([-1, 1], size=mask_W.sum())
        
        if verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
        return None
    
    def try_structured_search(rank, max_attempts=50):
        """
        Structured search exploiting the 9 = 3*3 factorization.
        Uses block structure and algebraic constraints.
        """
        if time.time() > deadline - 20:
            return None
        
        # Strategy 1: Block decomposition using 9 = 3+3+3
        # Split k into 3 blocks of 3
        # For each block, we have a 4x4 * 4x3 subproblem
        # Try to share U and W across blocks
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(2000 + rank * 100 + attempt)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Assign columns to different "types"
            # Type 0: full terms (affect all i,j,k)
            # Type 1: block terms (affect specific k-block)
            
            n_full = rank // 3
            n_block = (rank - n_full) // 3
            
            col = 0
            # Full terms
            for c in range(n_full):
                U[:, col] = np.random.choice([-1, 0, 1], size=n*m)
                V[:, col] = np.random.choice([-1, 0, 1], size=m*p)
                W[:, col] = np.random.choice([-1, 0, 1], size=p*n)
                col += 1
            
            # Block terms: each affects one k-block
            for block in range(3):
                k_start = block * 3
                k_end = (block + 1) * 3
                for c in range(n_block):
                    # U: random
                    U[:, col] = np.random.choice([-1, 0, 1], size=n*m)
                    
                    # V: only nonzero for k in [k_start, k_end)
                    V[:, col] = 0
                    for j in range(m):
                        for k in range(k_start, k_end):
                            V[j*p + k, col] = np.random.choice([-1, 0, 1])
                    
                    # W: only nonzero for k in [k_start, k_end)
                    W[:, col] = 0
                    for k in range(k_start, k_end):
                        for i in range(n):
                            W[k*n + i, col] = np.random.choice([-1, 0, 1])
                    
                    col += 1
            
            if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        # Strategy 2: Use known low-rank methods for small matrices
        # For 4x4 * 4x9, try to use Winograd-like transforms
        
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(3000 + rank * 100 + attempt)
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Try sparse random with structured patterns
            for c in range(rank):
                # Choose a random "pattern" for this term
                pattern = np.random.randint(0, 8)
                
                if pattern < 2:
                    # Diagonal-like: U[i,j] nonzero only if i==j, W[k,i] nonzero only if k==i
                    for i in range(min(n, m)):
                        U[i*m + i, c] = np.random.choice([-1, 1])
                        W[i*n + i, c] = np.random.choice([-1, 1])
                    V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                elif pattern < 4:
                    # Block diagonal: group i,j into pairs
                    for i in range(0, n, 2):
                        for j in range(0, m, 2):
                            U[i*m + j, c] = np.random.choice([-1, 1])
                            U[(i+1)*m + j, c] = np.random.choice([-1, 1])
                            U[i*m + (j+1), c] = np.random.choice([-1, 1])
                            U[(i+1)*m + (j+1), c] = np.random.choice([-1, 1])
                    V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                    W[:, c] = np.random.choice([-1, 0, 1], size=p*n)
                else:
                    # Random sparse
                    U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
                    V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                    W[:, c] = np.random.choice([-1, 0, 1], size=p*n)
            
            if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to find lower rank decompositions
    best_result = None
    
    # Try ranks below schoolbook with various strategies
    # Start from a reasonable lower bound and go up
    # For 4x4*4x9, known bounds suggest rank around 100-140
    for rank in range(100, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Try structured search first (more likely to succeed)
        result = try_structured_search(rank, max_attempts=30)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
        
        # Try random decomposition
        result = try_random_decomposition(rank, seed=rank*1000, sparsity=0.3)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (U, V, W, rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
