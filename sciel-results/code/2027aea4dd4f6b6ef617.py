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
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using NumPy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        total = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            Uc = U[:, c].reshape(n, m)  # (n, m)
            Vc = V[:, c].reshape(m, p)  # (m, p)
            Wc = W[:, c].reshape(p, n)  # (p, n)
            for i in range(n):
                contrib = np.outer(Uc[i, :], Wc[:, i])  # (m, p)
                total[i, :, :] += contrib * Vc
        
        if np.max(np.abs(total - 1.0)) > tol:
            return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition (rank n*m*p)."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def try_sparse_random(rank, seed, sparsity=0.25):
        """Try a sparse random decomposition."""
        np.random.seed(seed)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for c in range(rank):
            mask_U = np.random.random((n*m,)) < sparsity
            mask_V = np.random.random((m*p,)) < sparsity
            mask_W = np.random.random((p*n,)) < sparsity
            U[mask_U, c] = np.random.choice([-1, 1], size=mask_U.sum())
            V[mask_V, c] = np.random.choice([-1, 1], size=mask_V.sum())
            W[mask_W, c] = np.random.choice([-1, 1], size=mask_W.sum())
        
        if verify_fast(U, V, W):
            return U, V, W
        return None
    
    def try_structured(rank, seed):
        """
        Try a structured decomposition exploiting the 9=3*3 structure.
        Group the 9 output columns into 3 groups of 3.
        """
        np.random.seed(seed)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Divide rank into 3 parts for the 3 column groups
        ranks_per_group = [rank // 3] * 3
        ranks_per_group[-1] += rank - sum(ranks_per_group[:-1])
        
        offset = 0
        for g in range(3):
            rg = ranks_per_group[g]
            for c in range(rg):
                col = offset + c
                # U: sparse random
                mask_U = np.random.random((n*m,)) < 0.3
                U[mask_U, col] = np.random.choice([-1, 1], size=mask_U.sum())
                
                # V: only nonzero for columns in group g (cols 3g..3g+2)
                V[:, col] = 0
                for j in range(m):
                    for k in range(3*g, 3*g+3):
                        if np.random.random() < 0.3:
                            V[j*p + k, col] = np.random.choice([-1, 1])
                
                # W: only nonzero for columns in group g
                W[:, col] = 0
                for k in range(3*g, 3*g+3):
                    for i in range(n):
                        if np.random.random() < 0.3:
                            W[k*n + i, col] = np.random.choice([-1, 1])
            
            offset += rg
        
        if verify_fast(U, V, W):
            return U, V, W
        return None
    
    # Search for lower rank
    best_result = None
    
    # Try ranks from 130 to 143
    for rank in range(130, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        
        # Try structured search
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            result = try_structured(rank, seed=rank*1000 + attempt)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, rank)
                break
        
        if best_result is not None:
            break
        
        # Try random search
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            result = try_sparse_random(rank, seed=rank*10000 + attempt, sparsity=0.25)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, rank)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

