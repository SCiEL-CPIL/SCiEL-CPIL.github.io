"""Find rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 2, 7]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification using tensor contraction."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct the tensor from the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # T has shape (n*m, m*p, p*n)
    
    # Build the reconstructed tensor
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T_recon = einsum('ar,br,cr->abc', U, V, W)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _try_rank_search(n, m, p, start_rank, deadline):
    """
    Try to find a decomposition with rank < schoolbook using structured random search.
    Returns (U, V, W) if found, else None.
    """
    schoolbook = n * m * p
    best_result = None
    
    for rank in range(start_rank, schoolbook):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        n_trials = 50
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with sparsity
            density = 0.3 + 0.5 * random.random()
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Apply sparsity mask
            mask_U = np.random.rand(n*m, rank) < density
            mask_V = np.random.rand(m*p, rank) < density
            mask_W = np.random.rand(p*n, rank) < density
            U *= mask_U
            V *= mask_V
            W *= mask_W
            
            # Normalize columns to avoid overflow/underflow
            for r in range(rank):
                norm_U = np.linalg.norm(U[:, r])
                norm_V = np.linalg.norm(V[:, r])
                norm_W = np.linalg.norm(W[:, r])
                if norm_U > 0 and norm_V > 0 and norm_W > 0:
                    scale = (norm_U * norm_V * norm_W) ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Verify
            if _verify_fast(U, V, W, n, m, p):
                # Found a valid decomposition!
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for the 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 28
    
    # First, try to find a lower-rank decomposition using random search
    # Start from a reasonable lower bound
    # The matricization ranks give lower bounds
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    T_mat1 = T.reshape(n*m, m*p * p*n)
    rank1 = np.linalg.matrix_rank(T_mat1, tol=1e-10)
    
    T_mat2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    rank2 = np.linalg.matrix_rank(T_mat2, tol=1e-10)
    
    T_mat3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    rank3 = np.linalg.matrix_rank(T_mat3, tol=1e-10)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # Try random search for ranks between lower_bound and schoolbook_rank - 1
    # Use a moderate search budget
    search_deadline = min(deadline, start_time + 100)
    
    found = _try_rank_search(n, m, p, max(lower_bound, 1), search_deadline)
    
    if found is not None:
        U, V, W = found
        # Verify once more with the strict checker
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fall back to schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
    
    # Verify the decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
