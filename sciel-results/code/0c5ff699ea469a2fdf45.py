"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _local_search_from_schoolbook(n, m, p, target_rank, deadline, max_iter=2000):
    """
    Start from schoolbook (rank n*m*p) and try to reduce rank by merging columns.
    Use local search: try to replace two columns with one column that preserves the tensor.
    """
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    U = U_sb[:, :target_rank].copy()
    V = V_sb[:, :target_rank].copy()
    W = W_sb[:, :target_rank].copy()
    
    # We need to find a valid decomposition of rank target_rank.
    # Start with schoolbook of rank target_rank (just take first target_rank columns).
    # This won't work directly since schoolbook has rank n*m*p.
    
    # Alternative: Use a structured approach.
    # For 4x5x5, try to use Kronecker product structures.
    
    # Let's try a different approach: start with random sparse matrices and do local search.
    # But that's what the original code does.
    
    # Instead, let's try to use the fact that matrix multiplication tensor has
    # symmetries and try to exploit them.
    
    # For now, return None to indicate this approach is not implemented yet.
    return None

def _structured_search(n, m, p, target_rank, deadline):
    """
    Try structured decompositions using Kronecker products and block matrices.
    """
    # For 4x5x5, we can try to decompose using:
    # - 2x2 blocks with Strassen (but 5 is not divisible by 2)
    # - Use padding to make dimensions powers of 2, then unpadded
    
    # Try a simple structured approach:
    # Use the fact that 4 = 2*2, 5 = 5, 5 = 5
    # Try to use Kronecker product of smaller decompositions
    
    # For now, this is a placeholder. Return None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Attempt to find lower rank (99, 98, etc.)
    # Strategy: Informed random search with local modifications.
    
    for target_rank in [99, 98, 97]:
        if time.time() > deadline - 10:
            break
            
        # Try informed random search
        # Use a mixture of random and structured initialization
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Random sparse initialization with some structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with random sparse patterns
            for c in range(target_rank):
                # Each column has a few non-zero entries
                nnz_u = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                nnz_v = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                nnz_w = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                
                u_idx = np.random.choice(n*m, nnz_u, replace=False)
                v_idx = np.random.choice(m*p, nnz_v, replace=False)
                w_idx = np.random.choice(p*n, nnz_w, replace=False)
                
                U[u_idx, c] = np.random.choice([-1, 1], nnz_u)
                V[v_idx, c] = np.random.choice([-1, 1], nnz_v)
                W[w_idx, c] = np.random.choice([-1, 1], nnz_w)
            
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < n * m * p:
            break
            
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
