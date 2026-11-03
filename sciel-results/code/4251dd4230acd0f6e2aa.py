"""Program for matrix multiplication 3,3,6 tensor decomposition.
Goal: find a rank-minimal exact decomposition (lower rank = better score).
We start from the schoolbook decomposition (rank 54) and attempt to reduce
rank via structured sparsification and verification."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE_S = 110.0  # safety margin within 120s limit


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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
    
    return U, V, W


def _verify(U, V, W, n, m, p, seed=42):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Efficient verification using matrix operations
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is equivalent to: for each r, outer product of reshaped vectors
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r]  # (n*m,)
        v_vec = V[:, r]  # (m*p,)
        w_vec = W[:, r]  # (p*n,)
        
        # Reshape to matrices
        u_mat = u_vec.reshape(n, m)  # (n, m)
        v_mat = v_vec.reshape(m, p)  # (m, p)
        w_mat = w_vec.reshape(p, n)  # (p, n)
        
        # Contribution: sum over j of u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # = (u_mat @ v_mat) * w_mat^T element-wise
        uv = u_mat @ v_mat  # (n, p)
        C_computed += uv * w_mat.T  # (n, p)
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True


def _try_strassen_3x3x3():
    """
    Try to construct a rank-23 decomposition for 3x3x3 matmul.
    Returns (U, V, W) if successful, None otherwise.
    """
    n = m = p = 3
    
    # Known rank-23 decomposition for 3x3x3 matmul
    # This uses the algorithm from Landsberg et al.
    # The decomposition has 23 terms with specific integer/half-integer coefficients.
    
    # For 3x3x3, we can use a combination of Strassen-like ideas.
    # Let me construct a specific known decomposition.
    
    # Actually, let me try a different approach: use the fact that we can
    # embed smaller decompositions.
    
    # For now, let's try to find a valid lower-rank decomposition through
    # structured search.
    
    # One approach: use the 2x2x2 Strassen decomposition (rank 7) and extend it.
    # But this is complex for 3x3x3.
    
    # Let me try a random search for a lower-rank decomposition.
    return None


def _random_search_for_lower_rank(n, m, p, target_rank, max_iterations=10000, timeout_s=30):
    """
    Randomly search for a rank-`target_rank` decomposition.
    This is unlikely to succeed for large rank reductions but may find
    small improvements.
    """
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > timeout_s:
            break
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First, build the schoolbook decomposition as a guaranteed valid baseline
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p  # 54
    
    # Try to find a lower-rank decomposition
    # Strategy 1: Try rank 53, 52, ... down to some minimum
    # This is a search for small improvements
    
    # For 3x3x6, the theoretical lower bound is at least 18 (number of outputs)
    # but realistically much higher.
    
    # Let's try a few ranks below 54
    for target_rank in range(best_rank - 1, max(17, best_rank - 20), -1):
        if time.time() - start_time > DEADLINE_S - 20:
            break
        
        # Try random search for this rank
        result = _random_search_for_lower_rank(n, m, p, target_rank, 
                                                max_iterations=5000, 
                                                timeout_s=10)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            break  # Found a valid lower rank, stop searching
    
    # Verify the best result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return result
