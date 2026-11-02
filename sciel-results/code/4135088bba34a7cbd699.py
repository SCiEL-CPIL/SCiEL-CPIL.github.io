"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We need T[i, j, k] = 1 for all i,j,k (identity for MM).
    """
    rank = U.shape[1]
    # Build the full tensor via einsum
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum
    # First: U_r[i,j,r] * V_r[j,k,r] -> (n, m, p, rank)
    UV = np.einsum('ijr,jkr->ijk r'.replace(' ', ''), U_r, V_r)
    # Then multiply by W_r[k,i,r] and sum over r
    # UV[i,j,k,r] * W_r[k,i,r] -> sum_r
    T = np.einsum('ijk r,ki r->ijk'.replace(' ', ''), UV, W_r)
    
    return np.allclose(T, 1.0, atol=1e-10)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_rank(r, n, m, p, max_time=100.0):
    """Try to find a rank-r decomposition using random initialization and gradient-like refinement.
    Returns (U, V, W) as numpy arrays or None if not found within max_time.
    """
    start_time = time.time()
    
    # For small ranks, we can use a direct algebraic approach.
    # The tensor T[i,j,k] = delta (always 1) needs rank-r decomposition.
    # This is equivalent to finding r vectors u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # such that sum_c u_c ⊗ v_c ⊗ w_c = T.
    
    # Strategy: use the fact that for matrix multiplication, there are known constructions.
    # For 3x5x7, the trivial bound is 105. Let's try to reduce.
    
    # One approach: if one dimension is small, we can use block structure.
    # Let's try a structured approach based on the minimum dimension.
    
    # Actually, for this specific problem, let's use a numerical optimization approach
    # with random restarts for small ranks.
    
    if r >= n * m * p:
        return _schoolbook(n, m, p)
    
    # For ranks close to the schoolbook, we can try to merge terms.
    # But for now, let's use a simple random search for very small ranks.
    
    # Actually, a better approach: use the fact that the MM tensor has structure.
    # Let's try to find a decomposition using the minimum of the dimensions.
    
    # For n=3, m=5, p=7:
    # We can write the multiplication as summing over 5 intermediate values.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is already rank 3*5*7 = 105 in the tensor sense.
    
    # To reduce rank, we need to find linear dependencies or use special structure.
    # One known technique: if we can factor one of the matrices, we can reduce rank.
    
    # Let's try a simple approach: use numpy's SVD-like decomposition on the tensor.
    # Reshape T to (n*m, p) for each... no, that's not quite right.
    
    # The tensor T is of shape (n, m, p) with all entries 1.
    # Wait, no! The MM tensor T[i,j,k] is 1 for all i,j,k.
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a constant tensor! We can decompose a constant tensor with much lower rank.
    # T[i,j,k] = 1 = (1/sqrt(n)) * (1/sqrt(m)) * (1/sqrt(p)) * sqrt(n) * sqrt(m) * sqrt(p)
    # Actually, T = u ⊗ v ⊗ w where u[i,j] = 1, v[j,k] = 1, w[k,i] = 1? No, that gives 1.
    
    # Wait: if U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i
    # Then sum_c ... = 1*1*1 = 1. So rank 1 works!
    
    # Let me verify: U has shape (15, 1), all entries 1.
    # V has shape (35, 1), all entries 1.
    # W has shape (21, 1), all entries 1.
    # T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 * 1 * 1 = 1. ✓
    
    # So the rank is 1!
    # But wait, is that correct? Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 7 matrix."
    # The matrix multiplication tensor T[i,j,k] represents the coefficient of A[i,j] * B[j,k] in C[i,k].
    # For standard MM, C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all i,j,k.
    
    # Yes! The MM tensor for multiplying an n×m by m×p matrix has T[i,j,k] = 1 for all valid i,j,k.
    # This is a rank-1 tensor!
    
    # So the minimal rank is 1.
    
    if r >= 1:
        U = np.ones((n * m, 1), dtype=float)
        V = np.ones((m * p, 1), dtype=float)
        W = np.ones((p * n, 1), dtype=float)
        return U, V, W
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # The MM tensor T[i,j,k] = 1 for all i,j,k is rank 1.
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # gives T[i,j,k] = 1 * 1 * 1 = 1. ✓
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
