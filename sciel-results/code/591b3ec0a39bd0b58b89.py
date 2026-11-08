import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the full tensor T[i,j,k] = delta
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-12:
                    return False
    return True

def _qr_rank_revealing(U, V, W, n, m, p):
    """
    Apply rank-revealing QR to the Gram matrix of the rank-1 terms.
    Each term r has a vector in R^{n*m} (from U column r) and R^{m*p} (from V column r).
    The Gram matrix G[r,s] = <u_r, u_s> * <v_r, v_s> where u_r is column r of U, v_r is column r of V.
    We want to find a minimal subset of terms that spans the tensor.
    
    Actually, the correct approach: the tensor T can be viewed as a matrix of size (n*m) x (m*p) 
    when we flatten indices. But the decomposition is T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r].
    
    Let me think differently. The multiplication tensor T ∈ R^{n*m*p} with T[i,j,k]=1.
    A rank-1 term is u ⊗ v ⊗ w where u ∈ R^{n*m}, v ∈ R^{m*p}, w ∈ R^{p*n}.
    T = sum_r u_r ⊗ v_r ⊗ w_r.
    
    For rank minimization, we can use the fact that if we fix the W factors, we get a bilinear form.
    But a more direct approach: use the Gram matrix approach on the "flattened" tensor.
    
    Actually, let's use a simpler approach: treat the problem as finding the minimum number of 
    rank-1 terms. We start with the known decomposition and try to eliminate terms that are 
    linear combinations of others.
    """
    rank = len(U[0]) if U else 0
    
    # Build Gram matrix: G[r,s] = sum over all (i,j,k) of (U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) * (U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s])
    # This is the inner product of the "flattened" rank-1 tensors.
    # Actually, the inner product of two rank-1 tensors u⊗v⊗w and u'⊗v'⊗w' is <u,u'>*<v,v'>*<w,w'>.
    
    # Compute column inner products
    U_inner = [[0]*rank for _ in range(rank)]  # <u_r, u_s>
    V_inner = [[0]*rank for _ in range(rank)]  # <v_r, v_s>
    W_inner = [[0]*rank for _ in range(rank)]  # <w_r, w_s>
    
    for r in range(rank):
        for s in range(r, rank):
            ui = 0
            vi = 0
            wi = 0
            for idx in range(n*m):
                ui += U[idx][r] * U[idx][s]
            for idx in range(m*p):
                vi += V[idx][r] * V[idx][s]
            for idx in range(p*n):
                wi += W[idx][r] * W[idx][s]
            U_inner[r][s] = ui
            U_inner[s][r] = ui
            V_inner[r][s] = vi
            V_inner[s][r] = vi
            W_inner[r][s] = wi
            W_inner[s][r] = wi
    
    # The Gram matrix of the flattened tensors is G[r,s] = U_inner[r][s] * V_inner[r][s] * W_inner[r][s]
    # But this isn't quite right for determining linear dependence.
    
    # Better approach: The tensor T = sum_r t_r where t_r = u_r ⊗ v_r ⊗ w_r.
    # We want to find the minimum subset of {t_r} that sums to T.
    # This is equivalent to: find coefficients c_r such that sum_r c_r * t_r = T, 
    # and minimize the number of non-zero c_r.
    # But since T = sum_r t_r, we have sum_r (c_r - 1) * t_r = 0.
    # So we need to find linear dependencies among the t_r.
    
    # Flatten each t_r into a vector of length n*m*p
    # t_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Build a matrix M of size (n*m*p) x rank where M[i*m*p + j*m*p + k, r] = t_r[i,j,k]
    # Then find the rank of M and a basis.
    
    dim = n * m * p
    M = [[0.0] * rank for _ in range(dim)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    M[idx][r] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Now we need to find the rank of M and express each column as a combination of basis columns.
    # Use QR with column pivoting.
    import numpy as np
    M_np = np.array(M, dtype=float)
    
    # QR with column pivoting
    try:
        from scipy.linalg import qr
        Q, R, piv = qr(M_np, pivoting=True, mode='economic')
    except ImportError:
        # Fallback: use numpy's QR without pivoting, then determine rank
        Q, R = np.linalg.qr(M_np)
        piv = list(range(rank))
    
    # Determine numerical rank
    tol = 1e-10
    num_rank = 0
    for r in range(rank):
        if abs(R[r, r]) > tol:
            num_rank = r + 1
        else:
            break
    
    # The first num_rank columns (after pivoting) form a basis.
    # The remaining columns are linear combinations.
    # We can try to eliminate the non-basis columns.
    
    # For now, return the original if we can't improve
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank-28 decomposition
    U0 = [[0] * 28 for _ in range(n * m)]
    V0 = [[0] * 28 for _ in range(m * p)]
    W0 = [[0] * 28 for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4)
    for i in range(n):
        for k in range(p):
            U0[i * m + 2][r] = 1
            V0[2 * p + k][r] = 1
            W0[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4)
    for i in range(n):
        for j in range(2):
            U0[i * m + j][r] = 1
            V0[j * p + 4][r] = 1
            W0[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen
    # M1
    U0[0 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 0][r] = 1
    V0[1 * p + 1][r] = 1
    W0[0 * n + 0][r] = 1
    W0[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U0[1 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 0][r] = 1
    W0[0 * n + 1][r] = 1
    W0[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U0[0 * m + 0][r] = 1
    V0[0 * p + 1][r] = 1
    V0[1 * p + 1][r] = -1
    W0[1 * n + 0][r] = 1
    W0[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U0[1 * m + 1][r] = 1
    V0[1 * p + 0][r] = 1
    V0[0 * p + 0][r] = -1
    W0[0 * n + 0][r] = 1
    W0[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U0[0 * m + 0][r] = 1
    U0[0 * m + 1][r] = 1
    V0[1 * p + 1][r] = 1
    W0[0 * n + 0][r] = -1
    W0[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U0[1 * m + 0][r] = 1
    U0[0 * m + 0][r] = -1
    V0[0 * p + 0][r] = 1
    V0[0 * p + 1][r] = 1
    W0[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U0[0 * m + 1][r] = 1
    U0[1 * m + 1][r] = -1
    V0[1 * p + 0][r] = 1
    V0[1 * p + 1][r] = 1
    W0[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen
    # M1
    U0[0 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 2][r] = 1
    V0[1 * p + 3][r] = 1
    W0[2 * n + 0][r] = 1
    W0[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U0[1 * m + 0][r] = 1
    U0[1 * m + 1][r] = 1
    V0[0 * p + 2][r] = 1
    W0[2 * n + 1][r] = 1
    W0[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U0[0 * m + 0][r] = 1
    V0[0 * p + 3][r] = 1
    V0[1 * p + 3][r] = -1
    W0[3 * n + 0][r] = 1
    W0[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U0[1 * m + 1][r] = 1
    V0[1 * p + 2][r] = 1
    V0[0 * p + 2][r] = -1
    W0[2 * n + 0][r] = 1
    W0[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U0[0 * m + 0][r] = 1
    U0[0 * m + 1][r] = 1
    V0[1 * p + 3][r] = 1
    W0[2 * n + 0][r] = -1
    W0[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U0[1 * m + 0][r] = 1
    U0[0 * m + 0][r] = -1
    V0[0 * p + 2][r] = 1
    V0[0 * p + 3][r] = 1
    W0[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U0[0 * m + 1][r] = 1
    U0[1 * m + 1][r] = -1
    V0[1 * p + 2][r] = 1
    V0[1 * p + 3][r] = 1
    W0[2 * n + 0][r] = 1
    r += 1
    
    # Verify the initial decomposition
    assert _verify_decomposition(U0, V0, W0, n, m, p), "Initial decomposition is invalid"
    
    # Try to reduce rank using QR-based rank revelation
    U, V, W, rank = _qr_rank_revealing(U0, V0, W0, n, m, p)
    
    # Verify the result
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to original
        return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
