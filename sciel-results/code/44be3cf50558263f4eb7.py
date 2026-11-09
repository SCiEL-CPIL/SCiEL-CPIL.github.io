"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is exact: sum_c U[:,c] * V[:,c] * W[:,c] == MM tensor."""
    r = len(U[0]) if U else 0
    # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 if (i,j,k) is a valid index, else 0
    # Actually for matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = delta_{i,j,k} where the multiplication tensor has 1s at (i,j,k)
    # The MM tensor T_{ijk} = 1 for all valid (i,j,k), else 0. Wait no.
    # The matrix multiplication tensor M has M[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as list of (u, v, w) tuples, each a list."""
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

def _try_merge(U, V, W, c1, c2, n, m, p):
    """Try to merge columns c1 and c2 into a single column by solving for a linear combination.
    Returns (new_U, new_V, new_W) if successful, else None."""
    r = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # We want to find alpha, beta such that for all (i,j,k):
    # alpha * U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] + beta * U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
    # equals the original contribution of both columns
    # Original: U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] + U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2]
    # This is hard to do in general. Instead, try simple merges where one column is a scalar multiple.
    
    # Check if column c2 is a scalar multiple of column c1 in all three matrices
    u1 = U_np[:, c1]
    u2 = U_np[:, c2]
    v1 = V_np[:, c1]
    v2 = V_np[:, c2]
    w1 = W_np[:, c1]
    w2 = W_np[:, c2]
    
    # Try to find alpha such that u2 = alpha * u1 (where u1 is nonzero)
    def find_scalar(a, b):
        """Find scalar s such that b = s*a, or None."""
        mask = np.abs(a) > 1e-12
        if not np.any(mask):
            if np.all(np.abs(b) < 1e-12):
                return 0.0
            return None
        ratios = b[mask] / a[mask]
        if np.all(np.abs(ratios - ratios[0]) < 1e-12):
            return float(ratios[0])
        return None
    
    alpha_u = find_scalar(u1, u2)
    if alpha_u is None:
        return None
    alpha_v = find_scalar(v1, v2)
    if alpha_v is None or abs(alpha_u - alpha_v) > 1e-12:
        return None
    alpha_w = find_scalar(w1, w2)
    if alpha_w is None or abs(alpha_u - alpha_w) > 1e-12:
        return None
    
    # Merge: new column = c1 + alpha * c2, remove c2
    # Actually if c2 = alpha * c1, then sum = c1 + alpha*c1 = (1+alpha)*c1
    # So we can just scale c1 by (1+alpha) and remove c2
    if abs(1.0 + alpha_u) < 1e-12:
        # They cancel out, just remove both
        new_cols = [c for c in range(r) if c != c1 and c != c2]
    else:
        new_cols = [c for c in range(r) if c != c2]
        # Scale c1 by (1+alpha)
        U_np[:, c1] *= (1.0 + alpha_u)
    
    # Reconstruct
    if len(new_cols) == 0:
        return None
    new_U = np.array([U_np[i, new_cols] for i in range(len(U_np))]).tolist()
    new_V = np.array([V_np[i, new_cols] for i in range(len(V_np))]).tolist()
    new_W = np.array([W_np[i, new_cols] for i in range(len(W_np))]).tolist()
    
    return new_U, new_V, new_W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Iteratively try to reduce rank by merging columns
    improved = True
    while improved and time.time() < deadline:
        improved = False
        r = len(U[0])
        for c1 in range(r):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, r):
                if time.time() > deadline:
                    break
                result = _try_merge(U, V, W, c1, c2, n, m, p)
                if result is not None:
                    new_U, new_V, new_W = result
                    if _verify(new_U, new_V, new_W, n, m, p):
                        U, V, W = new_U, new_V, new_W
                        best_U, best_V, best_W = U, V, W
                        best_rank = len(U[0])
                        improved = True
                        break
            if improved:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
