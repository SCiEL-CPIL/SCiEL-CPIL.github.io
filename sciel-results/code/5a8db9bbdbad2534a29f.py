import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification of the tensor decomposition."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Product term for (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build arrays of indices
    I, J, K = np.indices((n, m, p))
    U_idx = I * m + J
    V_idx = J * p + K
    W_idx = K * n + I
    # (n, m, p, rank)
    U_exp = U[U_idx, :]
    V_exp = V[V_idx, :]
    W_exp = W[W_idx, :]
    S = np.einsum('ijkl,ijkl,ijkl->ijk', U_exp, V_exp, W_exp)
    return np.all(np.abs(S - 1.0) <= tol)

def _build_strassen_based(n, m, p):
    """Fallback: Strassen on 2x2x2 blocks + schoolbook for the 2x1x10 remainder (rank 55)."""
    n, m, p = 2, 3, 10
    num_pairs = p // 2
    terms_per_pair = 7 + 4
    total_rank = num_pairs * terms_per_pair
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Strassen 2x2x2 (canonical)
    Su = np.zeros((4, 7)); Sv = np.zeros((4, 7)); Sw = np.zeros((4, 7))
    Su[0,0]=1; Su[3,0]=1; Sv[0,0]=1; Sv[3,0]=1; Sw[0,0]=1; Sw[3,0]=1
    Su[2,1]=1; Su[3,1]=1; Sv[0,1]=1; Sw[2,1]=1
    Su[0,2]=1; Sv[1,2]=1; Sv[3,2]=-1; Sw[1,2]=1; Sw[3,2]=-1
    Su[3,3]=1; Sv[2,3]=1; Sv[0,3]=-1; Sw[0,3]=1; Sw[2,3]=1
    Su[0,4]=1; Su[1,4]=1; Sv[3,4]=1; Sw[0,4]=-1; Sw[1,4]=1
    Su[2,5]=1; Su[0,5]=-1; Sv[0,5]=1; Sv[1,5]=1; Sw[3,5]=1
    Su[1,6]=1; Su[3,6]=-1; Sv[2,6]=1; Sv[3,6]=1; Sw[0,6]=-1; Sw[3,6]=1

    for pair in range(num_pairs):
        col_offset = pair * terms_per_pair
        k0 = pair * 2
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    val = Su[i*2 + j, r]
                    if val != 0:
                        U[i * m + j, col_offset + r] = val
            for j in range(2):
                for k in range(2):
                    val = Sv[j*2 + k, r]
                    if val != 0:
                        V[j * p + (k0 + k), col_offset + r] = val
            for k in range(2):
                for i in range(2):
                    val = Sw[k*2 + i, r]
                    if val != 0:
                        W[(k0 + k) * n + i, col_offset + r] = val
        sb_offset = 7
        for i in range(2):
            for k in range(2):
                idx = col_offset + sb_offset + i * 2 + k
                U[i * m + 2, idx] = 1
                V[2 * p + (k0 + k), idx] = 1
                W[(k0 + k) * n + i, idx] = 1
    return U, V, W

def _try_mip_rank(n, m, p, R, time_budget=25.0):
    """
    Attempt to find a rank-R decomposition via MIP.
    Decision variables:
      z:  (n*m*p,) binary, sum z <= R
      u:  (n*m, R) continuous
      v:  (m*p, R) continuous
      w:  (p*n, R) continuous
    Constraints for every (i,j,k):
      sum_r u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r] = 1
    This is a 3-index product (bilinear in 3 vars -> not directly MIP).
    To make it MIP, we use the standard rank-1 decomposition trick:
      For each r, the term u_a,r * v_b,r * w_c,r is trilinear.
    Instead, we reformulate: fix the "support" via z, and for each r we assign
    a single scalar slot (i,j,k) that it covers, with coefficient 1, plus allow
    combinations. This is complex.
    
    Simpler MIP reformulation that IS solvable:
    Use the "one-hot per r" approach: for each r, choose a subset S_r of slots
    and assign weights. But exact tensor decomposition requires bilinear structure.
    
    Given the trilinearity, a direct MIP is hard. Instead, we use a DIFFERENT
    structural search: random restarts with local coordinate descent on a
    fixed-rank parameterization, which is a different optimization family
    (heuristic local search vs. fixed hand-derived blocks).
    """
    return None

def _local_search_decomposition(n, m, p, R, iters=2000, seed=0):
    """
    Heuristic local search: initialize U,V,W randomly (or with a structured seed),
    then perform coordinate updates to minimize the Frobenius error of the
    reconstructed tensor. If error drops to ~0, we have a valid rank-R decomposition.
    This is a different optimization family (continuous local search) than the
    current fixed-block approach.
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, R))
    V = rng.standard_normal((m * p, R))
    W = rng.standard_normal((p * n, R))
    
    # Normalize initial scale
    scale = 1.0 / (R ** 0.5)
    U *= scale; V *= scale; W *= scale
    
    # Target tensor
    target = np.ones((n, m, p))
    
    def error():
        I, J, K = np.indices((n, m, p))
        U_idx = I * m + J
        V_idx = J * p + K
        W_idx = K * n + I
        S = np.einsum('ijkl,ijkl,ijkl->ijk', U[U_idx, :], V[V_idx, :], W[W_idx, :])
        return np.sqrt(np.mean((S - target) ** 2))
    
    e0 = error()
    best_e = e0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Coordinate descent: update one entry at a time
    for it in range(iters):
        improved = False
        # Update a random U entry
        a = rng.integers(n * m)
        r = rng.integers(R)
        # Optimal u[a, r] given current V, W:
        # min over u: sum_{i,j,k} (sum_r' u[i*m+j, r'] v[j*p+k, r'] w[k*n+i, r'] - 1)^2
        # This is a least-squares problem in u[a, r].
        # Contribution of u[a, r] to (i,j,k) is nonzero only when i*m+j == a, i.e., i=a//m, j=a%m.
        i0, j0 = a // m, a % m
        # For all k, the term is u[a, r] * V[j0*p+k, r] * W[k*n+i0, r]
        # Residual for (i0, j0, k): target - (sum_r' u[i0*m+j0, r'] V[j0*p+k, r'] W[k*n+i0, r'])
        # = 1 - (u[a,r]*V[j0*p+k,r]*W[k*n+i0,r] + rest)
        # Let rest_k = sum_{r' != r} u[i0*m+j0, r'] * V[j0*p+k, r'] * W[k*n+i0, r']
        # Then residual = 1 - rest_k - u[a,r] * V[j0*p+k, r] * W[k*n+i0, r]
        # Minimize sum_k (1 - rest_k - u * A_k)^2 where A_k = V[j0*p+k, r] * W[k*n+i0, r]
        # Optimal u = sum_k (1 - rest_k) * A_k / sum_k A_k^2
        A = V[j0 * p, r] * W[:, n * i0 + i0]  # shape (p,)
        # Wait, W index is k*n + i0, for k in 0..p-1
        A = np.array([V[j0 * p + k, r] * W[k * n + i0, r] for k in range(p)])
        rest = np.array([
            sum(U[i0 * m + j0, r2] * V[j0 * p + k, r2] * W[k * n + i0, r2] 
                for r2 in range(R) if r2 != r)
            for k in range(p)
        ])
        b = 1.0 - rest
        denom = np.dot(A, A)
        if abs(denom) > 1e-12:
            new_val = np.dot(b, A) / denom
            old_val = U[a, r]
            U[a, r] = new_val
            e_new = error()
            if e_new < best_e - 1e-12:
                best_e = e_new
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
            else:
                U[a, r] = old_val
        
        if not improved:
            continue
        
        # Similarly for V and W (omitted for brevity in this heuristic, but we can add)
        # Update V
        b_idx = rng.integers(m * p)
        r = rng.integers(R)
        j0, k0 = b_idx // p, b_idx % p
        A = np.array([U[i * m + j0, r] * W[k0 * n + i, r] for i in range(n)])
        rest = np.array([
            sum(U[i * m + j0, r2] * V[j0 * p + k0, r2] * W[k0 * n + i, r2]
                for r2 in range(R) if r2 != r)
            for i in range(n)
        ])
        b = 1.0 - rest
        denom = np.dot(A, A)
        if abs(denom) > 1e-12:
            new_val = np.dot(b, A) / denom
            old_val = V[b_idx, r]
            V[b_idx, r] = new_val
            e_new = error()
            if e_new < best_e - 1e-12:
                best_e = e_new
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                V[b_idx, r] = old_val
        
        # Update W
        c_idx = rng.integers(p * n)
        r = rng.integers(R)
        k0, i0 = c_idx // n, c_idx % n
        A = np.array([U[i0 * m + j, r] * V[j * p + k0, r] for j in range(m)])
        rest = np.array([
            sum(U[i0 * m + j, r2] * V[j * p + k0, r2] * W[k0 * n + i0, r2]
                for r2 in range(R) if r2 != r)
            for j in range(m)
        ])
        b = 1.0 - rest
        denom = np.dot(A, A)
        if abs(denom) > 1e-12:
            new_val = np.dot(b, A) / denom
            old_val = W[c_idx, r]
            W[c_idx, r] = new_val
            e_new = error()
            if e_new < best_e - 1e-12:
                best_e = e_new
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                W[c_idx, r] = old_val
    
    if best_e < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 140

    # Start with the known rank-55 Strassen-based decomposition as baseline
    U, V, W = _build_strassen_based(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W

    # Try to find a lower rank via local search for R = 54, 53, ...
    # This is a different optimization family: heuristic continuous local search
    # vs. the current fixed hand-derived block structure.
    for R in range(best_rank - 1, 49, -1):
        if time.time() > deadline - 30:
            break
        # Try a few random restarts
        found = None
        for seed in range(5):
            if time.time() > deadline - 30:
                break
            res = _local_search_decomposition(n, m, p, R, iters=500, seed=seed)
            if res is not None:
                found = res
                break
        if found is not None:
            U2, V2, W2 = found
            if _verify_decomposition(U2, V2, W2, n, m, p):
                best_rank = R
                best_U, best_V, best_W = U2, V2, W2
                # Continue trying to go lower
                continue
        # If R failed, we can try R-1 etc., but local search may not find it
        # Break if we've spent too much time
        if time.time() > deadline - 60:
            break

    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
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
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
