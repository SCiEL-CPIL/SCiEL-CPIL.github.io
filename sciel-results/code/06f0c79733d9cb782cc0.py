import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """7 terms for 2x2x2 Strassen."""
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def _build_kronecker_strassen_444():
    """Kronecker-embedded Strassen for 4x4x4 (rank 49)."""
    n, m, p = 4, 4, 4
    U_cols, V_cols, W_cols = [], [], []

    def add(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    st = _strassen_2x2x2_terms()
    for (u1s, v1s, w1s) in st:
        for (u2s, v2s, w2s) in st:
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            for (a, va) in u1s:
                i1, j1 = a // 2, a % 2
                for (b, vb) in u2s:
                    i2, j2 = b // 2, b % 2
                    u[ui(i1 * 2 + i2, j1 * 2 + j2)] = va * vb
            for (a, va) in v1s:
                j1, k1 = a // 2, a % 2
                for (b, vb) in v2s:
                    j2, k2 = b // 2, b % 2
                    v[vi(j1 * 2 + j2, k1 * 2 + k2)] = va * vb
            for (a, va) in w1s:
                k1, i1 = a // 2, a % 2
                for (b, vb) in w2s:
                    k2, i2 = b // 2, b % 2
                    w[wi(k1 * 2 + k2, i1 * 2 + i2)] = va * vb
            add(u, v, w)

    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T

def _build_optimized_baseline(n=4, m=4, p=5):
    """
    Builds the rank-50 baseline:
    49 terms for 4x4x4 (Strassen) + 1 term for 4x4x1 (rank 1).
    """
    U444, V444, W444 = _build_kronecker_strassen_444()
    
    U = np.zeros((n * m, 50))
    V = np.zeros((m * p, 50))
    W = np.zeros((p * n, 50))
    
    U[:, :49] = U444
    V[:, :49] = V444
    W[:, :49] = W444
    
    # Add the 1 term for k=4 (last column of B)
    # We need to compute C[:, 4] = A * B[:, 4]
    # This is a matrix-vector multiplication (4x4) * (4x1) -> (4x1)
    # Wait, the tensor is for 4x4 * 4x5 -> 4x5.
    # The indices are:
    # U[i, j] for A[i, j]
    # V[j, k] for B[j, k]
    # W[k, i] for C[k, i] ... wait, C is 4x5. C[row, col].
    # The convention says W[k*n+i, r].
    # Let's check the dimensions.
    # A is 4x4 (n=4, m=4).
    # B is 4x5 (m=4, p=5).
    # C is 4x5 (n=4, p=5).
    # C[i, k] = sum_j A[i, j] * B[j, k].
    # So W should map (k, i) to C[k, i]? No, C is indexed by row i, col k.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # So W is a vector of length p*n = 5*4 = 20.
    # Index in W is k*n + i.
    # This corresponds to C[i, k] if we flatten C row-major?
    # C[i, k] is at index i*p + k in row-major.
    # But W[k*n+i] suggests C is stored column-major? Or the indices are swapped?
    # Let's look at the verification in the seed program.
    # s = np.dot(row_u * V[j * p + k], W[k * n + i])
    # row_u is U[i*m+j].
    # V[j*p+k] is B[j,k].
    # W[k*n+i] is W index.
    # The sum is over r.
    # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This s should be 1 if (i,j,k) is the "target" for that term?
    # No, the tensor T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # We want T[i, j, k] = 1 if ...?
    # The matrix multiplication tensor T_{ijk} is 1 if the indices correspond to a valid multiplication path?
    # Actually, the tensor for matrix multiplication A(m) * B(n) -> C(p) is usually defined such that
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T_{ijk} has T_{ijk} = 1.
    # Wait, the standard tensor decomposition of matrix multiplication computes the *algorithm*, not the tensor itself.
    # The tensor T is the structure tensor. T_{ijk} = 1.
    # So we want sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k?
    # No, that's not right.
    # The tensor T has entries T_{ijk} = 1.
    # We want to decompose T = sum_r u_r v_r w_r.
    # So T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # Since T_{ijk} = 1 for all i,j,k, we want sum_r ... = 1.
    # Yes, the verification checks if s == 1.0.
    
    # So for the 4x4x4 part (k=0..3), we have the 49 terms.
    # For k=4, we need to handle the 5th column of B.
    # The tensor entries T_{i,j,4} must be 1.
    # Currently, for the 49 terms, W[k*n+i] is only non-zero for k=0..3 (since it's built for 4x4x4).
    # So for k=4, the contribution from the first 49 terms to T_{i,j,4} is 0.
    # We need to add terms such that for k=4, sum_r U_{ij,r} V_{j4,r} W_{4n+i,r} = 1.
    # And for k=0..3, the new terms must not disturb the existing 1s.
    # So for the new terms, we can set V_{jk,r} = 0 for k=0..3, and W_{kn+i,r} = 0 for k=0..3.
    # Then we just need to satisfy the condition for k=4.
    # This reduces to decomposing the "slice" of the tensor at k=4.
    # This slice is effectively the matrix multiplication 4x4 * 4x1 -> 4x1.
    # Or rather, it's a linear map.
    # T_{i,j,4} = 1.
    # We need sum_r' U'_{ij,r'} V'_{j4,r'} W'_{4n+i,r'} = 1.
    # This is a rank-1 tensor?
    # T_{i,j,4} is just 1 for all i,j.
    # Wait, T_{ijk} = 1 for all i,j,k.
    # So T_{i,j,4} = 1.
    # We can use a single term for this?
    # Let U'_{ij} = 1, V'_{j4} = 1, W'_{4n+i} = 1.
    # Then sum = 1*1*1 = 1.
    # Yes!
    # So we just add 1 term.
    # U[:, 49] = 1
    # V[4, 49] = 1  (j*p + k = j*4 + 4 = 4j+4. j=0..3. So indices 4, 8, 12, 16)
    # Wait, V is (m*p) x rank. m=4, p=5. Size 20.
    # V[j*p+k, r].
    # For k=4, indices are j*5+4.
    # j=0 -> 4
    # j=1 -> 9
    # j=2 -> 14
    # j=3 -> 19
    # W is (p*n) x rank. p=5, n=4. Size 20.
    # W[k*n+i, r].
    # For k=4, indices are 4*4+i = 16+i.
    # i=0 -> 16
    # i=1 -> 17
    # i=2 -> 18
    # i=3 -> 19
    
    U[0:16, 49] = 1.0
    V[4, 49] = 1.0
    V[9, 49] = 1.0
    V[14, 49] = 1.0
    V[19, 49] = 1.0
    W[16, 49] = 1.0
    W[17, 49] = 1.0
    W[18, 49] = 1.0
    W[19, 49] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-7):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            row_u = U[i * m + j]
            for k in range(p):
                s = np.dot(row_u * V[j * p + k], W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_int(M, tol=1e-8):
    M = np.asarray(M, dtype=np.float64)
    return bool(np.all(np.abs(M * 2 - np.round(M * 2)) <= tol))

def _round_half(M):
    return np.round(np.asarray(M, dtype=np.float64) * 2) / 2.0

def _build_schoolbook(n=4, m=4, p=5):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_list(M):
    return [[float(x) for x in row] for row in np.asarray(M)]

def _null_space_prune(U, V, W, n, m, p, max_iter=1000, tol=1e-8):
    """
    Uses null space of the tensor flattening to find linear dependencies
    and prune terms.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    # Flatten tensor vectors
    vecs = []
    for r in range(rank):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0: continue
                for k in range(p):
                    v_val = V[j*p+k, r]
                    if v_val == 0: continue
                    w_val = W[k*n+i, r]
                    if w_val == 0: continue
                    vec[i*m*p + j*p + k] = u_val * v_val * w_val
        vecs.append(vec)
        
    A = np.array(vecs).T # Shape (n*m*p, rank)
    
    N = null_space(A)
    
    if N.shape[1] == 0:
        return U, V, W
        
    alpha = np.ones(rank)
    
    best_alpha = alpha.copy()
    best_active_count = rank
    
    for restart in range(5):
        alpha = np.ones(rank)
        
        improved = True
        while improved:
            improved = False
            best_k = -1
            best_new_alpha = None
            best_active = 0
            
            for k in range(rank):
                if alpha[k] == 0:
                    continue
                n_k = N[:, k]
                norm_sq = np.dot(n_k, n_k)
                if norm_sq < 1e-12:
                    continue
                z = -n_k / norm_sq
                new_alpha = 1 + N @ z
                
                valid = True
                active = 0
                for r in range(rank):
                    if abs(new_alpha[r]) > 1e-8:
                        if not _is_half_int(np.array([[new_alpha[r]]])):
                            valid = False
                            break
                        active += 1
                    else:
                        if abs(new_alpha[r]) > 1e-4:
                            valid = False
                            break
                        active += 0
                
                if valid and active > best_active:
                    best_active = active
                    best_k = k
                    best_new_alpha = new_alpha.copy()
            
            if best_k >= 0 and best_new_alpha is not None:
                alpha = best_new_alpha
                alpha[np.abs(alpha) < 1e-6] = 0
                improved = True
        
        active_count = np.sum(np.abs(alpha) > 1e-8)
        if active_count < best_active_count:
            best_active_count = active_count
            best_alpha = alpha.copy()
    
    alpha = best_alpha
    
    final_U = U.copy()
    final_V = V.copy()
    final_W = W.copy()
    
    active = []
    for r in range(rank):
        if abs(alpha[r]) > 1e-8:
            a = alpha[r]
            success = False
            
            final_U[:, r] = U[:, r] * a
            if _is_half_int(final_U[:, r]):
                active.append(r)
                success = True
            else:
                final_U[:, r] = U[:, r]
            
            if not success:
                final_V[:, r] = V[:, r] * a
                if _is_half_int(final_V[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_V[:, r] = V[:, r]
            
            if not success:
                final_W[:, r] = W[:, r] * a
                if _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_W[:, r] = W[:, r]
            
            if not success and a > 0:
                s = np.sqrt(a)
                final_U[:, r] = U[:, r] * s
                final_V[:, r] = V[:, r] * s
                if _is_half_int(final_U[:, r]) and _is_half_int(final_V[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_U[:, r] = U[:, r]
                    final_V[:, r] = V[:, r]
            
            if not success and a > 0:
                s = np.sqrt(a)
                final_U[:, r] = U[:, r] * s
                final_W[:, r] = W[:, r] * s
                if _is_half_int(final_U[:, r]) and _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_U[:, r] = U[:, r]
                    final_W[:, r] = W[:, r]
            
            if not success and a > 0:
                s = np.sqrt(a)
                final_V[:, r] = V[:, r] * s
                final_W[:, r] = W[:, r] * s
                if _is_half_int(final_V[:, r]) and _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_V[:, r] = V[:, r]
                    final_W[:, r] = W[:, r]
            
            if not success:
                pass
        else:
            pass
    
    if len(active) >= rank:
        return U, V, W
        
    new_U = final_U[:, active]
    new_V = final_V[:, active]
    new_W = final_W[:, active]
    
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
        
    return U, V, W

def _als_refine(U, V, W, n, m, p, target_rank, max_iter=200, tol=1e-9):
    """
    Attempts to reduce the rank to target_rank using Alternating Least Squares.
    """
    if target_rank >= U.shape[1]:
        return U, V, W
    
    # This is a placeholder for a more sophisticated search.
    # For now, return input unchanged.
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    n, m, p = DIMENSIONS

    best = None

    def consider(U, V, W, label=""):
        nonlocal best
        if time.time() > deadline:
            return
        U = _round_half(U); V = _round_half(V); W = _round_half(W)
        if not _verify(U, V, W, n, m, p):
            return
        if not (_is_half_int(U) and _is_half_int(V) and _is_half_int(W)):
            return
        r = U.shape[1]
        if best is None or r < best[0]:
            best = (r, U, V, W)

    # 1) Optimized baseline (Rank 50)
    try:
        U, V, W = _build_optimized_baseline(n, m, p)
        consider(U, V, W, "optimized_baseline")
        
        # Try pruning
        if time.time() < deadline - 20:
            U2, V2, W2 = _null_space_prune(U, V, W, n, m, p)
            consider(U2, V2, W2, "pruned_baseline")
            
            # Try ALS refinement (structural change)
            if time.time() < deadline - 10:
                # Try to reduce to 49, 48, etc.
                for target_rank in range(49, max(10, U2.shape[1] - 10), -1):
                    if time.time() > deadline - 5:
                        break
                    U3, V3, W3 = _als_refine(U2, V2, W2, n, m, p, target_rank)
                    consider(U3, V3, W3, f"als_{target_rank}")
    except Exception:
        pass

    # 2) Fallback schoolbook
    try:
        U, V, W = _build_schoolbook(n, m, p)
        consider(U, V, W, "schoolbook")
    except Exception:
        pass

    if best is None:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    _, U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
