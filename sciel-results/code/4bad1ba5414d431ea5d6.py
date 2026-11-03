import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [4, 4, 5]

def _riemannian_als_refine(U, V, W, n, m, p, max_iter=50, tol=1e-10):
    """
    Riemannian-style ALS refinement on the Stiefel manifold.
    Uses orthogonal Procrustes to project updates back to the manifold,
    exploiting intrinsic curvature to escape local minima.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    if rank == 0:
        return U, V, W
    
    # Precompute index mappings
    nmp = n * m * p
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Target: T[i,j,k] = 1 for all i,j,k
    
    # Build the tensor target vector
    target = np.ones(nmp)
    
    def _residual(U, V, W):
        """Compute ||sum_r u_r ⊗ v_r ⊗ w_r - target||^2"""
        res = np.zeros(nmp)
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # Outer product u ⊗ v ⊗ w reshaped to (nmp,)
            outer = np.einsum('i,j,k->ijk', u_col, v_col, w_col)
            res += outer.reshape(nmp)
        diff = res - target
        return np.dot(diff, diff)
    
    def _procrustes(X):
        """Project X to nearest matrix with orthonormal columns via SVD."""
        if X.shape[1] == 0:
            return X
        try:
            U_svd, _, Vt = np.linalg.svd(X, full_matrices=False)
            return U_svd @ Vt
        except:
            return X
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_res = _residual(U, V, W)
    
    for iteration in range(max_iter):
        # Alternate: update U given V,W; update V given U,W; update W given U,V
        for _ in range(3):
            # Update U: minimize ||sum_r u_r ⊗ (v_r ⊗ w_r) - target||^2
            # This is: ||sum_r u_r * (v_r ⊗ w_r)_flat - target||^2
            # = ||U @ M^T - target_flat||^2 where M[r, idx] = v_r[idx/p] * w_r[idx%p]... 
            # Actually: for fixed r, the contribution to position (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # So for U update: target[i*m*p + j*p + k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
            # Let B[r, i*m+j, j*p+k, k*n+i]... this is complex.
            # Simpler: build the linear system for U.
            # For each (i,j): target_slice[i,j,:] = sum_r U[i*m+j,r] * C_r[i,j,:]
            # where C_r[i,j,k] = V[j*p+k,r] * W[k*n+i,r]
            
            # Build B matrix: B[i*m+j, r, k] = V[j*p+k,r] * W[k*n+i,r]
            # Then for each (i,j): target[i,j,:] = U[i*m+j,:] @ B[i*m+j,:,:].T
            # So U[i*m+j,:] = argmin ||U[i*m+j,:] @ B_ij^T - target_ij||^2
            # Solution: U[i*m+j,:] = target_ij @ B_ij @ (B_ij^T @ B_ij)^{-1}
            
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # Build B_ij: shape (rank, p)
                    B_ij = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B_ij[r, k] = V[j*p+k, r] * W[k*n+i, r]
                    # target_ij: shape (p,)
                    target_ij = np.ones(p)
                    # Solve: U[i*m+j,:] @ B_ij.T = target_ij
                    # U[i*m+j,:] = target_ij @ B_ij @ pinv(B_ij.T @ B_ij)
                    G = B_ij.T @ B_ij  # (p, p)
                    # Use least squares
                    try:
                        # U[i*m+j,:] is (rank,), B_ij is (rank, p)
                        # We want: U_row @ B_ij.T = target_ij
                        # Equivalent: B_ij @ U_row.T = target_ij
                        U_row = np.linalg.lstsq(B_ij, target_ij, rcond=None)[0]
                        U_new[i*m+j, :] = U_row
                    except:
                        pass
            
            # Procrustes projection
            U_new = _procrustes(U_new)
            res = _residual(U_new, V, W)
            if res < best_res - tol:
                U = U_new
                best_res = res
                best_U = U.copy()
            
            # Update V similarly
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    # For V[j*p+k, r]: contribution to (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # target[i,j,k] = sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r])
                    # Build B_jk: shape (rank, n)
                    B_jk = np.zeros((rank, n))
                    for r in range(rank):
                        for i in range(n):
                            B_jk[r, i] = U[i*m+j, r] * W[k*n+i, r]
                    target_jk = np.ones(n)
                    try:
                        V_row = np.linalg.lstsq(B_jk, target_jk, rcond=None)[0]
                        V_new[j*p+k, :] = V_row
                    except:
                        pass
            V_new = _procrustes(V_new)
            res = _residual(U, V_new, W)
            if res < best_res - tol:
                V = V_new
                best_res = res
                best_V = V.copy()
            
            # Update W similarly
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    # For W[k*n+i, r]: contribution to (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # target[i,j,k] = sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r])
                    # Build B_ki: shape (rank, m)
                    B_ki = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            B_ki[r, j] = U[i*m+j, r] * V[j*p+k, r]
                    target_ki = np.ones(m)
                    try:
                        W_row = np.linalg.lstsq(B_ki, target_ki, rcond=None)[0]
                        W_new[k*n+i, :] = W_row
                    except:
                        pass
            W_new = _procrustes(W_new)
            res = _residual(U, V, W_new)
            if res < best_res - tol:
                W = W_new
                best_res = res
                best_W = W.copy()
    
    return best_U, best_V, best_W

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

def _build_kronecker_strassen(n=4, m=4, p=5):
    """Kronecker-embedded Strassen baseline (rank 65)."""
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

    for i in range(n):
        for j in range(m):
            u = np.zeros(n * m); u[ui(i, j)] = 1
            v = np.zeros(m * p); v[vi(j, 4)] = 1
            w = np.zeros(p * n); w[wi(4, i)] = 1
            add(u, v, w)

    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T

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
    and prune terms with enhanced greedy strategy.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    # Flatten tensor vectors
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Vector length n*m*p
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
    t = np.sum(A, axis=1) # The target tensor vector
    
    # Solve A alpha = t
    # We know alpha = 1 is a solution.
    # Find null space N of A.
    N = null_space(A)
    
    if N.shape[1] == 0:
        return U, V, W
        
    # Current alpha
    alpha = np.ones(rank)
    
    # Enhanced greedy zeroing with multiple restart strategies
    best_alpha = alpha.copy()
    best_active_count = rank
    
    for restart in range(5):
        alpha = np.ones(rank)
        
        improved = True
        while improved:
            improved = False
            # Try all possible zeroings, pick the best one
            best_k = -1
            best_new_alpha = None
            best_active = 0
            
            for k in range(rank):
                if alpha[k] == 0:
                    continue
                # Try to set alpha[k] = 0
                n_k = N[:, k]
                norm_sq = np.dot(n_k, n_k)
                if norm_sq < 1e-12:
                    continue
                z = -n_k / norm_sq
                new_alpha = 1 + N @ z
                
                # Check if new_alpha has half-integer non-zeros
                valid = True
                active = 0
                for r in range(rank):
                    if abs(new_alpha[r]) > 1e-8:
                        if not _is_half_int(np.array([[new_alpha[r]]])):
                            valid = False
                            break
                        active += 1
                    else:
                        # Close to 0, check if it can be rounded to 0
                        if abs(new_alpha[r]) > 1e-4:
                            # Too large to be zero, but not half-integer
                            valid = False
                            break
                        # Will be rounded to 0
                        active += 0  # Not active
                
                if valid and active > best_active:
                    best_active = active
                    best_k = k
                    best_new_alpha = new_alpha.copy()
            
            # Find the k that, when zeroed, gives the FEWEST active terms
            best_k = -1
            best_new_alpha = None
            best_active = rank + 1
            
            for k in range(rank):
                if alpha[k] == 0:
                    continue
                # Try to set alpha[k] = 0
                n_k = N[:, k]
                norm_sq = np.dot(n_k, n_k)
                if norm_sq < 1e-12:
                    continue
                z = -n_k / norm_sq
                new_alpha = 1 + N @ z
                
                # Check if new_alpha has half-integer non-zeros
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
                
                if valid and active < best_active:
                    best_active = active
                    best_k = k
                    best_new_alpha = new_alpha.copy()
            
            if best_k >= 0 and best_new_alpha is not None:
                alpha = best_new_alpha
                alpha[np.abs(alpha) < 1e-6] = 0
                improved = True
        
        # Count active terms
        active_count = np.sum(np.abs(alpha) > 1e-8)
        if active_count < best_active_count:
            best_active_count = active_count
            best_alpha = alpha.copy()
    
    # Use the best alpha found
    alpha = best_alpha
    
    # Apply alpha with multiple scaling strategies
    final_U = U.copy()
    final_V = V.copy()
    final_W = W.copy()
    
    active = []
    for r in range(rank):
        if abs(alpha[r]) > 1e-8:
            a = alpha[r]
            success = False
            
            # Strategy 1: Scale U
            final_U[:, r] = U[:, r] * a
            if _is_half_int(final_U[:, r]):
                active.append(r)
                success = True
            else:
                final_U[:, r] = U[:, r]
            
            # Strategy 2: Scale V
            if not success:
                final_V[:, r] = V[:, r] * a
                if _is_half_int(final_V[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_V[:, r] = V[:, r]
            
            # Strategy 3: Scale W
            if not success:
                final_W[:, r] = W[:, r] * a
                if _is_half_int(final_W[:, r]):
                    active.append(r)
                    success = True
                else:
                    final_W[:, r] = W[:, r]
            
            # Strategy 4: Split scaling between U and V
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
            
            # Strategy 5: Split scaling between U and W
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
            
            # Strategy 6: Split scaling between V and W
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
            
            # If no strategy worked, we can't use this term
            if not success:
                # Skip this term (don't add to active)
                pass
        else:
            # alpha[r] is zero, skip
            pass
    
    # If we failed to scale some, we might have a problem.
    # For safety, if the number of active terms is not less than rank, return original.
    if len(active) >= rank:
        return U, V, W
        
    # Construct new matrices
    new_U = final_U[:, active]
    new_V = final_V[:, active]
    new_W = final_W[:, active]
    
    # Verify
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
        
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

    # 1) Kronecker Strassen baseline
    try:
        U, V, W = _build_kronecker_strassen(n, m, p)
        # Apply null space pruning
        U2, V2, W2 = _null_space_prune(U, V, W, n, m, p)
        consider(U2, V2, W2, "null_space")
        # Also consider original if pruning failed or was worse
        consider(U, V, W, "kronecker")
    except Exception:
        pass

    # 2) Try additional pruning passes on the best result
    if best is not None and time.time() < deadline - 30:
        _, Ub, Vb, Wb = best
        try:
            U3, V3, W3 = _null_space_prune(Ub, Vb, Wb, n, m, p)
            consider(U3, V3, W3, "second_prune")
        except Exception:
            pass

    # 3) Riemannian ALS refinement to find lower-rank decompositions
    if best is not None and time.time() < deadline - 20:
        _, Ub, Vb, Wb = best
        try:
            U4, V4, W4 = _riemannian_als_refine(Ub, Vb, Wb, n, m, p, max_iter=30)
            # Round and verify
            U4r = _round_half(U4); V4r = _round_half(V4); W4r = _round_half(W4)
            if _verify(U4r, V4r, W4r, n, m, p) and _is_half_int(U4r) and _is_half_int(V4r) and _is_half_int(W4r):
                r = U4r.shape[1]
                if best is None or r < best[0]:
                    best = (r, U4r, V4r, W4r)
        except Exception:
            pass

    # 4) Try Riemannian refinement from the Kronecker baseline directly
    try:
        U5, V5, W5 = _build_kronecker_strassen(n, m, p)
        if time.time() < deadline - 20:
            U5, V5, W5 = _riemannian_als_refine(U5, V5, W5, n, m, p, max_iter=30)
            U5r = _round_half(U5); V5r = _round_half(V5); W5r = _round_half(W5)
            if _verify(U5r, V5r, W5r, n, m, p) and _is_half_int(U5r) and _is_half_int(V5r) and _is_half_int(W5r):
                r = U5r.shape[1]
                if best is None or r < best[0]:
                    best = (r, U5r, V5r, W5r)
    except Exception:
        pass

    # 5) Fallback schoolbook
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
