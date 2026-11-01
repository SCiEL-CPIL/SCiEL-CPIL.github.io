"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)
    
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]
        
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 2  # A[1,0]
        u_d = 3  # A[1,1]
        
        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2        # B[0, 10]
        v_g2 = p + k2    # B[1, 10]
        w_c02 = k2 * n + 0   # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]
        
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=60):
    """
    Attempt to reduce rank by finding a minimal subset of terms that still
    reconstructs the target tensor. Uses QR with column pivoting and greedy
    elimination.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    
    # Build the matricized tensor matrix: M[:, r] = vec(U[:,r] ⊗ V[:,r] ⊗ W[:,r])
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    T_flat = T.flatten()
    
    # Step 1: Use QR with column pivoting to find numerically independent columns
    try:
        Q, R, P = np.linalg.qr(M, mode='complete')
        # R is upper triangular; diagonal entries indicate column importance
        # P is the permutation matrix
        diag_abs = np.abs(np.diag(R))
        # Find the rank based on threshold
        tol = max(M.shape) * np.finfo(float).eps * (diag_abs[0] if diag_abs.size > 0 else 1.0)
        num_rank = np.sum(diag_abs > tol)
        
        if num_rank < current_rank:
            # Get the pivot columns (first num_rank in pivoted order)
            pivot_cols = P[:num_rank, :].argmax(axis=1)
            pivot_cols = np.sort(pivot_cols)
            U_new = U[:, pivot_cols]
            V_new = V[:, pivot_cols]
            W_new = W[:, pivot_cols]
            if _verify(U_new, V_new, W_new, T) and U_new.shape[1] < current_rank:
                return U_new, V_new, W_new
    except Exception:
        pass
    
    # Step 2: Greedy backward elimination - try removing one column at a time
    active = list(range(current_rank))
    removed_any = False
    
    for attempt in range(current_rank * 2):
        if time.time() - start_time > max_time:
            break
        if len(active) <= 1:
            break
        
        found_removal = False
        for idx in range(len(active)):
            if time.time() - start_time > max_time:
                break
            
            r = active[idx]
            others = [a for a in active if a != r]
            
            # Check if T_flat is in the column space of M[:, others]
            M_sub = M[:, others]
            
            # Use least squares to check if T_flat can be reconstructed
            try:
                sol, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, T_flat, rcond=1e-10)
                residual = np.linalg.norm(M_sub @ sol - T_flat)
            except Exception:
                continue
            
            if residual < 1e-8:
                # Column r can be removed
                active.remove(r)
                removed_any = True
                found_removal = True
                break
        
        if not found_removal:
            break
    
    if removed_any:
        keep = sorted(active)
        U_red = U[:, keep]
        V_red = V[:, keep]
        W_red = W[:, keep]
        if _verify(U_red, V_red, W_red, T):
            return U_red, V_red, W_red
    
    return None

def _try_double_removal(U, V, W, T, max_time=30):
    """
    Try removing pairs of columns simultaneously. This can find reductions
    that single-column removal misses.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 2:
        return None
    
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    T_flat = T.flatten()
    
    active = list(range(current_rank))
    removed_any = False
    
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        if len(active) <= 2:
            break
        
        found_removal = False
        for i in range(len(active)):
            if time.time() - start_time > max_time:
                break
            for j in range(i + 1, len(active)):
                if time.time() - start_time > max_time:
                    break
                
                r1 = active[i]
                r2 = active[j]
                others = [a for a in active if a != r1 and a != r2]
                
                M_sub = M[:, others]
                try:
                    sol, residuals, rank_sub, sv = np.linalg.lstsq(M_sub, T_flat, rcond=1e-10)
                    residual = np.linalg.norm(M_sub @ sol - T_flat)
                except Exception:
                    continue
                
                if residual < 1e-8:
                    active.remove(r2)
                    active.remove(r1)
                    removed_any = True
                    found_removal = True
                    break
            
            if found_removal:
                break
        
        if not found_removal:
            break
    
    if removed_any:
        keep = sorted(active)
        U_red = U[:, keep]
        V_red = V[:, keep]
        W_red = W[:, keep]
        if _verify(U_red, V_red, W_red, T):
            return U_red, V_red, W_red
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave margin before 150s limit

    T = _build_target_tensor()

    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()

    if not _verify(U, V, W, T):
        # Fall back to schoolbook rank 44
        U, V, W = _build_schoolbook()
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        raise RuntimeError("No valid decomposition found")
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try rank reduction with generous time budget
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 3
    if remaining_time > 5:
        reduced = _try_rank_reduction(U, V, W, T, max_time=remaining_time)
        if reduced is not None:
            rU, rV, rW = reduced
            if rU.shape[1] < best_rank:
                best_U, best_V, best_W = rU, rV, rW
                best_rank = rU.shape[1]
    
    # Try double removal if time permits
    elapsed = time.time() - start_time
    remaining_time = DEADLINE - elapsed - 3
    if remaining_time > 10 and best_rank > 2:
        reduced2 = _try_double_removal(best_U, best_V, best_W, T, max_time=remaining_time)
        if reduced2 is not None:
            rU, rV, rW = reduced2
            if rU.shape[1] < best_rank:
                best_U, best_V, best_W = rU, rV, rW
                best_rank = rU.shape[1]
    
    # Final verification
    if not _verify(best_U, best_V, best_W, T):
        # Fallback to original Strassen
        U, V, W = _build_strassen_decomposition()
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        # Last resort: schoolbook
        U, V, W = _build_schoolbook()
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
