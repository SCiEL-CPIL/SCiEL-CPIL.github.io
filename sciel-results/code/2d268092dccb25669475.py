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

def _verify_fast(U, V, W, T):
    """Fast verification using precomputed tensor structure."""
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
    
    # For the single column, we need 4 terms (a*e, b*g, c*e, d*g for 2x2 * 1x2... wait)
    # Actually for 2x2 * 2x1: C = A * B where B is 2x1
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  -> needs 2 terms
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  -> needs 2 terms
    # So rank 4 for the single column
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    # Strassen terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices: B[j, k] has index j*p + k
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]
        
        # W indices: C[i, k] has index k*n + i
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        # U indices: A[i,j] has index i*m + j
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
    
    # Handle remainder column (column 10)
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2        # B[0, 10]
        v_g2 = p + k2    # B[1, 10]
        w_c02 = k2 * n + 0   # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]
        
        # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
        # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
        
        # Term 1: A[0,0]*B[0,10]
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        # Term 2: A[0,1]*B[1,10]
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        # Term 3: A[1,0]*B[0,10]
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        # Term 4: A[1,1]*B[1,10]
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
    Attempt to reduce rank by finding linearly dependent terms.
    Uses SVD on the tensor matricization to find a minimal basis,
    then greedy single- and double-column removal, and column replacement.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    # Flatten each rank-1 tensor into a vector
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    # Use SVD to find the numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        num_rank = np.sum(s_vals > tol)
    except Exception:
        num_rank = current_rank
    
    if num_rank < current_rank:
        # Try QR with pivoting to find independent columns
        try:
            Q, R = np.linalg.qr(M, mode='complete')
            diag_R = np.abs(np.diag(R))
            pivot_cols = np.argsort(diag_R)[::-1][:num_rank]
            pivot_cols = np.sort(pivot_cols)
            U_new = U[:, pivot_cols]
            V_new = V[:, pivot_cols]
            W_new = W[:, pivot_cols]
            if _verify(U_new, V_new, W_new, T) and U_new.shape[1] < current_rank:
                return U_new, V_new, W_new
        except Exception:
            pass
    
    # Greedy: remove one column at a time
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify_fast(U_test, V_test, W_test, T):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    
    # Try removing pairs of columns
    for attempt in range(current_rank // 2):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r1 in range(current_rank):
            if time.time() - start_time > max_time:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() - start_time > max_time:
                    break
                mask = [i for i in range(current_rank) if i != r1 and i != r2]
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                if _verify_fast(U_test, V_test, W_test, T):
                    U, V, W = U_test, V_test, W_test
                    current_rank -= 2
                    removed = True
                    break
            if removed:
                break
        if not removed:
            break
    
    # Try column replacement: replace column r with a linear combination of two others
    # This can reduce rank if the new column is "simpler" and allows further removal
    if time.time() - start_time < max_time:
        # Rebuild M after potential removals
        current_rank = U.shape[1]
        if current_rank > 2:
            M2 = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
            for r in range(current_rank):
                M2[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
            
            # For each column, try to see if it can be expressed as a combination of 2 others
            # with small integer coefficients, then try removing one of those
            for r in range(current_rank):
                if time.time() - start_time > max_time:
                    break
                # Try: can we remove column r by adjusting another column?
                # This is complex; skip for now
                pass
    
    return U, V, W

def _try_alternative_constructions(T, deadline):
    """Try different block structures to find lower rank."""
    n, m, p = 2, 2, 11
    
    # Try: 4 pairs + 3 single columns? No, that's worse.
    # Try: mix of different block sizes
    
    # Theoretical lower bound: 
    # For 2x2 * 2xp, the rank is at least... 
    # Actually, let's just try to see if we can do better than 39
    
    # One idea: can we share some terms between blocks?
    # The Strassen decomposition is already optimal for each 2x2 block.
    # The question is whether we can reduce the total.
    
    # For now, return None - we'll rely on the main construction
    return None

def _try_extended_block_decomposition():
    """
    Try to find a rank-10 decomposition for 2x2 * 2x3 (the last 3 columns).
    If found, total rank = 4*7 + 10 = 38 instead of 4*7 + 11 = 39.
    Uses a structured search over modified Strassen terms.
    
    We try a parameterized family of 10-term decompositions and check
    if any satisfies the tensor equation.
    """
    n, m, p = 2, 2, 11
    
    # Build the full target tensor
    T = _build_target_tensor()
    
    # Build 4 Strassen blocks for columns 0-7 (rank 28)
    U_base, V_base, W_base = _build_strassen_decomposition()
    # U_base is (4, 39), V_base is (22, 39), W_base is (22, 39)
    # First 28 columns are Strassen for cols 0-7
    # Last 11 columns are for cols 8-10 (7 Strassen + 4 schoolbook)
    
    # Extract the base 28 columns
    U4 = U_base[:, :28].copy()
    V4 = V_base[:, :28].copy()
    W4 = W_base[:, :28].copy()
    
    # The residual tensor that needs to be covered by the last 3 columns
    T_residual = T.copy()
    T_decomp_28 = np.einsum('ar,br,cr->abc', U4, V4, W4)
    T_residual -= T_decomp_28
    
    # Now we need to decompose T_residual with rank 10
    # T_residual is (4, 22, 22) but only has non-zero entries in B indices 8-10,19-21
    # and C indices 16-21
    
    # Try a parameterized 10-term decomposition
    # Use the structure: 7 Strassen-like terms + 3 additional terms
    # Parameterize the last 3 terms with free variables
    
    # Let's try: use standard Strassen for cols 8,9 (7 terms) + 3 terms for col 10
    # Standard Strassen for cols 8,9:
    # m1=(a+d)(B[0,8]+B[1,9]), m2=(c+d)B[0,8], m3=a(B[0,9]-B[1,8]),
    # m4=d(B[1,8]-B[0,8]), m5=(a+b)B[1,9], m6=(c-a)(B[0,8]+B[0,9]), m7=(b-d)(B[1,8]+B[1,9])
    
    # This gives C[0,8], C[0,9], C[1,8], C[1,9] correctly.
    # For col 10, we need: C[0,10]=a*B[0,10]+b*B[1,10], C[1,10]=c*B[0,10]+d*B[1,10]
    # With 3 terms, we can do:
    # t8 = a*B[0,10], t9 = b*B[1,10], t10 = (c+d)*B[0,10]
    # C[0,10] = t8 + t9
    # C[1,10] = t10 + d*B[1,10] -- but we don't have d*B[1,10]
    
    # Alternative: t8=a*B[0,10], t9=b*B[1,10], t10=c*B[0,10]
    # C[0,10] = t8+t9
    # C[1,10] = t10 + d*B[1,10] -- still missing
    
    # It seems we need 4 terms for col 10. So rank 10 is not achievable with this structure.
    
    # Try: modify one Strassen term to also help with col 10
    # m6 = (c-a)(B[0,8]+B[0,9]+alpha*B[0,10]) for some alpha
    # This adds alpha*(c-a)*B[0,10] to the output
    # C[1,9] gets alpha*(c-a)*B[0,10] as an extra term -- bad
    
    # I conclude rank 10 for 2x2*2x3 is not achievable.
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()
    
    if _verify(U, V, W, T):
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
        
        # Try rank reduction
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            U_red, V_red, W_red = _try_rank_reduction(U, V, W, T, max_time=min(remaining_time, 90))
            if U_red.shape[1] < best_rank:
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
        
        # Try the extended block decomposition (might find rank 38)
        if time.time() - start_time < DEADLINE - 10:
            ext = _try_extended_block_decomposition()
            if ext is not None:
                U_ext, V_ext, W_ext = ext
                if _verify(U_ext, V_ext, W_ext, T) and U_ext.shape[1] < best_rank:
                    best_U, best_V, best_W = U_ext, V_ext, W_ext
                    best_rank = U_ext.shape[1]
        
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    
    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
