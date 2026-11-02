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
    Attempt to reduce rank via greedy single-column removal and
    column-absorption (solve for alpha so that col_r can be folded into col_s).
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    
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
    
    # Column absorption: for each pair (r, s), try to find alpha such that
    # replacing col_s with col_s + alpha*col_r allows removing col_r.
    # This works when the rank-1 tensors are "close" in some sense.
    current_rank = U.shape[1]
    if current_rank > 2 and time.time() - start_time < max_time - 5:
        for attempt in range(current_rank):
            if time.time() - start_time > max_time:
                break
            absorbed = False
            for r in range(current_rank):
                if time.time() - start_time > max_time:
                    break
                for s in range(current_rank):
                    if time.time() - start_time > max_time:
                        break
                    if r == s:
                        continue
                    # Try: new_col_s = col_s + alpha * col_r, then remove col_r
                    # We need: sum_{r' != r} u_{r'} v_{r'} w_{r'}^T = T
                    # where for r'=s we use (v_s + alpha*v_r) and w stays
                    # This is equivalent to: the residual after removing r must be
                    # expressible by modifying s.
                    # Simpler approach: just try alpha in {-1, -0.5, 0.5, 1, 2}
                    for alpha in [-1.0, -0.5, 0.5, 1.0, 2.0]:
                        if time.time() - start_time > max_time:
                            break
                        U_mod = U.copy()
                        V_mod = V.copy()
                        W_mod = W.copy()
                        # Modify column s: V_s += alpha * V_r
                        V_mod[:, s] += alpha * V_mod[:, r]
                        # Remove column r
                        mask = [i for i in range(current_rank) if i != r]
                        U_test = U_mod[:, mask]
                        V_test = V_mod[:, mask]
                        W_test = W_mod[:, mask]
                        if _verify_fast(U_test, V_test, W_test, T):
                            U, V, W = U_test, V_test, W_test
                            current_rank -= 1
                            absorbed = True
                            break
                    if absorbed:
                        break
                if absorbed:
                    break
            if not absorbed:
                break
    
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
    Try to find a rank-38 decomposition by using 4 Strassen blocks (28)
    plus a rank-10 decomposition for the last 3 columns.
    
    Strategy: use 7 Strassen terms for cols 8,9 and try to cover col 10
    with 3 terms by modifying one Strassen term to share structure.
    
    Alternative: use a different partition - 3 Strassen blocks (21) +
    a rank-17 decomposition for the last 5 columns.
    """
    n, m, p = 2, 2, 11
    T = _build_target_tensor()
    
    # Strategy 1: 4 Strassen (28) + 10 for last 3 cols
    # Try: 7 Strassen for cols 8,9 + 3 custom terms for col 10
    # The 3 custom terms must produce:
    #   C[0,10] = a*B[0,10] + b*B[1,10]
    #   C[1,10] = c*B[0,10] + d*B[1,10]
    # with 3 rank-1 terms. This requires the 2x2 matrix [[a,b],[c,d]]
    # to have rank <= 3 in the tensor sense, which is always true (rank 4 max).
    # But we need exactly 3 terms. The schoolbook uses 4. Can we do 3?
    # For 2x2 * 2x1, the tensor has rank 4 (it's the identity tensor of size 2x2x1).
    # Actually no - it's a 4x1x2 tensor (A has 4 entries, B has 2, C has 2).
    # The rank of this sub-tensor is 4. So we need 4 terms. Rank 10 is not possible.
    
    # Strategy 2: 3 Strassen blocks (21) + 17 for last 5 cols
    # 5 cols = 2 pairs + 1 single. 2 pairs need 14, single needs 4. Total 18.
    # Can we do 17? Try modifying one Strassen term in the last pair to share with single.
    
    # Strategy 3: Direct search - try rank 38 with a parameterized family
    # Use 5 Strassen blocks but modify the 5th to also cover col 10 partially.
    # Block 5 covers cols 10, and we add 1 extra term.
    # 5*7 + 1 = 36? No, that doesn't work dimensionally.
    
    # Strategy 4: Use the fact that for 2x2*2x11, we can use
    # 5 blocks of (2x2 * 2x2) for cols 0-9 (rank 35) + 4 for col 10 = 39.
    # Or: 4 blocks of (2x2*2x2) for cols 0-7 (rank 28) + 11 for cols 8-10.
    # The 11 = 7 (Strassen for 8,9) + 4 (schoolbook for 10).
    
    # Let's try a different approach: can we use a 2x2*2x3 block of rank 10?
    # The tensor for 2x2*2x3 has dimensions 4x6x6 = 144 entries.
    # Its rank is at most 24 (schoolbook). Strassen gives 7+4=11.
    # Can we do 10? Let's set up the linear system.
    
    # For 2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We need rank-10: 10 triples (u_r in R^4, v_r in R^6, w_r in R^6)
    # such that sum_r u_r v_r w_r^T = T
    
    # Set up: unknowns are the 10*4 + 10*6 + 10*6 = 160 values
    # But we can fix gauge: set u_r[0]=1 for each r (if nonzero)
    # Actually, let's just try a specific parameterization.
    
    # Use 7 Strassen terms for cols 0,1 of the 3-col block + 3 terms for col 2
    # Strassen for 2x2*2x2 (cols 0,1):
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    # where B = [[e,f],[g,h]] for the 2x2 sub-block
    
    # For col 2, B has entries [p, q] (B[0,2]=p, B[1,2]=q)
    # C[0,2] = a*p + b*q, C[1,2] = c*p + d*q
    
    # 3 terms for col 2:
    # t8 = a*p -> contributes to C[0,2]
    # t9 = b*q -> contributes to C[0,2]  
    # t10 = c*p -> contributes to C[1,2]
    # Missing: d*q for C[1,2]
    
    # Can we modify m6 or m7 to include d*q?
    # m6 = (c-a)(e+f) -> if we change to (c-a)(e+f) + beta*d*q
    # This adds beta*d*q to C[1,1] (where m6 appears) -- unwanted!
    
    # m7 = (b-d)(g+h) -> if we change to (b-d)(g+h) + beta*d*q
    # This adds beta*d*q to C[0,0] (where m7 appears) -- unwanted!
    
    # What about modifying m1?
    # m1 = (a+d)(e+h) -> (a+d)(e+h) + beta*d*q
    # m1 appears in C[0,0] and C[1,1]
    # C[0,0] gets +beta*d*q -- unwanted (should be a*e+b*g)
    # C[1,1] gets +beta*d*q -- this could help! C[1,1] = m1-m2+m3+m6
    # But we also get unwanted in C[0,0].
    
    # It seems impossible to get rank 10 for 2x2*2x3 with this structure.
    
    # Strategy 5: Try a completely different partition.
    # Use 2x2*2x4 blocks? 4 cols = 2 pairs, rank 14 per block.
    # 2 blocks of 4 cols (8 cols) = 28, remaining 3 cols = 11. Total 39.
    # Same as before.
    
    # Strategy 6: Accept rank 39 is optimal for this structure.
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()
    
    if _verify(U, V, W, T):
        best_U, best_V, best_W = U, V, W
        best_rank = U.shape[1]
        
        # Try rank reduction (faster version)
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
