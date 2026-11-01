"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np
from fractions import Fraction

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


def _build_target_tensor_exact():
    """Build the multiplication tensor with exact integer entries."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T


def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor.
    Checks both nonzero entries match AND all-zero entries remain zero."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    # Check all entries: both where T is nonzero and where T is zero
    return np.allclose(T, T_decomp, atol=1e-9)


def _verify_exact(U, V, W, T_exact=None):
    """Exact verification using Fraction arithmetic."""
    if T_exact is None:
        T_exact = _build_target_tensor_exact()
    U_frac = np.array([[Fraction(str(U[i, r])) for r in range(U.shape[1])] for i in range(U.shape[0])], dtype=object)
    V_frac = np.array([[Fraction(str(V[i, r])) for r in range(V.shape[1])] for i in range(V.shape[0])], dtype=object)
    W_frac = np.array([[Fraction(str(W[i, r])) for r in range(W.shape[1])] for i in range(W.shape[0])], dtype=object)
    
    rank = U_frac.shape[1]
    for a in range(U_frac.shape[0]):
        for b in range(V_frac.shape[0]):
            for c in range(W_frac.shape[0]):
                val = sum(U_frac[a, r] * V_frac[b, r] * W_frac[c, r] for r in range(rank))
                if val != T_exact[a, b, c]:
                    return False
    return True


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
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)

    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1

        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3

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
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = k2 * n + 0
        w_c12 = k2 * n + 1

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


def _build_subspace_splitting_decomposition():
    """
    Subspace splitting method: analyze the Grassmannian geometry of the
    matrix multiplication tensor to find a lower-rank decomposition.
    
    Key insight: For A(2x2) * B(2x11) = C(11x2), we can view this as:
    - The A-matrix has 4 independent entries: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    - The B-matrix has 22 entries organized as 11 columns of 2 rows
    - The C-matrix has 22 entries organized as 11 columns of 2 rows
    
    The tensor T[a_idx, b_idx, c_idx] = 1 when the multiplication connects them.
    
    Strategy: Group the 11 B-columns into blocks and find shared linear combinations
    across blocks. The key is that Strassen's 7-term formula for 2x2*2x2 can be
    modified to handle 2x2*2x3 in fewer than 7+4=11 terms by sharing structure.
    
    For 2x2 * 2x3:
    C[0,0] = ae + bg, C[0,1] = af + bh, C[0,2] = ai + bj  (where i=B[0,2], j=B[1,2])
    C[1,0] = ce + dg, C[1,1] = cf + dh, C[1,2] = ci + dj
    
    Try a 10-term decomposition:
    s1 = (a+d)(e+h)     -> contributes to C[0,0], C[1,1]
    s2 = (c+d)e         -> contributes to C[1,0], C[1,1]
    s3 = a(f-h)         -> contributes to C[0,1], C[1,1]
    s4 = d(g-e)         -> contributes to C[0,0], C[1,0]
    s5 = (a+b)h         -> contributes to C[0,0], C[0,1]
    s6 = (c-a)(e+f)     -> contributes to C[1,1]
    s7 = (b-d)(g+h)     -> contributes to C[0,0]
    s8 = a*i            -> contributes to C[0,2]
    s9 = b*j            -> contributes to C[0,2]
    s10 = (c+d)*i - c*i + d*j ... this doesn't work cleanly.
    
    Alternative: Try to modify s4 to help with column 2.
    s4' = d*(g-e) + something involving i,j
    
    Let's try a different approach: use the fact that for the last column,
    we need: C[0,2] = a*i + b*j and C[1,2] = c*i + d*j
    
    What if we modify s5 = (a+b)h to s5' = (a+b)(h + alpha*i)?
    Then s5' contributes: a*h + a*alpha*i + b*h + b*alpha*i
    This gives us a*i and b*i terms, but we need a*i and b*j.
    
    What if s5 = (a+b)h and s8 = a*(i - beta*h), s9 = b*(j - gamma*h)?
    Then:
    C[0,2] gets: a*i - a*beta*h + b*j - b*gamma*h
    We need to cancel the h terms: a*beta + b*gamma must be absorbed elsewhere.
    
    This is getting complex. Let me try a direct parameterized search.
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    # First, build the standard Strassen for 5 pairs (rank 35)
    # Then try to handle the last column with fewer than 4 terms by
    # modifying existing Strassen terms.
    
    # Actually, let's try a completely different approach:
    # Use 2x2 * 2x2 Strassen (7 terms) for 5 pairs = 35 terms
    # For the 11th column, try to use 3 terms instead of 4.
    # 
    # For column 10: C[0,10] = a*B[0,10] + b*B[1,10]
    #                C[1,10] = c*B[0,10] + d*B[1,10]
    #
    # With 3 terms: t1*u1*v1 + t2*u2*v2 + t3*u3*v3
    # We need:
    #   a*B[0,10] appears in C[0,10]
    #   b*B[1,10] appears in C[0,10]
    #   c*B[0,10] appears in C[1,10]
    #   d*B[1,10] appears in C[1,10]
    #
    # With 3 rank-1 terms, we have 3 U vectors (each in R^4), 3 V vectors (each in R^22),
    # 3 W vectors (each in R^22). The constraint is that the 4 products must be formed
    # from these 3 terms.
    #
    # Let's say:
    #   term1: U1 = [x1, x2, x3, x4], V1 has B[0,10] component, W1 has C[0,10] or C[1,10]
    #   term2: U2 = [y1, y2, y3, y4], V2 has B[0,10] or B[1,10] component, W2 has C entry
    #   term3: U3 = [z1, z2, z3, z4], V3 has B component, W3 has C component
    #
    # This is a system of bilinear equations. Let me set up the specific case:
    #
    # Suppose:
    #   t1: U=[1,0,0,0], V has v1 at B[0,10], W has w1 at C[0,10]  -> gives a*B[0,10]*w1 to C[0,10]
    #   t2: U=[0,1,0,0], V has v2 at B[1,10], W has w2 at C[0,10]  -> gives b*B[1,10]*w2 to C[0,10]
    #   t3: U=[0,0,1,1], V has v3 at B[0,10]+B[1,10]... no, V must be a single vector.
    #
    # Actually, let me think about this differently. The 3 terms must collectively produce:
    #   C[0,10] = a*B[0,10] + b*B[1,10]
    #   C[1,10] = c*B[0,10] + d*B[1,10]
    #
    # And they must NOT produce any other C entries (or any other entries must cancel).
    #
    # Let's parameterize:
    #   Term r has U_r = (u_r1, u_r2, u_r3, u_r4), V_r (22-dim), W_r (22-dim)
    #
    # For the contribution to C[0,10] (index 20 in C-space, since C[k*n+i] with k=10,i=0):
    #   sum_r u_r1 * V_r[0] * W_r[20] = 1  (coefficient of a*B[0,10])
    #   sum_r u_r2 * V_r[1] * W_r[20] = 1  (coefficient of b*B[1,10])
    #
    # For C[1,10] (index 21):
    #   sum_r u_r3 * V_r[0] * W_r[21] = 1  (coefficient of c*B[0,10])
    #   sum_r u_r4 * V_r[1] * W_r[21] = 1  (coefficient of d*B[1,10])
    #
    # And for all other C entries C[k,n+i] where (k,i) != (10,0) or (10,1):
    #   sum_r u_rj * V_r[...] * W_r[k*2+i] = 0
    #
    # And for the V components: V_r should only have nonzero entries at positions
    # corresponding to B[0,10] (index 10) and B[1,10] (index 21), otherwise
    # it would create unwanted products.
    #
    # So V_r = (0,...,0, v_r0, 0,...,0, v_r1, 0,...,0) where v_r0 is at position 10
    # and v_r1 is at position 21.
    #
    # Similarly, W_r should only have nonzero entries at positions 20 and 21
    # (C[0,10] and C[1,10]), otherwise it creates unwanted outputs.
    #
    # So W_r = (0,...,0, w_r0, w_r1, 0,...,0) where w_r0 is at position 20, w_r1 at 21.
    #
    # Now the equations become:
    #   C[0,10]: sum_r u_r1 * v_r0 * w_r0 = 1
    #   C[0,10]: sum_r u_r2 * v_r1 * w_r0 = 1
    #   C[1,10]: sum_r u_r3 * v_r0 * w_r1 = 1
    #   C[1,10]: sum_r u_r4 * v_r1 * w_r1 = 1
    #
    # With 3 terms, we have 3*4 + 3*2 + 3*2 = 12+6+6 = 24 unknowns, and 4 equations.
    # This is very underdetermined, so solutions should exist.
    #
    # Let's try a simple solution:
    #   Term 1: U=[1,0,0,0], V=[v10=1, v11=0], W=[w10=1, w11=0]
    #     -> contributes a*1*1 = a to C[0,10] ✓
    #   Term 2: U=[0,1,0,0], V=[v20=0, v21=1], W=[w20=1, w21=0]
    #     -> contributes b*1*1 = b to C[0,10] ✓
    #   Term 3: U=[0,0,1,1], V=[v30=1, v31=1], W=[w30=0, w31=1]
    #     -> contributes c*1*1 = c to C[1,10] ✓
    #     -> contributes d*1*1 = d to C[1,10] ✓
    #
    # Wait, that's 3 terms! But let me verify:
    #   Term 3: U=[0,0,1,1], V has v30=1 at B[0,10] and v31=1 at B[1,10]
    #   W has w31=1 at C[1,10]
    #   Contribution to C[1,10]: 0*... + 0*... + 1*1*1 + 1*1*1 = c + d ✓
    #   Contribution to C[0,10]: 0*... + 0*... + 1*1*0 + 1*1*0 = 0 ✓ (no unwanted)
    #
    # So 3 terms suffice for the last column! Total rank = 5*7 + 3 = 38.
    
    rank = num_pairs * 7 + 3  # 38
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1

        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3

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

    # Handle remainder column (column 10) with 3 terms
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        v_e2 = k2          # B[0,10] index
        v_g2 = p + k2      # B[1,10] index
        w_c02 = k2 * n + 0 # C[0,10] index
        w_c12 = k2 * n + 1 # C[1,10] index

        # Term 1: a * B[0,10] -> C[0,10]
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1

        # Term 2: b * B[1,10] -> C[0,10]
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1

        # Term 3: (c+d) * (B[0,10] + B[1,10]) -> C[1,10]
        U[2, col] = 1.0
        U[3, col] = 1.0
        V[v_e2, col] = 1.0
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
    then greedy single- and double-column removal.
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

    # Try removing triples of columns
    for attempt in range(current_rank // 3):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r1 in range(current_rank):
            if time.time() - start_time > max_time:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() - start_time > max_time:
                    break
                for r3 in range(r2 + 1, current_rank):
                    if time.time() - start_time > max_time:
                        break
                    mask = [i for i in range(current_rank) if i != r1 and i != r2 and i != r3]
                    U_test = U[:, mask]
                    V_test = V[:, mask]
                    W_test = W[:, mask]
                    if _verify_fast(U_test, V_test, W_test, T):
                        U, V, W = U_test, V_test, W_test
                        current_rank -= 3
                        removed = True
                        break
                if removed:
                    break
            if removed:
                break
        if not removed:
            break

    return U, V, W


def _try_cross_block_reduction(U, V, W, T, max_time=30):
    """
    Try to find a lower-rank decomposition by allowing linear combinations
    of the original terms. This is a local optimization in the space of
    rank-r decompositions.
    
    Also tries a "subspace splitting" approach: identify groups of terms that
    operate on disjoint subsets of the B and C spaces, and check if terms
    from different groups can be merged or if the total rank can be reduced
    by exploiting linear dependencies across groups.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = 4, 22, 22

    # Look for terms that have the same U and V vectors (up to scaling)
    # but different W vectors, and see if we can merge them.
    for r1 in range(current_rank):
        if time.time() - start_time > max_time:
            break
        for r2 in range(r1 + 1, current_rank):
            if time.time() - start_time > max_time:
                break
            U1 = U[:, r1]
            U2 = U[:, r2]
            if np.linalg.norm(U1) < 1e-10 or np.linalg.norm(U2) < 1e-10:
                continue
            ratio = np.dot(U1, U2) / np.dot(U1, U1)
            if abs(np.dot(U1, U2) - ratio * np.dot(U1, U1)) < 1e-8:
                V1 = V[:, r1]
                V2 = V[:, r2]
                if np.linalg.norm(V1) < 1e-10 or np.linalg.norm(V2) < 1e-10:
                    continue
                ratio_v = np.dot(V1, V2) / np.dot(V1, V1)
                if abs(np.dot(V1, V2) - ratio_v * np.dot(V1, V1)) < 1e-8:
                    W_new = W[:, r1] + ratio * ratio_v * W[:, r2]
                    mask = [i for i in range(current_rank) if i != r1 and i != r2]
                    U_test = np.column_stack([U[:, mask], U[:, r1]])
                    V_test = np.column_stack([V[:, mask], V[:, r1]])
                    W_test = np.column_stack([W[:, mask], W_new])
                    if _verify_fast(U_test, V_test, W_test, T):
                        U, V, W = U_test, V_test, W_test
                        current_rank -= 1

    # Subspace splitting: try to merge terms from different Strassen blocks
    # by checking if their combined contribution can be represented with fewer terms.
    # This is particularly relevant when the last column's 3 terms might interact
    # with Strassen terms from adjacent blocks.
    if time.time() - start_time < max_time:
        # Try replacing pairs of terms with a single term (rank reduction by 1)
        # by solving a small linear system.
        for r1 in range(current_rank):
            if time.time() - start_time > max_time:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() - start_time > max_time:
                    break
                # Check if U1⊗V1⊗W1 + U2⊗V2⊗W2 can be replaced by U3⊗V3⊗W3
                # This requires the 4x22x22 tensor to have rank 1.
                # Quick check: if U1 and U2 are proportional, or V1 and V2 are proportional,
                # or W1 and W2 are proportional, we can merge.
                
                # Check U proportionality
                if np.linalg.norm(U[:, r1]) > 1e-10 and np.linalg.norm(U[:, r2]) > 1e-10:
                    cross_U = np.abs(np.outer(U[:, r1], U[:, r2]) - np.outer(U[:, r2], U[:, r1]))
                    if np.max(cross_U) < 1e-8 * np.max(np.abs(np.outer(U[:, r1], U[:, r2]))):
                        # U vectors are proportional
                        ratio_u = np.dot(U[:, r1], U[:, r2]) / np.dot(U[:, r1], U[:, r1])
                        V_new = V[:, r1] + ratio_u * V[:, r2]
                        W_new = W[:, r1] + ratio_u * W[:, r2]
                        mask = [i for i in range(current_rank) if i != r1 and i != r2]
                        U_test = np.column_stack([U[:, mask], U[:, r1]])
                        V_test = np.column_stack([V[:, mask], V_new])
                        W_test = np.column_stack([W[:, mask], W_new])
                        if _verify_fast(U_test, V_test, W_test, T):
                            U, V, W = U_test, V_test, W_test
                            current_rank -= 1
                            break
                if time.time() - start_time > max_time:
                    break
            if time.time() - start_time > max_time:
                break

    return U, V, W


def _try_3col_rank10():
    """
    Try to find a rank-10 decomposition for 2x2 * 2x3.
    
    The 2x2 * 2x3 problem has 12 output entries, each a sum of 2 products.
    Total 24 unit products. With 10 rank-1 terms, we need clever sharing.
    
    Let's try a systematic search over parameterized families.
    """
    n, m = 2, 2
    p3 = 3
    # Local subproblem: A is 2x2, B is 2x3, C is 3x2
    # T shape: (4, 6, 6)
    T3 = np.zeros((4, 6, 6), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i * 2 + j
                b = j * 3 + k
                c = k * 2 + i
                T3[a, b, c] = 1.0

    # Try various 10-term constructions
    # The key idea: can we share terms between the 2x2 Strassen block and the 3rd column?
    
    # Construction attempt:
    # Use 7 Strassen terms for columns 0,1
    # Plus 3 terms for column 2, but modify some Strassen terms to help
    
    # Let's try:
    # t1-t7: standard Strassen for cols 0,1
    # t8: a*(e+g2) where g2 = B[0,2]
    # t9: b*(f+h2) where h2 = B[1,2]  
    # t10: (c+d)*(e+g2) - but this doesn't work cleanly
    
    # Alternative: try a completely different 10-term pattern
    # C[0,0] = ae + bg
    # C[0,1] = af + bh
    # C[0,2] = ag + bj
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    # C[1,2] = cg + dj
    
    # Try:
    # s1 = (a+d)(e+h)     -> C[0,0] += 1, C[1,1] += 1
    # s2 = (c+d)e         -> C[1,0] += 1, C[1,1] -= 1
    # s3 = a(f-h)         -> C[0,1] += 1, C[1,1] += 1
    # s4 = d(g-e)         -> C[0,0] += 1, C[1,0] += 1
    # s5 = (a+b)h         -> C[0,0] -= 1, C[0,1] += 1
    # s6 = (c-a)(e+f)     -> C[1,1] += 1
    # s7 = (b-d)(g+h)     -> C[0,0] += 1
    # s8 = a*g            -> C[0,2] += 1
    # s9 = b*j            -> C[0,2] += 1
    # s10 = c*g + d*j     -> NOT rank 1!
    
    # So we need: c*g and d*j separately, but that's 2 more terms = 12 total.
    # Unless we can modify existing terms.
    
    # What if s10 = (c+d)*j, and we adjust s4 to provide c*(g-j)?
    # s4' = d*(g-e) + c*(g-j) = d*g - d*e + c*g - c*j
    # That's not rank 1 either.
    
    # What if we use: s8 = a*(g), s9 = b*(j), s10 = (c+d)*(g) ... no
    # s10 = c*g + d*j is the problem.
    
    # Try: s10 = (c-d)*g + d*(g+j) ... no, that's 2 terms.
    
    # I think rank 10 for 2x2*2x3 is not achievable with simple patterns.
    # Let's verify: the lower bound. The tensor has 12 nonzero entries in C-space,
    # each requiring 2 products. The A-space has 4 entries, B-space has 6.
    # The rank must be at least max(4, 6, 12/2) = 6, but structural constraints
    # likely push it higher.
    
    # For now, return None (no improvement found)
    return None


def _try_aggressive_merging(U, V, W, T, max_time=45):
    """
    Try more aggressive merging: look for pairs of terms where one can be
    expressed as a linear combination of the other plus a third term,
    effectively reducing rank by 1.
    
    Also try: can we replace 3 terms with 2?
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 2:
        return U, V, W
    
    # For each pair (r1, r2), check if their sum is rank-1
    # i.e., U1⊗V1⊗W1 + U2⊗V2⊗W2 = U3⊗V3⊗W3
    # This requires the 4x22x22 tensor to have rank 1.
    
    # A faster check: if U1 and U2 are linearly independent, and V1 and V2
    # are linearly independent, then the sum is generally not rank-1.
    # But if one of them is proportional, we can merge.
    
    # Try triple merging: can terms r1, r2, r3 be replaced by 2 terms?
    # This is harder. Let's try a different approach.
    
    # Look for terms that share the same U vector (exactly, not just proportional)
    # and see if their V⊗W parts can be combined.
    from collections import defaultdict
    
    u_groups = defaultdict(list)
    for r in range(current_rank):
        U_r = U[:, r]
        # Use a hashable representation
        key = tuple(np.round(U_r, 8))
        u_groups[key].append(r)
    
    for key, indices in u_groups.items():
        if len(indices) < 2:
            continue
        # All these terms have the same U vector
        # Check if any pair has proportional V vectors
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                r1, r2 = indices[i], indices[j]
                V1 = V[:, r1]
                V2 = V[:, r2]
                if np.linalg.norm(V1) < 1e-10 or np.linalg.norm(V2) < 1e-10:
                    continue
                ratio_v = np.dot(V1, V2) / np.dot(V1, V1)
                if abs(np.dot(V1, V2) - ratio_v * np.dot(V1, V1)) < 1e-8:
                    # V vectors are proportional, merge
                    W_new = W[:, r1] + ratio_v * W[:, r2]
                    mask = [i for i in range(current_rank) if i != r1 and i != r2]
                    U_test = np.column_stack([U[:, mask], U[:, r1]])
                    V_test = np.column_stack([V[:, mask], V[:, r1]])
                    W_test = np.column_stack([W[:, mask], W_new])
                    if _verify_fast(U_test, V_test, W_test, T):
                        U, V, W = U_test, V_test, W_test
                        current_rank -= 1
    
    return U, V, W


def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit

    T = _build_target_tensor()

    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')

    # Try the subspace splitting construction (rank 38)
    try:
        U, V, W = _build_subspace_splitting_decomposition()
        if _verify(U, V, W, T):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    except Exception:
        pass

    # Try the Strassen-based construction (rank 39)
    try:
        U, V, W = _build_strassen_decomposition()
        if _verify(U, V, W, T):
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    except Exception:
        pass

    # Use the best construction found so far
    if best_U is not None:
        U, V, W = best_U, best_V, best_W

        # Try rank reduction via SVD/QR
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            try:
                U_red, V_red, W_red = _try_rank_reduction(
                    U, V, W, T, max_time=min(remaining_time, 60)
                )
                if U_red.shape[1] < U.shape[1]:
                    U, V, W = U_red, V_red, W_red
            except Exception:
                pass

        # Try cross-block reduction (look for mergeable terms)
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 5:
            try:
                U_red, V_red, W_red = _try_cross_block_reduction(
                    U, V, W, T, max_time=min(remaining_time, 30)
                )
                if U_red.shape[1] < U.shape[1]:
                    U, V, W = U_red, V_red, W_red
            except Exception:
                pass
        
        # Try aggressive merging
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 5:
            try:
                U_red, V_red, W_red = _try_aggressive_merging(
                    U, V, W, T, max_time=min(remaining_time, 45)
                )
                if U_red.shape[1] < U.shape[1]:
                    U, V, W = U_red, V_red, W_red
            except Exception:
                pass

        # Final verification before returning
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Should not reach here
    raise RuntimeError("No valid decomposition found")
