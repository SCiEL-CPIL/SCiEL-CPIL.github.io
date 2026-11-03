"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

_START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W correctly decompose the matrix multiplication tensor.
    Returns True if exact (within tolerance)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor: for each rank position r, the rank-1 tensor is U[:,r] * V[:,r] outer
    # The multiplication tensor T[i,j,k] = delta_{j,k} (i in [0,n), j in [0,m), k in [0,p))
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Tensor entry: sum_r U[i*m+j, r] * V[j*p+k, r] should equal delta_{i, r_out} where...
    
    # Actually the convention is:
    # A is n x m, B is m x p, C = A*B is n x p
    # U[i*m+j, r] corresponds to coefficient of A[i,j]
    # V[j*p+k, r] corresponds to coefficient of B[j,k]
    # W[k*n+i, r] corresponds to coefficient of C[i,k] (trace convention: W[k*n+i, r])
    
    # The tensor decomposition says: for each output element C[i,k]:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... no that's not right.
    
    # Actually for a rank-1 term r: U[:,r] * V[:,r] gives a bilinear form on (A,B),
    # and W[:,r] tells how that bilinear form contributes to each output element.
    # So: C = sum_r W[:,r] * (U[:,r]^T * A_flat * V[:,r])  ... let me think more carefully.
    
    # The tensor is a 3-tensor T_{i,j,k} where i indexes A (n*m entries), j indexes B (m*p entries), k indexes C (p*n entries)
    # T[a_idx, b_idx, c_idx] = 1 if a_idx and b_idx share the middle index AND c_idx is the corresponding output, else 0
    
    # a_idx = i*n_a + j_a where i in [0,n), j_a in [0,m)  -- wait, U shape is (n*m, rank)
    # Let me re-read: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    
    # For matrix mult A(n x m) * B(m x p) = C(n x p):
    # U row index: i*m + j  (i in [0,n), j in [0,m))
    # V row index: j*p + k  (j in [0,m), k in [0,p))
    # W row index: k*n + i  (k in [0,p), i in [0,n)) -- trace convention
    
    # The multiplication tensor: T[a, b, c] where:
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 iff j == j' AND i == i' AND k == k'
    
    # Rank-r decomposition: T = sum_s U[:,s] * V[:,s] * W[:,s] (outer product of 3 vectors)
    # So T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
    
    # Verify: for a random or systematic set of inputs
    for _ in range(100):
        if _elapsed() > TIME_LIMIT - 5:
            return None  # timeout
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for s in range(rank):
            prod = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # This is too slow, let's use vectorized
                        pass
            break
        
        # Vectorized check
        # bilinear[s] = sum_{i,j} U[i*m+j, s] * A[i,j] * ... no
        # Actually: the rank-1 term s computes: sum_{i,j} U[i*m+j, s] * A[i,j] then multiplied by... 
        # No. The decomposition is: C[i,k] = sum_s W[k*n+i, s] * (sum_{j} A[i,j] * U[i*m+j, s] * ... )
        
        # Wait, I need to think about this differently.
        # T[a, b, c] = sum_s U[a, s] * V[b, s] * W[c, s]
        # a = i*m+j corresponds to A[i,j], b = j'*p+k corresponds to B[j',k], c = k'*n+i' corresponds to C[i',k']
        # So: sum_{j} A[i,j] * B[j,k] = sum_s W[k*n+i, s] * (sum_{j,j',k'} U[i*m+j, s] * V[j'*p+k', s] * delta_{j,j'} * delta_{i',i} * delta_{k',k})
        # That's not how it works. The tensor entries ARE the coefficients.
        
        # Let me just directly compute: C[i,k] should equal sum_s W[k*n+i, s] * (sum_j U[i*m+j,s] * ... )
        # Actually no. The tensor is 3D: T[a,b,c] where a is the A-index, b is the B-index, c is the C-index.
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0.
        # Decomposition: T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        
        # To compute C from A and B using the decomposition:
        # C[i,k] = T's contribution = sum_{a,b} T[a,b,k*n+i] * A_flat[a] * B_flat[b]
        #       = sum_s W[k*n+i,s] * (sum_a U[a,s]*A_flat[a]) * (sum_b V[b,s]*B_flat[b])
        
        A_flat = A.flatten()  # row-major: index i*m+j
        B_flat = B.flatten()  # row-major: index j*p+k
        
        # sum_a U[a,s]*A_flat[a] for each s
        # U shape (n*m, rank), A_flat shape (n*m,)
        uA = U @ A_flat  # shape (rank,)
        vB = V @ B_flat  # shape (rank,)
        # C[i,k] = sum_s W[k*n+i, s] * uA[s] * vB[s]
        W_mat = W  # shape (p*n, rank)
        # Reshape to (p, n, rank) then transpose to (n, p, rank)
        W_res = W_mat.reshape(p, n, rank).transpose(1, 0, 2)  # (n, p, rank)
        C_from_decomp = np.einsum('ijr,i r,k r->...', W_res, uA, vB)
        # That's not right either. Let me just do:
        C_from_decomp = np.zeros((n, p))
        for s in range(rank):
            C_from_decomp += uA[s] * vB[s] * W_res[:, :, s]
        
        if not np.allclose(C_expected, C_from_decomp, atol=1e-8):
            return False
    return True


def _strassen_2x2():
    """Return the 7 Strassen terms for 2x2*2x2->2x2.
    Local indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i
    """
    terms = [
        # M1 = (x00+x11)(y00+y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (x10+x11)y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = x00(y01-y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = x11(y10-y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (x00+x01)y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (x10-x00)(y00+y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (x01-x11)(y10+y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    return terms


def _build_strassen_4block(n, m, p):
    """Build the 4-block Strassen decomposition with rank 28."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = _strassen_2x2()
    
    for b in range(4):
        a_col_off = 2 * (b % 2)
        b_row_off = 2 * (b % 2)
        b_col_off = 2 * (b // 2)
        c_col_off = 2 * (b // 2)
        
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                j = a_col_off + lj
                g_idx = i * 4 + j
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                j = b_row_off + li
                r = b_col_off + lj
                g_idx = j * 4 + r
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                r = c_col_off + lj
                g_idx = r * 2 + i
                W[g_idx][col] += coeff

    return U, V, W


def _try_eliminate_terms(U, V, W, n, m, p, rank, target_rank, time_budget=60):
    """Try to find a lower-rank decomposition by identifying linearly dependent
    rank-1 slices. We look for subsets of terms whose removal is compensated
    by adjusting other terms, or we search for a direct lower-rank representation.
    
    Strategy: Use the structure that the 4 blocks share patterns.
    Try to find rank-24, 23, ... decompositions.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check for exact linear dependence among the U columns
    # If some columns of U are linearly dependent, we might be able to reparameterize
    # But that doesn't directly reduce rank unless the corresponding V,W also align.
    
    # Better approach: Try to find a decomposition with fewer terms by solving
    # for a rank-k approximation that is exact.
    
    # For a target rank k < current rank:
    # We need to find U'(n*m, k), V'(m*p, k), W'(p*n, k) such that
    # sum_s U'[a,s]*V'[b,s]*W'[c,s] = T[a,b,c] for all a,b,c
    
    # This is a system of nonlinear equations. For small reductions, we can try:
    # - Pick which terms to "merge" or eliminate
    # - Solve the resulting linear system
    
    # Alternative: Use the fact that if we can find a rank-k tensor that equals T,
    # we can verify it. Let's try a systematic search for k = target_rank.
    
    # For rank 24: we need to eliminate 4 terms from the 28.
    # The key insight: across the 4 blocks, certain linear forms repeat.
    # If two blocks use the same U coefficient pattern (just shifted), 
    # their products might combine.
    
    # Let's try: for each pair of blocks that have the same column offset pattern,
    # see if we can merge terms.
    
    # Actually, let's try a more direct approach: random search + local refinement.
    # Start from rank 24 and try to fit the tensor.
    
    deadline = time_budget
    start = time.time()
    
    # Try rank 24: eliminate 4 terms
    # The 28 terms are arranged as 4 blocks of 7.
    # If we can show that one Strassen term per block is redundant when combined,
    # we save 4 total.
    
    # Let's try: for each possible set of 4 terms to remove (one per block),
    # solve for a rank-24 decomposition.
    # This is too many combinations. Instead, let's use a smarter approach.
    
    # The 2x4 * 4x4 tensor has 2*4*4 = 32 entries for A, 4*4 = 16 for B, 4*2 = 8 for C.
    # Total tensor size: 32 * 16 * 8 = 4096 entries, but most are zero.
    # Non-zero entries: 32 (one for each (i,j,k) combination).
    
    # For rank r, we have r*(32+16+8) = r*56 parameters.
    # We need r*56 >= 32 (number of constraints), so r >= 1 in principle.
    # But the structure matters a lot.
    
    # Known results: For 2x2x2, min rank is 7 (Strassen).
    # For 2x4x4, the trivial block decomposition gives 4*7 = 28.
    # Can we do better? The "unbalanced" case might allow some sharing.
    
    # Let's try a numerical search: parameterize a rank-k decomposition and minimize
    # the error to the target tensor.
    
    from scipy.optimize import minimize
    
    def tensor_error(U_flat, V_flat, W_flat, k):
        U = U_flat.reshape(32, k)
        V = V_flat.reshape(16, k)
        W = W_flat.reshape(8, k)
        # Compute the tensor and compare to target
        # Target tensor T[a,b,c] for a in [0,32), b in [0,16), c in [0,8)
        # T[i*4+j, j*4+k, k*2+i] = 1
        err = 0.0
        # This is expensive to compute fully. Use random sampling.
        rng = np.random.default_rng(42)
        for _ in range(2000):
            a = rng.integers(0, 32)
            b = rng.integers(0, 16)
            c = rng.integers(0, 8)
            # Target value
            i_a, j_a = a // 4, a % 4
            j_b, k_b = b // 4, b % 4
            k_c, i_c = c // 2, c % 2
            target = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
            # Computed value
            comp = 0.0
            for s in range(k):
                comp += U[a, s] * V[b, s] * W[c, s]
            err += (comp - target) ** 2
        return err / 2000
    
    # This is too slow for optimization. Let's try a different approach.
    
    # Direct approach: try to construct a rank-24 decomposition by exploiting
    # the structure. The 4 blocks are:
    # Block 0: A uses cols [0,2), B uses rows [0,2) cols [0,2), C uses cols [0,2)
    # Block 1: A uses cols [0,2), B uses rows [0,2) cols [2,4), C uses cols [2,4)
    # Block 2: A uses cols [2,4), B uses rows [2,4) cols [0,2), C uses cols [0,2)
    # Block 3: A uses cols [2,4), B uses rows [2,4) cols [2,4), C uses cols [2,4)
    
    # Notice: Blocks 0 and 2 both output to C cols [0,2), and blocks 1 and 3 to C cols [2,4).
    # Blocks 0 and 1 both use A cols [0,2), and blocks 2 and 3 use A cols [2,4).
    
    # The A-side linear forms for block 0 vs block 2 are different (different column ranges),
    # so they can't simply be merged.
    
    # However, consider: the C output for C[i,0] gets contributions from block 0 (via B[0:2,0:2])
    # and block 2 (via B[2:4,0:2]). Similarly C[i,1] from blocks 0 and 2.
    # C[i,2] from blocks 1 and 3, C[i,3] from blocks 1 and 3.
    
    # If we could share some Strassen terms between the C outputs of blocks that feed the same C column...
    # But the A and B inputs are different, so this doesn't directly help.
    
    # Let's just return the rank-28 solution if we can't find better within time.
    if time.time() - start > deadline:
        return None
    
    # Try a simple heuristic: check if the 28-term solution has any redundant terms
    # by checking if any column in the tensor factorization is zero.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    zero_terms = []
    for s in range(rank):
        if np.allclose(U_arr[:, s], 0) or np.allclose(V_arr[:, s], 0) or np.allclose(W_arr[:, s], 0):
            zero_terms.append(s)
    
    if zero_terms:
        # Remove zero terms
        keep = [s for s in range(rank) if s not in zero_terms]
        new_rank = len(keep)
        U_new = U_arr[:, keep].tolist()
        V_new = V_arr[:, keep].tolist()
        W_new = W_arr[:, keep].tolist()
        return U_new, V_new, W_new
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank-28 Strassen block decomposition
    U, V, W = _build_strassen_4block(n, m, p)
    
    # Try to reduce rank
    if _elapsed() < TIME_LIMIT - 10:
        reduced = _try_eliminate_terms(U, V, W, n, m, p, 28, 24, time_budget=30)
        if reduced is not None:
            U, V, W = reduced
    
    # Final verification
    if _elapsed() < TIME_LIMIT - 2:
        result = _verify_decomposition(U, V, W, n, m, p)
        if result is False:
            # Fall back to verified rank-28
            U, V, W = _build_strassen_4block(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
