"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_strassen_decomposition():
    """
    Build the rank-84 Strassen-based block decomposition for 4x4 * 4x6.
    Uses 2x2 Strassen for each pair of 2x2 blocks in the sum over j.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]
                
                # A_ij element indices (row-major: row*m + col)
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                # B_jk element indices (row-major: row*p + col)
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index convention: W[col*n + row, r]
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                # Strassen 7 multiplications for A_ij * B_jk:
                
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p = 96."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute from decomposition using batched outer products
    # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    return np.allclose(T_decomp, T_expected, atol=tol)

def _build_rank80_decomposition():
    """
    Attempt a rank-80 decomposition by exploiting structure in the 4x4 x 4x6 case.
    
    Idea: For the 4x4 x 4x4 part (first 4 columns of B), use a rank-49 algorithm
    if available, otherwise use Strassen (rank 56 for 4x4x4x4).
    For the 4x4 x 4x2 part (last 2 columns), use schoolbook (rank 32).
    
    Total with Strassen: 56 + 32 = 88 (worse than 84).
    Total with rank-49: 49 + 32 = 81 (better!).
    
    However, implementing rank-49 for 4x4x4x4 is complex. Let's try a different
    approach: can we share linear combinations between the two parts?
    
    Actually, let's try: use Strassen for all 12 block pairs but see if we can
    merge some of the 7 multiplications across different (i,k) pairs.
    
    For fixed i and j, we have 3 k-values. The Strassen M1-M7 for each (i,j,k)
    use different B_jk blocks. But the A_ij block is the same for all k.
    
    Specifically, for fixed (i,j):
    - M1 uses (a+d) from A_ij and (e+h) from B_jk
    - M2 uses (c+d) from A_ij and e from B_jk
    - etc.
    
    The A-side linear combinations are reused across k! So for fixed (i,j),
    we compute 7 A-side combos once, but need 7*B-side combos per k.
    
    This doesn't directly reduce rank because each (a_combo, b_combo) pair
    is still a separate multiplication.
    
    Let me try a different idea: use a 2x2 x 2x6 algorithm.
    A is 2x2, B is 2x6, C is 2x6.
    B can be split into three 2x2 blocks: B = [B1 | B2 | B3]
    C = [C1 | C2 | C3] where Ck = A * Bk
    
    For each Ck = A * Bk, we use Strassen (rank 7).
    Total: 3 * 7 = 21 per (i_block, j_block) pair.
    We have 2 i_blocks and 2 j_blocks, so 4 * 21 = 84. Same as before.
    
    No improvement with this approach.
    
    Return None to indicate no improvement found.
    """
    return None

def _try_alternative_strassen(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using a modified Strassen approach.
    For 4x4 * 4x6, we can potentially use a 3x3 block structure with 2x2 Strassen
    or explore other algebraic patterns.
    
    This function explores a few structured alternatives to see if we can beat rank 84.
    """
    # Alternative: use 3x3 block partitioning where possible
    # For 4x4 * 4x6, we can partition as:
    # A: 4x4 -> 2x2 blocks of 2x2
    # B: 4x6 -> 2x3 blocks of 2x2
    # C: 4x6 -> 2x3 blocks of 2x2
    # This is exactly what the current approach does, giving rank 84.
    
    # Try: Can we share some Strassen factors across different (i,k) pairs?
    # The key insight: for fixed j, the Strassen factors for different (i,k) pairs
    # might share structure. But in standard Strassen, they don't because
    # the linear combinations are specific to each block pair.
    
    # Another idea: use a 4x4 * 4x4 decomposition (rank 49 or 64) and extend to 4x6
    # 4x4 * 4x4 has known rank <= 49 (Coppersmith-Winograd for small cases)
    # But we need 4x4 * 4x6, which is 4x4 * 4x4 + 4x4 * 4x2
    # This doesn't directly help without more sophisticated analysis.
    
    # For now, return None to indicate no improvement found
    return None

def _build_hybrid_strassen(n, m, p):
    """
    Try a hybrid approach: use Strassen for the 4x4 * 4x4 part (rank 49 via known
    small-matrix results or rank 64 via 2x2 Strassen blocks) and schoolbook for
    the 4x4 * 4x2 part.
    
    For 4x4 * 4x4: 2x2 block Strassen gives 7 * 4 = 28 rank per 2x2 block pair,
    but we have 4 block pairs (i,k in {0,1}^2) each with 2 j-iterations, so 7*4*2=56.
    Actually for 4x4*4x4: 2x2 blocks, 4 (i,k) pairs, 2 j values each = 8 block multiplications
    at 7 rank each = 56.
    
    For 4x4 * 4x2: schoolbook = 4*4*2 = 32.
    
    Total: 56 + 32 = 88. That's worse than 84.
    
    Alternative: For 4x4*4x4, can we do better than 56?
    Known: 4x4*4x4 has rank <= 49 (Coppersmith-Winograd style).
    But implementing that is complex. Let's try a different split.
    
    What about 4x4 * 4x6 as (4x2 * 4x6) + (4x2 * 4x6)?
    Each 4x2 * 4x6: 2x2 blocks for A (2 blocks), 2x3 blocks for B (3 blocks), 
    2x3 blocks for C (3 blocks). For each (i_block, k_block) pair with 2 j values:
    2 * 3 * 2 * 7 = 84 per half. Total 168. Worse.
    
    The current 84 seems hard to beat with simple block methods.
    Return None.
    """
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit (evaluator kills at 150s)
    
    n, m, p = DIMENSIONS
    
    # Start with the Strassen-based rank-84 decomposition
    strassen_result = _build_strassen_decomposition()
    
    # Verify the Strassen decomposition (fast einsum-based check)
    if time.time() < deadline - 5:
        if _verify_decomposition(strassen_result['U'], strassen_result['V'], strassen_result['W'], n, m, p):
            # Valid rank-84 decomposition
            best_rank = 84
            best_result = strassen_result
        else:
            # Fallback to schoolbook
            schoolbook = _build_schoolbook()
            best_rank = 96
            best_result = schoolbook
    else:
        # Not enough time to verify, assume Strassen is valid (it's well-known)
        best_rank = 84
        best_result = strassen_result
    
    # Try to find a better decomposition using alternative approaches
    if time.time() < deadline - 10:
        alternative = _try_alternative_strassen(n, m, p, deadline)
        if alternative is not None:
            alt_rank = len(alternative['U'][0])
            if alt_rank < best_rank:
                if time.time() < deadline - 3:
                    if _verify_decomposition(alternative['U'], alternative['V'], alternative['W'], n, m, p):
                        best_rank = alt_rank
                        best_result = alternative
    
    # Also try the hybrid approach
    if time.time() < deadline - 10:
        hybrid = _build_hybrid_strassen(n, m, p)
        if hybrid is not None:
            hyd_rank = len(hybrid['U'][0])
            if hyd_rank < best_rank:
                if time.time() < deadline - 3:
                    if _verify_decomposition(hybrid['U'], hybrid['V'], hybrid['W'], n, m, p):
                        best_rank = hyd_rank
                        best_result = hybrid
    
    # Try the rank-80 approach
    if time.time() < deadline - 10:
        r80 = _build_rank80_decomposition()
        if r80 is not None:
            r80_rank = len(r80['U'][0])
            if r80_rank < best_rank:
                if time.time() < deadline - 3:
                    if _verify_decomposition(r80['U'], r80['V'], r80['W'], n, m, p):
                        best_rank = r80_rank
                        best_result = r80
    
    # Final fallback: if we somehow didn't get a valid result, use schoolbook
    if best_rank > 96:
        best_result = _build_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
