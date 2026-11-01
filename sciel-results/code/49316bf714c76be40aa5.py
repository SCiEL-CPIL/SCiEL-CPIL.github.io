"""Rank-minimal decomposition for 2x3 * 3x8 matrix multiplication via Strassen block structure."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0


def _strassen_2x2():
    """
    Return U(4,7), V(4,7), W(4,7) for 2x2 * 2x2 = 2x2 Strassen decomposition.
    Indexing: a = i*2+j, b = j*2+k, d = k*2+i.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Standard Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b22-b12)
    # m4 = a22(b11-b21)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b12+b22)
    # m7 = (a12-a22)(b11+b21)
    
    # U rows: a11=0, a12=1, a21=2, a22=3
    # V rows: b11=0, b12=1, b21=2, b22=3
    # W rows: c11=0, c12=1, c21=2, c22=3
    
    # m1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # m3
    U[0, 2] = 1
    V[3, 2] = 1; V[1, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # m4
    U[3, 3] = 1
    V[0, 3] = 1; V[2, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # m5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # m6
    U[2, 5] = 1; U[0, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # m7
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W


def _build_sub_2x3_3x2():
    """
    Rank-6 decomposition for 2x3 * 3x2 = 2x2.
    Returns U(6x6), V(6x6), W(4x6) where:
    a = i*3+j (i in {0,1}, j in {0,1,2})
    b = j*2+k (j in {0,1,2}, k in {0,1})
    d = k*2+i (k in {0,1}, i in {0,1})
    """
    U = np.zeros((6, 6))
    V = np.zeros((6, 6))
    W = np.zeros((4, 6))
    
    # m0: A[0,0] * (B[0,0] + B[0,1])
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    # m1: A[1,0] * (B[0,0] + B[0,1])
    U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # m2: A[0,1] * (B[1,0] + B[1,1])
    U[1, 2] = 1
    V[2, 2] = 1; V[3, 2] = 1
    W[0, 2] = 1; W[1, 2] = 1
    
    # m3: A[1,1] * (B[1,0] + B[1,1])
    U[4, 3] = 1
    V[2, 3] = 1; V[3, 3] = 1
    W[2, 3] = 1; W[3, 3] = 1
    
    # m4: A[0,2] * (B[2,0] + B[2,1])
    U[2, 4] = 1
    V[4, 4] = 1; V[5, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # m5: A[1,2] * (B[2,0] + B[2,1])
    U[5, 5] = 1
    V[4, 5] = 1; V[5, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    return U, V, W


def _verify_decomp(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition for n x m * m x p = p x n."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T, atol=tol, rtol=0)


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _construct_rank_28():
    """
    Construct the rank-28 decomposition using 4 blocks of 2x3x2 (rank 6 each).
    This is the current best known approach.
    """
    n, m, p = 2, 3, 8
    U_sub, V_sub, W_sub = _build_sub_2x3_3x2()
    r_sub = U_sub.shape[1]  # 6
    p_sub = 2
    num_blocks = p // p_sub  # 4
    total_rank = num_blocks * r_sub  # 24... wait, 4*6=24, but current says 28?
    
    # Actually the current program has r_sub = 6 but total_rank = 28? Let me re-check.
    # In the current code: r_sub = U_sub.shape[1] which is 6, num_blocks = 4, total_rank = 24.
    # But the docstring says rank-28. Let me check if there's an error in the current code.
    # Actually looking at the current code more carefully, it says "Rank-28" in the docstring
    # but constructs 4*6=24. This might be a bug or I'm misreading.
    # Let me just build it correctly.
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for t in range(num_blocks):
        U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(p_sub):
                b_global = j * p + (t * p_sub + k_prime)
                b_local = j * p_sub + k_prime
                V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
        
        for k_prime in range(p_sub):
            for i in range(n):
                d_global = (t * p_sub + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
    
    return U_full, V_full, W_full


def _construct_rank_44_strassen():
    """
    Construct a rank-44 decomposition using Strassen for 2x2x8 + rank-16 for 2x1x8.
    
    Split A = [A_left | A_right] where A_left is 2x2 (cols 0,1), A_right is 2x1 (col 2).
    Split B = [B_top; B_bot] where B_top is 2x8 (rows 0,1), B_bot is 1x8 (row 2).
    
    C = A_left * B_top + A_right * B_bot
    
    For A_left * B_top (2x2 * 2x8 = 2x8):
    Split B_top's 8 columns into 4 groups of 2. Each group is 2x2*2x2, use Strassen (rank 7).
    Total: 4*7 = 28.
    
    For A_right * B_bot (2x1 * 1x8 = 2x8):
    C[i,k] = A_right[i] * B_bot[k]
    This is rank 16 (one term per output entry).
    
    Total rank: 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    
    # Part 1: A_left (2x2) * B_top (2x8) using Strassen
    # A_left indices: a = i*2+j for i in {0,1}, j in {0,1} -> a in {0,1,2,3}
    # B_top indices: b = j*8+k for j in {0,1}, k in {0..7} -> b in {0..7, 8..15}
    # C_part1 indices: d = k*2+i for k in {0..7}, i in {0,1} -> d in {0..15}
    
    U_strassen, V_strassen, W_strassen = _strassen_2x2()
    r_str = 7
    
    # For each of 4 column groups (2 columns each)
    U_p1 = np.zeros((n*m, 4 * r_str))  # but only first 4 rows of U are used (A_left)
    V_p1 = np.zeros((m*p, 4 * r_str))
    W_p1 = np.zeros((p*n, 4 * r_str))
    
    for t in range(4):
        col_offset = t * r_str
        # U: same Strassen pattern for A_left (rows 0-3 of global A)
        U_p1[0:4, col_offset:col_offset+r_str] = U_strassen
        
        # V: B_top is rows 0-1 of B, columns 2t, 2t+1
        # Global B index: b = j*8 + k, where j in {0,1} for B_top
        # Local Strassen B index: b_local = j_local*2 + k_local, j_local in {0,1}, k_local in {0,1}
        for j_local in range(2):
            for k_local in range(2):
                b_global = j_local * p + (t * 2 + k_local)
                b_local = j_local * 2 + k_local
                V_p1[b_global, col_offset:col_offset+r_str] = V_strassen[b_local]
        
        # W: C output, columns 2t, 2t+1
        # Global C index: d = k*n + i = k*2 + i
        # Local Strassen C index: d_local = k_local*2 + i
        for k_local in range(2):
            for i in range(2):
                d_global = (t * 2 + k_local) * n + i
                d_local = k_local * 2 + i
                W_p1[d_global, col_offset:col_offset+r_str] = W_strassen[d_local]
    
    # Part 2: A_right (2x1) * B_bot (1x8)
    # A_right: column 2 of A, so a = i*3 + 2, i in {0,1} -> a in {2, 5}
    # B_bot: row 2 of B, so b = 2*8 + k = 16 + k, k in {0..7}
    # C: d = k*2 + i, k in {0..7}, i in {0,1}
    
    r_p2 = 16
    U_p2 = np.zeros((n*m, r_p2))
    V_p2 = np.zeros((m*p, r_p2))
    W_p2 = np.zeros((p*n, r_p2))
    
    col = 0
    for i in range(2):
        for k in range(8):
            a_idx = i * m + 2  # i*3 + 2
            b_idx = 2 * p + k  # 2*8 + k
            d_idx = k * n + i  # k*2 + i
            U_p2[a_idx, col] = 1
            V_p2[b_idx, col] = 1
            W_p2[d_idx, col] = 1
            col += 1
    
    # Combine
    total_rank = 4 * r_str + r_p2  # 28 + 16 = 44
    U_full = np.hstack([U_p1, U_p2])
    V_full = np.hstack([V_p1, V_p2])
    W_full = np.hstack([W_p1, W_p2])
    
    return U_full, V_full, W_full


def _construct_rank_44_hybrid():
    """
    Alternative rank-44: Use the 2x3x2 sub-decomposition (rank 6) for 4 blocks = 24,
    but that gives rank 24 which should be better. Wait, let me re-examine.
    
    Actually, the 2x3x2 rank-6 decomposition IS valid and gives 4*6=24 total.
    Let me verify this is correct. If so, rank 24 is better than 44.
    
    But the current program claims rank 28. Let me check if the rank-6 sub-decomp is valid.
    """
    n, m, p = 2, 3, 8
    U_sub, V_sub, W_sub = _build_sub_2x3_3x2()
    
    # Verify sub-decomposition
    if not _verify_decomp(U_sub, V_sub, W_sub, 2, 3, 2):
        return None
    
    r_sub = U_sub.shape[1]  # 6
    p_sub = 2
    num_blocks = p // p_sub  # 4
    total_rank = num_blocks * r_sub  # 24
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for t in range(num_blocks):
        U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
        
        for j in range(m):
            for k_prime in range(p_sub):
                b_global = j * p + (t * p_sub + k_prime)
                b_local = j * p_sub + k_prime
                V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
        
        for k_prime in range(p_sub):
            for i in range(n):
                d_global = (t * p_sub + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
    
    # Verify full decomposition
    if not _verify_decomp(U_full, V_full, W_full, n, m, p):
        return None
    
    return U_full, V_full, W_full


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Try rank-24 construction first (4 blocks of rank-6 2x3x2)
    result = _construct_rank_44_hybrid()
    if result is not None:
        U, V, W = result
        rank = U.shape[1]
        # Final verification
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Strassen-based rank-44
    U, V, W = _construct_rank_44_strassen()
    if _verify_decomp(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
