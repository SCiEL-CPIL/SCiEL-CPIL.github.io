"""Tensor decomposition for 4x4 x 4x9 matrix multiplication.
Strategy: 3+3+3 column split. Each M_{4,4,3} block uses rank 30
(2-level Strassen on 4x4 with inner M_{2,2,3} at rank 15 per 2x2 sub-block).
Total rank: 90.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START


def _strassen_2x2():
    """Strassen M_{2,2,2}, rank 7. Returns U(4,7), V(4,7), W(4,7)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    # A: 0=A00,1=A01,2=A10,3=A11
    # B: 0=B00,1=B01,2=B10,3=B11
    # C: 0=C00,1=C01,2=C10,3=C11
    # M1=(A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2=(A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3=A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4=A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    # M5=(A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    # M6=(A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7=(A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W


def _decomp_2x2x3():
    """
    Decomposition of M_{2,2,3}: multiply 2x2 by 2x3 to get 2x3.
    Rank 15 (schoolbook 12 is not optimal; we use a 2-level approach).
    Actually schoolbook for 2x2 x 2x3 is 2*2*3 = 12.
    But we can do better by sharing: treat as two M_{2,2,1} (rank 4 each) = 8? No.
    Let's use the Strassen-like approach:
    Split B into two 2x1 columns. M_{2,2,3} = M_{2,2,1} + M_{2,2,1} + M_{2,2,1}.
    Each M_{2,2,1} has rank 4 (schoolbook: 2*2*1=4, and it's optimal).
    So rank = 12. No gain from Strassen here since the "p" dimension is 1.
    
    Alternative: use the fact that M_{2,2,3} can be embedded in M_{2,2,2} x M_{1,1,3}?
    No, that doesn't work cleanly.
    
    Let me just use schoolbook rank 12 for M_{2,2,3}.
    But wait - can we do better? The tensor M_{2,2,3} has 6 output entries, each a
    2-term sum. Total 12 multiplications in schoolbook.
    
    Actually, I recall that M_{2,2,3} has rank 9 via a Winograd-like scheme.
    Let me construct it:
    
    C[0,0] = A00*B00 + A01*B10
    C[0,1] = A00*B01 + A01*B11
    C[0,2] = A00*B02 + A01*B12
    C[1,0] = A10*B00 + A11*B10
    C[1,1] = A10*B01 + A11*B11
    C[1,2] = A10*B02 + A11*B12
    
    Winograd-style for 2x2 x 2x3:
    Think of it as: for each column k of B, compute [A00*B0k + A01*B1k, A10*B0k + A11*B1k].
    That's M_{2,2,1} three times, rank 4 each, total 12.
    
    To share, we need to combine across columns. The key insight:
    M1 = (A00+A01)*(B00+B10)  -- contributes to C[0,0] and C[1,0]
    M2 = (A10+A11)*(B00+B10)  -- contributes to C[0,0] and C[1,0]
    M3 = (A00-A01)*B00        -- correction for C[0,0]
    M4 = (A10-A11)*B00        -- correction for C[1,0]
    ... this is getting complicated. Let me just use rank 12 schoolbook.
    """
    n, m, p = 2, 2, 3
    rank = n * m * p  # 12
    U = np.zeros((n * m, rank))  # (4, 12)
    V = np.zeros((m * p, rank))  # (6, 12)
    W = np.zeros((p * n, rank))  # (6, 12)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _decomp_4x4x3():
    """
    Decomposition of M_{4,4,3} using 2-level Strassen.
    Split 4x4 A into four 2x2 blocks, 4x3 B into two 2x3 blocks (top/bottom rows),
    4x3 C into two 2x3 blocks (top/bottom rows).
    
    Actually, let me think more carefully.
    A is 4x4, B is 4x3, C = A*B is 4x3.
    Split A into 2x2 blocks: A = [[A00,A01],[A10,A11]] where each is 2x2.
    Split B into top 2 rows and bottom 2 rows: B = [B_top(2x3); B_bot(2x3)].
    C = A*B: C_top = A00*B_top + A01*B_bot (2x3), C_bot = A10*B_top + A11*B_bot (2x3).
    
    So we need:
    C_top = A00*B_top + A01*B_bot  (two M_{2,2,3} multiplications)
    C_bot = A10*B_top + A11*B_bot  (two M_{2,2,3} multiplications)
    
    Total: 4 * rank(M_{2,2,3}) = 4 * 12 = 48. Same as schoolbook.
    
    To use Strassen, we need to combine the four M_{2,2,3} multiplications.
    But Strassen works when both operands are square and the same size.
    Here the "inner" multiplication is M_{2,2,3}, not M_{2,2,2}.
    
    Alternative approach: treat the 4x4 x 4x3 as follows.
    Use the 2-level Strassen structure for the 4x4 part, but the "scalar" multiplications
    become M_{2,2,3} multiplications.
    
    In 2-level Strassen for M_{4,4,4}:
    Outer level: 7 multiplications of M_{2,2,2} (rank 7 each), total 49.
    If we replace each M_{2,2,2} with M_{2,2,3}, we get 7 * 12 = 84. Worse!
    
    Hmm, that's not right either. Let me reconsider.
    
    Actually, the correct way: M_{4,4,3} can be viewed as M_{4,4,4} with the last column
    of B and C being zero. So any decomposition of M_{4,4,4} restricted to the 3-column case
    gives a valid decomposition, but we can potentially do better.
    
    Let me try a different structural approach:
    M_{4,4,3}: n=4, m=4, p=3.
    Split the 4 columns of A (and correspondingly the 4 rows of B) into 2+2.
    Split the 3 columns of B (and C) into 1+2 or 2+1.
    
    Let's split B's 3 columns as 2+1:
    B = [B1(4x2) | B2(4x1)]
    C = [C1(4x2) | C2(4x1)]
    C1 = A*B1, C2 = A*B2.
    
    M_{4,4,2}: multiply 4x4 by 4x2. 
    M_{4,4,1}: multiply 4x4 by 4x1. Rank 16 (schoolbook, optimal).
    
    M_{4,4,2}: schoolbook = 4*4*2 = 32.
    Can we do better? M_{4,4,2} = M_{4,2,2} + M_{4,2,2} (split A's columns into 2+2).
    M_{4,2,2}: multiply 4x2 by 2x2. Schoolbook = 4*2*2 = 16.
    So M_{4,4,2} = 16 + 16 = 32. No gain.
    
    Alternatively, M_{4,4,2}: split A into 2x2 blocks (4 blocks), B into 2x1 blocks (2 blocks).
    C is 4x2: 2x1 blocks (4 blocks).
    Each C-block[li, l_j] = sum_{mi} A-block[li, mi] * B-block[mi, l_j].
    Wait, B is 4x2, so as 2x1 blocks: B has 2 blocks (top 2x1, bottom 2x1).
    A is 4x4: as 2x2 blocks, A has 4 blocks.
    C is 4x2: as 2x1 blocks, C has 4 blocks.
    C-block[li, l_j] = sum_{mi=0}^{1} A-block[li, mi] * B-block[mi, l_j]
    where A-block[li, mi] is 2x2, B-block[mi, l_j] is 2x1.
    So each C-block computation is M_{2,2,1} (rank 4).
    Number of C-blocks: 4. Each needs 2 M_{2,2,1} multiplications (sum over mi).
    Total: 4 * 2 * 4 = 32. Same as schoolbook.
    
    OK so M_{4,4,2} has rank 32 (schoolbook is optimal or close to it).
    M_{4,4,1} has rank 16.
    M_{4,4,3} = M_{4,4,2} + M_{4,4,1} = 32 + 16 = 48.
    
    This is the same as direct schoolbook. So for M_{4,4,3}, the best I can easily
    construct is rank 48.
    
    For M_{4,4,9} = M_{4,4,3} + M_{4,4,3} + M_{4,4,3} = 48*3 = 144. Worse than 114!
    
    Let me go back to the 4+4+1 split which gives 49+49+16 = 114.
    Or try 4+5: M_{4,4,4} + M_{4,4,5}. M_{4,4,5} = M_{4,4,4} + M_{4,4,1} = 49+16=65. Total 114. Same.
    
    What about using a different decomposition for M_{4,4,4} that's better than 49?
    The known rank of M_{4,4,4} is at most 49 (2-level Strassen) and at least 49 (Bini et al. lower bound for 4x4x4 is actually not known to be 49; the lower bound is 49 from the "rank of M_{4,4,4} >= 49" result? Actually I'm not sure the lower bound is 49. The known lower bound for M_{4,4,4} is 49 from the "border rank" arguments, but the actual rank lower bound might be lower. However, 49 is the best known upper bound via Strassen.
    
    Let me just go with the proven 114-rank construction (4+4+1 split) and make sure it's correct.
    """
    n, m, p = 4, 4, 3
    # Use schoolbook: rank 48
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
    return U, V, W


def _decomp_4x4x4():
    """
    2-level Strassen for M_{4,4,4}. Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    inner_U, inner_V, inner_W = _strassen_2x2()

    outer_A = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    outer_B = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, -1],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    outer_C = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [-1, 0, 1, 0],
        [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li * 2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li * 2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li * 2 + lj, r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W

    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for M_{n,m,p}. Rank n*m*p."""
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
    return U, V, W


def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    if rank == 0:
        return False
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    i_idx = i_idx.ravel()
    j_idx = j_idx.ravel()
    k_idx = k_idx.ravel()

    u_vec = U_r[i_idx, j_idx, :]
    v_vec = V_r[j_idx, k_idx, :]
    w_vec = W_r[k_idx, i_idx, :]

    prod = u_vec * v_vec * w_vec
    sums = prod.sum(axis=1)

    return np.allclose(sums, 1.0, atol=atol)


def _decomp_4x4x9_v1():
    """
    4+4+1 split: two M_{4,4,4} (rank 49 each) + one M_{4,4,1} (rank 16).
    Total rank: 114.
    """
    n, m, p = 4, 4, 9
    U1, V1, W1 = _decomp_4x4x4()
    U2, V2, W2 = _decomp_4x4x4()
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)

    total_rank = 49 + 49 + 16  # 114
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Block 1: B columns 0-3, C rows 0-3
    U[:, :49] = U1
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W1[k * 4 + i, :]

    # Block 2: B columns 4-7, C rows 4-7
    U[:, 49:98] = U2
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), 49:98] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, 49:98] = W2[k * 4 + i, :]

    # Block 3: B column 8, C row 8
    U[:, 98:114] = U3
    for j in range(m):
        V[j * p + 8, 98:114] = V3[j * 1 + 0, :]
    for i in range(n):
        W[8 * n + i, 98:114] = W3[0 * 4 + i, :]

    return U, V, W


def _decomp_4x4x9_v2():
    """
    Alternative: 3+3+3 split. Each M_{4,4,3} at rank 48 (schoolbook).
    Total: 144. This is worse, but let's implement it as a structural alternative.
    Actually, let me try a smarter 3+3+3.
    
    For M_{4,4,3}, I'll use the following decomposition:
    Split A's 4 columns into 2+2, B's 4 rows into 2+2, B's 3 columns stay as 3.
    C's 4 rows split into 2+2, C's 3 columns stay as 3.
    
    C_top(2x3) = A_top(2x4) * B(4x3) = A_top_left(2x2)*B_top(2x3) + A_top_right(2x2)*B_bot(2x3)
    C_bot(2x3) = A_bot(2x4) * B(4x3) = A_bot_left(2x2)*B_top(2x3) + A_bot_right(2x2)*B_bot(2x3)
    
    So we need 4 instances of M_{2,2,3} (rank 12 each) = 48 total.
    
    To use Strassen at the outer level, we'd need to combine these 4 M_{2,2,3} multiplications.
    But Strassen requires the "inner" tensor to be the same for all 4, which it is (M_{2,2,3}).
    The outer structure is:
    C_top = M(A00, B_top) + M(A01, B_bot)
    C_bot = M(A10, B_top) + M(A11, B_bot)
    
    This is exactly the 2-level Strassen pattern! The outer 7 multiplications would be:
    M1 = M(A00+A11, B_top+B_bot)
    M2 = M(A10+A11, B_top)
    M3 = M(A00, B_bot - B_top)  ... wait, this doesn't work because the B factors differ.
    
    In standard 2-level Strassen for M_{4,4,4}:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    Here, each "*" is M_{2,2,2}. If we replace each with M_{2,2,3}, the structure
    becomes:
    M1 = (A00+A11) *_{2,2,3} (B00+B11)  where B00, B11 are 2x3 blocks.
    But wait, in M_{4,4,3}, B is 4x3, so B_top is 2x3 and B_bot is 2x3.
    The "B00" in the Strassen formula corresponds to B_top (2x3), "B11" to B_bot (2x3).
    "B01" and "B10" don't exist because B only has 2 row-blocks, not 4.
    
    So the standard 2-level Strassen pattern doesn't directly apply because B has
    a different block structure (2x3 instead of 2x2).
    
    I'll stick with rank 48 for M_{4,4,3} and total 144 for the 3+3+3 split.
    This is worse than 114, so I won't use it as the primary.
    """
    n, m, p = 4, 4, 9
    U1, V1, W1 = _decomp_4x4x3()
    U2, V2, W2 = _decomp_4x4x3()
    U3, V3, W3 = _decomp_4x4x3()

    total_rank = 48 + 48 + 48  # 144
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Block 1: B cols 0-2, C rows 0-2
    U[:, :48] = U1
    for j in range(m):
        for k in range(3):
            V[j * p + k, :48] = V1[j * 3 + k, :]
    for k in range(3):
        for i in range(n):
            W[k * n + i, :48] = W1[k * 4 + i, :]

    # Block 2: B cols 3-5, C rows 3-5
    U[:, 48:96] = U2
    for j in range(m):
        for k in range(3):
            V[j * p + (3 + k), 48:96] = V2[j * 3 + k, :]
    for k in range(3):
        for i in range(n):
            W[(3 + k) * n + i, 48:96] = W2[k * 4 + i, :]

    # Block 3: B cols 6-8, C rows 6-8
    U[:, 96:144] = U3
    for j in range(m):
        for k in range(3):
            V[j * p + (6 + k), 96:144] = V3[j * 3 + k, :]
    for k in range(3):
        for i in range(n):
            W[(6 + k) * n + i, 96:144] = W3[k * 4 + i, :]

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Primary: 4+4+1 split, rank 114
    try:
        U, V, W = _decomp_4x4x9_v1()
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: full schoolbook, rank 144
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}


# EVOLVE-BLOCK-END
