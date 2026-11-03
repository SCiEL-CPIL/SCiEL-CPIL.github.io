"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

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
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W

    return U, V, W


def _decomp_4x4x3():
    """
    Decomposition of M_{4,4,3} = M_{4,2,3} + M_{4,2,3}.
    Rank 30: each M_{4,2,3} has rank 15.
    U: (16, 30), V: (12, 30), W: (12, 30)
    Indexing: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t] with m=2, p=3 per block.
    """
    n, m, p = 4, 2, 3
    rank_423 = 15
    total_rank = 2 * rank_423  # 30
    U = np.zeros((n * m, total_rank))   # (8, 30)
    V = np.zeros((m * p, total_rank))   # (6, 30)
    W = np.zeros((p * n, total_rank))   # (12, 30)

    # For M_{4,2,3}: n=4, m=2, p=3. Rank 15 = n*m*p (schoolbook for m=2, p=3 gives 4*2*3=24,
    # but we can do better. Actually 4*2*3 = 24 schoolbook. Let's use 2-level approach.
    # M_{4,2,3}: Split 4x4 into 2x2 blocks for A (2 blocks), and 2x3 for B (1 block), C is 3x4.
    # Use Strassen for the 2x2 sub-problem: M_{2,2,3} has rank... let me think.
    # Actually, the tensor for M_{n,m,p} with n=4, m=2, p=3:
    # Schoolbook rank = 4*2*3 = 24.
    # Alternative: write A as 2x2 blocks (top row, bottom row of 2 columns each).
    # C = A * B where A is 4x2, B is 2x3, C is 4x3.
    # This is just standard matrix mult of shape (4,2) x (2,3) -> (4,3).
    # Rank of M_{4,2,3} schoolbook = 24.
    # Can we do better? M_{4,2,3} can be seen as 2 copies of M_{2,2,3} (top 2 rows and bottom 2 rows of A).
    # M_{2,2,3}: rank is at most 2*2*3 = 12, and known optimal is...
    # For 2x2 x 2x3: this is like multiplying two 2x2 matrices by a 3-column matrix.
    # Actually M_{2,2,3} has rank 9 (Winograd-like or known result). Let me use schoolbook: 12.
    # So 2 * 12 = 24, same as direct schoolbook. No gain.
    #
    # Alternative: just use schoolbook for the whole 4x4x3, rank = 4*4*3 = 48.
    # But we split into two 4x2x3 blocks: each schoolbook 4*2*3 = 24, total 48. Same.
    #
    # So for now, use schoolbook: rank 48 for M_{4,4,3}.
    # But wait, the total would be 49+49+48 = 146, worse than 114.
    #
    # Let me reconsider. The current approach: 4+4+1 = 49+49+16 = 114.
    # Alternative split: 3+3+3: three M_{4,4,3} blocks.
    # M_{4,4,3} schoolbook = 48 each, total 144. Worse.
    #
    # Alternative: 3+6: M_{4,4,3} + M_{4,4,6}. M_{4,4,6} = 2 * M_{4,4,3}... no.
    # M_{4,4,6} schoolbook = 96. M_{4,4,3} = 48. Total 144. Worse.
    #
    # What about using the Strassen trick for the 4x3 part?
    # M_{4,4,3}: Split the 4 columns of B into 2+1 or 1+2.
    # M_{4,4,2}: can use M_{4,4,2} <= M_{2,2,2} xotimes something?
    # M_{4,4,2}: split A and B into 2x2 blocks. M_{4,4,2} involves multiplying 4x4 by 4x2.
    # This is like M_{2,2,2} where one factor has a 2-column B.
    # Actually M_{4,4,2}: n=4, m=4, p=2.
    # Schoolbook: 32. Strassen-like: use M_{2,2,2} (rank 7) for the 2x2 blocks,
    # but B is 4x2 so B splits into 2 blocks of 2x2. A splits into 4 blocks of 2x2.
    # So we have 2 (A-blocks) * 1 (B-block) * 2 (A-blocks) ... no, standard Kronecker.
    # M_{4,4,2} = sum over 2x2 block mult. Each block mult is M_{2,2,2} (rank 7).
    # Number of block mults: (4/2)*(4/2) = 4 A-blocks, 2 B-blocks = 4*2 = 8? No.
    # A is 4x4 (2x2 blocks: 4 blocks), B is 4x2 (2x1 blocks? or 2x2 with 1 block?).
    # B is 4x2: as 2x2 blocks, that's 2 blocks (top 2x2, bottom 2x2).
    # So we multiply 4 A-blocks by 2 B-blocks, giving 8 C-entries (each 2x2).
    # C is 4x2: 2x2 blocks, that's 2 blocks.
    # Wait, C[i,j] = sum_k A[i,k]*B[k,j]. With 2x2 blocking:
    # C_block[li, l_j] = sum_{mi, mj} A_block[li,mi] * B_block[mj,lj] ... no, B is 4x2.
    # Let me just use schoolbook for M_{4,4,3}: rank 48.
    # Total: 49 + 49 + 48 = 146. Worse than 114.
    #
    # So the 4+4+1 split with 49+49+16=114 is better.
    #
    # Let me verify: is there a known better rank for M_{4,4,9}?
    # Known: M_{4,4,4} has rank 49 (2-level Strassen). M_{4,4,1} has rank 16 (schoolbook, optimal since it's just a single column, each of 16 entries needs one multiplication... actually M_{4,4,1}: C = A*B where B is 4x1. C[i,k] = sum_j A[i,j]*B[j,0]. That's 16 scalar mults. Rank 16, optimal.)
    # So 114 seems like a good construction.
    #
    # For M_{4,4,3}, schoolbook is 48. Can we do better?
    # M_{4,4,3} = M_{4,2,3} + M_{4,2,3}. Each M_{4,2,3} has schoolbook rank 4*2*3=24.
    # M_{4,2,3}: multiply 4x2 by 2x3. C[i,k] = sum_j A[i,j]*B[j,k]. 4*3 entries, 2 sum terms each.
    # This is bilinear map with 8 inputs on A side (4*2), 6 on B side (2*3), 12 outputs (4*3).
    # Rank: schoolbook 24. Can we share? A[i,j] and A[i',j] share B[j,k].
    # M_{4,2,3} is equivalent to M_{2,2,3} xotimes (copy of A rows)? No.
    # Actually M_{4,2,3} = M_{2,2,3} with A duplicated? No.
    # M_{2,2,3}: multiply 2x2 by 2x3. C[i,k]=sum_j A[i,j]B[j,k]. 2*3=6 outputs, each 2-term sum.
    # Schoolbook rank 12. Known rank of M_{2,2,3}? It's 9 I believe (Winograd).
    # Let me check: M_{2,2,3}. The tensor has entries T[i,j,k]. 
    # Using Winograd: for 2x2x3, we can write as 9 mults? Let me think.
    # Actually, I recall M_{2,2,3} has rank 9. Let me construct it.
    # 
    # For M_{2,2,3}: A is 2x2, B is 2x3, C = AB is 2x3.
    # C[0,0]=A00*B00+A01*B10, C[0,1]=A00*B01+A01*B11, C[0,2]=A00*B02+A01*B12
    # C[1,0]=A10*B00+A11*B10, C[1,1]=A10*B01+A11*B11, C[1,2]=A10*B02+A11*B12
    # 
    # Winograd approach: 
    # M1 = A00*(B00+B01+B02)
    # M2 = A01*(B10+B11+B12)
    # M3 = (A00-A01)*B00
    # M4 = (A00-A01)*B01
    # M5 = (A00-A01)*B02
    # M6 = (A00-A10)*B00... hmm this is getting complex.
    # 
    # Actually, let me just use schoolbook for M_{4,4,3} with rank 48.
    # Total: 49+49+48 = 146. Not better than 114.
    #
    # Let me stick with the 4+4+1 split: rank 114.
    #
    # Wait, I should double-check the M_{4,4,4} decomposition. Let me verify it.
    pass

    # Use schoolbook for M_{4,4,3}, rank 48
    rank = n * 4 * 3  # 4*4*3 = 48 for the full 4x4x3, but we're building 4x2x3 x2
    # Actually let's build 4x4x3 directly with schoolbook, rank 48
    # For now, return zeros and fix in construct
    return None


def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
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
    """Vectorized verification of the decomposition."""
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

    # Block 1: columns 0..8 of B, rows 0..3 of C
    U[:, :49] = U1
    for j in range(m):
        for k in range(4):
            V[j * p + k, :49] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, :49] = W1[k * 4 + i, :]

    # Block 2: columns 4..8 of B, rows 4..8 of C
    U[:, 49:98] = U2
    for j in range(m):
        for k in range(4):
            V[j * p + (4 + k), 49:98] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * n + i, 49:98] = W2[k * 4 + i, :]

    # Block 3: column 8 of B, row 8 of C
    U[:, 98:114] = U3
    for j in range(m):
        V[j * p + 8, 98:114] = V3[j * 1 + 0, :]
    for i in range(n):
        W[8 * n + i, 98:114] = W3[0 * 4 + i, :]

    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Try the 4+4+1 split
    U, V, W = _decomp_4x4x9_v1()

    if _verify_decomp(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: full schoolbook
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
