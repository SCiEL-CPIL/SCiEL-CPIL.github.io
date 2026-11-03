import numpy as np
import time

def _strassen_2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) for A = [[a00, a01], [a10, a11]]
    V: (4, 7) for B = [[b00, b01], [b10, b11]]
    W: (4, 7) for C = [[c00, c01], [c10, c11]]
    Index convention: U[i*2+j, t], V[i*2+j, t], W[i*2+j, t] for 2x2 blocks.
    """
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b10)
    # M4 = a11(b10-b01)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 - M4 + M5 + M6
    # c01 = M3 + M4
    # c10 = M2 + M5
    # c11 = M1 - M2 + M3 + M7

    # U rows: a00, a01, a10, a11
    U = np.zeros((4, 7))
    # V rows: b00, b01, b10, b11
    V = np.zeros((4, 7))
    # W rows: c00, c01, c10, c11
    W = np.zeros((4, 7))

    # M1: (a00+a11)(b00+b11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1  # c00, c11

    # M2: (a10+a11)b00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1  # c10, c11

    # M3: a00(b01-b10)
    U[0, 2] += 1
    V[1, 2] += 1; V[2, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1  # c01, c11

    # M4: a11(b10-b01)
    U[3, 3] += 1
    V[2, 3] += 1; V[1, 3] -= 1
    W[0, 3] -= 1; W[1, 3] += 1  # c00, c01

    # M5: (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] += 1; W[2, 4] += 1  # c00, c10

    # M6: (a10-a00)(b00+b01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[0, 5] += 1  # c00

    # M7: (a01-a11)(b10+b11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[3, 6] += 1  # c11

    return U, V, W


def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    deadline = t0 + 145.0

    # Strategy: Split B (4x7) into B1 (4x4) and B2 (4x3)
    # Compute A*B1 using Strassen block decomposition: rank 49
    # Compute A*B2 using schoolbook: rank 4*4*3 = 48
    # Total rank: 97

    U_222, V_222, W_222 = _strassen_222()
    r_str = 7

    # For A (4x4) * B1 (4x4) using Strassen:
    # Split A into 2x2 blocks: A11, A12, A21, A22
    # Split B1 into 2x2 blocks: B11, B12, B21, B22
    # C1 = A*B1 has 2x2 blocks: C11, C12, C21, C22

    # For each of the 4 output blocks (C11, C12, C21, C22),
    # we need to express them as sums of products of linear combinations.
    # Using Strassen, each 2x2 block product needs rank 7.
    # But we're computing all 4 blocks simultaneously.
    #
    # Actually, the standard way: A (4x4) * B (4x4) can be viewed as
    # computing 4 independent 2x2*2x2 multiplications? No.
    #
    # Better: Use the tensor product structure.
    # A*B where A is 4x4 and B is 4x4.
    # Split into quadrants. The result C[i,j] = sum_k A[i,k] B[k,j].
    #
    # For block (I, J) of C (each 2x2), C[I,J] = sum_{K in {1,2}} A[I,K] * B[K,J]
    # where I, J, K are block indices.
    #
    # So for block row I and block col J:
    # C[I,J] = A[I,1]*B[1,J] + A[I,2]*B[2,J]
    #
    # This is a 2x2*2x2 + 2x2*2x2 = sum of two rank-7 decompositions = rank 14?
    # No, we can do better. Let's just do schoolbook for the 4x4*4x4 part.
    # Actually, let me just use schoolbook for everything but try to be clever.

    # Let me just implement the straightforward block approach:
    # A (4x4) * B (4x7)
    # Split B into B1 (4x4) and B2 (4x3)
    # For A*B1: use schoolbook with 4x4*4 = 16 terms per row of A... 
    # Actually schoolbook for 4x4*4x4 is 64 terms.
    # For A*B2: schoolbook for 4x4*4x3 is 48 terms.
    # Total: 112. That's the same as before.

    # Let me try the Strassen approach properly for 4x4*4x4.
    # View A as 2x2 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    # View B1 as 2x2 blocks of 2x2: B1 = [[B11, B12], [B21, B22]]
    # C1 = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #       [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    #
    # Each C block is a sum of two 2x2*2x2 products.
    # Using Strassen rank 7 for each 2x2*2x2:
    # C11 = A11*B11 + A12*B21 -> rank 7+7=14
    # C12 = A11*B12 + A12*B22 -> rank 7+7=14
    # C21 = A21*B11 + A22*B21 -> rank 7+7=14
    # C22 = A21*B12 + A22*B22 -> rank 7+7=14
    # Total: 56. Better than 64!

    # But wait, can we do better by sharing? The Strassen factors for different
    # products are independent, so no sharing. 56 + 48 = 104. Still not great.

    # Let me just go with 104.

    rank_total = 56 + 48
    U = np.zeros((16, rank_total))
    V = np.zeros((28, rank_total))
    W = np.zeros((28, rank_total))

    col = 0
    # U rows: A[i,j] for i in 0..3, j in 0..3 -> index i*4+j
    # V rows: B[k,l] for k in 0..3, l in 0..6 -> index k*7+l
    # W rows: C[i,k] for i in 0..3, k in 0..6 -> index k*4+i (trace convention W[k*n+i, r])

    # A*B1 part (56 terms)
    # Block indices: A blocks (bi, bj) where bi, bj in {0,1} for 2x2 blocks
    # A[bi*2+di, bj*2+dj] for di,dj in {0,1}
    # B1 blocks: B1[bi*2+dk, bj*2+dl]
    # For each output block (bi, bj), C1[bi*2+di, bj*2+dj] = sum over bk of A[bi,bk]*B1[bk,bj]

    for bi in range(2):  # block row of A
        for bj in range(2):  # block col of C1 (and B1)
            for bk in range(2):  # block index to sum over
                # This is a 2x2 * 2x2 multiplication: A[bi,bk] * B1[bk,bj]
                # Use Strassen rank 7
                for s in range(7):
                    # U: contributions from A[bi,bk]
                    for di in range(2):
                        for dj in range(2):
                            U[(bi*2+di)*4 + (bk*2+dj), col] += U_222[di*2+dj, s]
                    # V: contributions from B1[bk,bj]
                    for dk in range(2):
                        for dl in range(2):
                            V[(bk*2+dk)*7 + (bj*2+dl), col] += V_222[dk*2+dl, s]
                    # W: contributions to C1[bi,bj]
                    for di in range(2):
                        for dj in range(2):
                            W[(bj*2+dj)*4 + (bi*2+di), col] += W_222[di*2+dj, s]
                    col += 1

    # A*B2 part (48 terms, schoolbook)
    # B2 is 4x3: columns 4,5,6 of B
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):  # B columns 4,5,6
                U[i*4+j, col] = 1
                V[j*7+k, col] = 1
                W[k*4+i, col] = 1
                col += 1

    assert col == rank_total, f"col={col}, rank_total={rank_total}"

    elapsed = time.time() - t0
    if elapsed > 140.0:
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
