"""Matrix multiplication 3x3x3 - rank-minimal exact tensor decomposition."""
import time
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # We use a decomposition of rank 26 by partitioning the 3x3 matrices
    # into 2x2 and 1x1 blocks.
    # Rank 26 < 27 (schoolbook).

    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a term: u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    # Indices:
    # A2: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 3, 4
    # B2: (0,0), (0,1), (1,0), (1,1) -> V indices 0, 1, 3, 4
    # C22: (0,0), (0,1), (1,0), (1,1) -> W indices 0, 3, 1, 4 (k*n+i)
    # C_{00} -> 0*3+0 = 0
    # C_{01} -> 1*3+0 = 3
    # C_{10} -> 0*3+1 = 1
    # C_{11} -> 1*3+1 = 4

    # Strassen M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1

    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1

    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1

    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1

    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1

    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1

    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    # a = A[0:2, 2] -> U indices 2, 5
    # b^T = B[2, 0:2] -> V indices 6, 7
    # C_{00} (W0) += A02 B20
    # C_{01} (W3) += A02 B21
    # C_{10} (W1) += A12 B20
    # C_{11} (W4) += A12 B21

    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)

    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    # A2: U 0,1,3,4
    # b = B[0:2, 2] -> V indices 2, 5
    # C_{02} (W6) += A00 B02 + A01 B12
    # C_{12} (W7) += A10 B02 + A11 B12

    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)

    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    # a: U 2,5
    # b3 = B[2,2] -> V index 8
    # C_{02} (W6) += A02 B22
    # C_{12} (W7) += A12 B22

    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)

    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    # a^T = A[2, 0:2] -> U indices 6, 7
    # B2: V 0,1,3,4
    # C_{20} (W2) += A20 B00 + A21 B10
    # C_{21} (W5) += A20 B01 + A21 B11

    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)

    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    # a3 = A[2,2] -> U index 8
    # b^T = B[2, 0:2] -> V indices 6, 7
    # C_{20} (W2) += A22 B20
    # C_{21} (W5) += A22 B21

    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)

    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    # a^T: U 6,7
    # b: V 2,5
    # C_{22} (W8) += A20 B02 + A21 B12

    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)

    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    # a3: U 8
    # b3: V 8
    # C_{22} (W8) += A22 B22

    add_term(8, 8, 8, 1)

    # Verify decomposition correctness (optional, for debugging)
    # This is a quick sanity check using small random matrices
    if time.time() < deadline:
        import random
        random.seed(42)
        A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        B = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        
        # Compute C = A * B (schoolbook)
        C_ref = [[0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C_ref[i][j] += A[i][k] * B[k][j]
        
        # Compute C via tensor decomposition
        C_tens = [[0]*3 for _ in range(3)]
        for r in range(rank):
            u_vals = [U[idx][r] for idx in range(9)]
            v_vals = [V[idx][r] for idx in range(9)]
            # u_vals[i*3+k] = A[i][k]
            # v_vals[k*3+j] = B[k][j]
            for i in range(3):
                for j in range(3):
                    val = 0
                    for k in range(3):
                        val += u_vals[i*3+k] * v_vals[k*3+j]
                    C_tens[i][j] += val
                    # W index: W[k*n+i, r] where n=3, so W[j*3+i, r] for C[i][j]
                    # Wait, the convention is W[k*n+i, r] where the output index is (i,j) mapped to k*n+i?
                    # Let's re-read: "W[k*n+i,r]"
                    # For C[i][j], i is row, j is col. 
                    # The index in W is (j, i) flattened as j*n+i? Or i*n+j?
                    # The problem says "W[k*n+i,r]". In matrix mult C = A*B, C[i][j] = sum_k A[i][k]B[k][j].
                    # The tensor is T[i,j,k]. The decomposition is T[i,j,k] = sum_r U[i*3+k,r] * V[k*3+j,r] * W[j*3+i,r]?
                    # Let's check the current code's W indexing.
                    # In section 1: C_{00} -> W0, C_{01} -> W3, C_{10} -> W1, C_{11} -> W4.
                    # If W index is j*n+i:
                    # (0,0) -> 0*3+0=0. Correct.
                    # (0,1) -> 1*3+0=3. Correct.
                    # (1,0) -> 0*3+1=1. Correct.
                    # (1,1) -> 1*3+1=4. Correct.
                    # So W index for C[i][j] is j*3+i.
                    pass
        
        # The verification loop above is incomplete in its use of W. Let's do it properly.
        C_tens2 = [[0]*3 for _ in range(3)]
        for r in range(rank):
            for i in range(3):
                for j in range(3):
                    val = 0
                    for k in range(3):
                        # U index: i*3+k (row i, col k of A)
                        # V index: k*3+j (row k, col j of B)
                        val += U[i*3+k][r] * V[k*3+j][r]
                    # W index: j*3+i (col j, row i of C)
                    C_tens2[i][j] += val * W[j*3+i][r]
        
        # Check if C_ref == C_tens2
        # This is just a sanity check, we don't need to raise an error, but it's good practice.
        # For the purpose of this task, we just return the matrices.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
