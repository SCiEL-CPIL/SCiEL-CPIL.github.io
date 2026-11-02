"""Rewritten program for the suite task; new approach using block decomposition with Strassen."""

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3

    # Split A (n x m) = [A1, A2] where A1 is n x 2, A2 is n x 1
    # B (m x p) = [[B11, B12], [B21, B22]] with 2x2 and 2x1 blocks
    # Actually, let's split m=3 into m1=2, m2=1
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B = [B1; B2] where B1 is 2x3, B2 is 1x3
    # C = A1 * B1 + A2 * B2

    # For A1 * B1: 2x2 times 2x3
    # Use Strassen for the 2x2x2 part (first 2 columns of B1), then handle 3rd column
    # Strassen 2x2x2: rank 7
    # For 2x2x1 (3rd column): rank 2 (since R(2,2,1)=2)
    # Total for 2x2x3: rank 7 + 2 = 9

    # For A2 * B2: 2x1 times 1x3
    # This is outer product of a 2-vector and 3-vector: rank 3 (schoolbook)
    # But can we do better? R(2,1,3) = 3 (min possible is max(2,1,3)=3? No, min is max(n,m,p) for outer product? Actually R(n,1,p) = max(n,p) if we think of it as sum of rank-1 tensors. For n=2,p=3, we need at least 3 because output is 2x3 and each rank-1 tensor contributes to at most one entry per row... Actually R(2,1,3) = 3.

    # Total rank: 9 + 3 = 12

    # Let's build this decomposition.

    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 6 x 12
    V = [[0] * rank for _ in range(m * p)]  # 9 x 12
    W = [[0] * rank for _ in range(p * n)]  # 6 x 12

    col = 0

    # Part 1: A1 * B1 where A1 is rows 0-1 of A (2x2), B1 is rows 0-1 of B (2x3)
    # Strassen for 2x2x2 (first two columns of B1)
    # Strassen 7 factors:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # In our indexing:
    # A rows: 0,1 (n=2), cols 0,1 (m1=2)
    # B rows: 0,1 (m1=2), cols 0,1 (p1=2)
    # U index: i*m + j = i*3 + j for i in {0,1}, j in {0,1}
    # V index: j*p + k = j*3 + k for j in {0,1}, k in {0,1}
    # W index: k*n + i = k*2 + i for k in {0,1}, i in {0,1}

    # Strassen factors:
    # Factor 0: M1
    U[0*3+0][col] = 1  # A11
    U[1*3+1][col] = 1  # A22
    V[0*3+0][col] = 1  # B11
    V[1*3+1][col] = 1  # B22
    W[0*2+0][col] = 1  # C11 += M1
    W[0*2+1][col] = 1  # C22 += M1
    col += 1

    # Factor 1: M2
    U[1*3+0][col] = 1  # A21
    U[1*3+1][col] = 1  # A22
    V[0*3+0][col] = 1  # B11
    W[1*2+0][col] = 1  # C21 += M2
    W[0*2+1][col] = -1  # C22 -= M2
    col += 1

    # Factor 2: M3
    U[0*3+0][col] = 1  # A11
    V[0*3+1][col] = 1  # B12
    V[1*3+1][col] = -1  # -B22
    W[0*2+1][col] = 1  # C12 += M3
    W[0*2+1][col] = 1  # C22 += M3
    col += 1

    # Factor 3: M4
    U[1*3+1][col] = 1  # A22
    V[1*3+0][col] = 1  # B21
    V[0*3+0][col] = -1  # -B11
    W[0*2+0][col] = 1  # C11 += M4
    W[1*2+0][col] = 1  # C21 += M4
    col += 1

    # Factor 4: M5
    U[0*3+0][col] = 1  # A11
    U[0*3+1][col] = 1  # A12
    V[1*3+1][col] = 1  # B22
    W[0*2+0][col] = -1  # C11 -= M5
    W[0*2+1][col] = 1  # C12 += M5
    col += 1

    # Factor 5: M6
    U[1*3+0][col] = 1  # A21
    U[0*3+0][col] = -1  # -A11
    V[0*3+0][col] = 1  # B11
    V[0*3+1][col] = 1  # B12
    W[0*2+1][col] = 1  # C22 += M6
    col += 1

    # Factor 6: M7
    U[0*3+1][col] = 1  # A12
    U[1*3+1][col] = -1  # -A22
    V[1*3+0][col] = 1  # B21
    V[1*3+1][col] = 1  # B22
    W[0*2+0][col] = 1  # C11 += M7
    col += 1

    # Part 1b: A1 * B1_col2 (third column of B1, which is col 2 of B, rows 0-1)
    # 2x2 times 2x1: rank 2
    # C_col2 = A1 * [b02, b12]^T
    # Use decomposition:
    # t1 = (a00 + a11) * (b02 + b12)
    # t2 = (a10 + a11) * b02
    # Actually, R(2,2,1) = 2. Let's use:
    # t1 = a00*b02 + a11*b12
    # t2 = a10*b02 + a01*b12
    # That's just schoolbook with 4 terms... No, rank 2 means we need 2 rank-1 tensors.
    # For 2x2x1 tensor with all ones, rank is 2.
    # Decomposition:
    # t1 = (a00 + a11) * (b02 + b12)  -- no, this doesn't work directly.
    # Let me think: C0 = a00*b02 + a01*b12
    # C1 = a10*b02 + a11*b12
    # We can write:
    # t1 = (a00 + a10) * b02
    # t2 = (a01 + a11) * b12
    # Then C0 = t1 part from a00*b02 + ... no, t1 gives (a00+a10)*b02 = a00*b02 + a10*b02
    # t2 gives (a01+a11)*b12 = a01*b12 + a11*b12
    # So C0 = a00*b02 + a01*b12 = (t1 - a10*b02) + (t2 - a11*b12)... not helpful.
    # Actually, the rank of the all-ones 2x2x1 tensor is 2.
    # Let's use:
    # t1 = (a00 + a01) * b02  -- no.
    # Standard: R(2,2,1) = 2.
    # t1 = a00*b02 + a11*b12  (one rank-1 tensor? No, rank-1 tensor is u_i * v_j * w_k)
    # A rank-1 term is u_i * v_j * w_k where w_k is just 1 since k=0.
    # So each rank-1 term is u_i * v_j.
    # We need u1_i * v1_j + u2_i * v2_j = [[1,0],[0,1]]? No, the tensor is T[i,j] = 1 for all i,j.
    # So we need u1_i * v1_j + u2_i * v2_j = 1 for all i,j in {0,1}.
    # u1 = [1,0], v1 = [1,0] gives [[1,0],[0,0]]
    # u2 = [0,1], v2 = [0,1] gives [[0,0],[0,1]]
    # Sum = [[1,0],[0,1]] not all ones.
    # u1 = [1,1], v1 = [1,0] gives [[1,0],[1,0]]
    # u2 = [0,0], v2 = [1,1]... 
    # Let me just use schoolbook for 2x2x1: rank 4? No, R(2,2,1) is definitely 2.
    # T[i,j] = 1. 
    # u1 = [1,1], v1 = [1,0] -> [[1,0],[1,0]]
    # u2 = [0,1], v2 = [0,1] -> [[0,0],[0,1]]
    # Sum = [[1,0],[1,1]] not all ones.
    # u1 = [1,0], v1 = [1,1] -> [[1,1],[0,0]]
    # u2 = [0,1], v2 = [0,1] -> [[0,0],[0,1]]
    # Sum = [[1,1],[0,1]] not all ones.
    # Hmm. Let me reconsider. Maybe I'm wrong about R(2,2,1).
    # Actually, R(2,2,1) = 2 is correct.
    # T = [[1,1],[1,1]]
    # u1 = [1,1], v1 = [1,1] -> [[1,1],[1,1]]
    # So one rank-1 tensor suffices! R(2,2,1) = 1? No, that's only if all entries are 1.
    # Wait, the tensor for 2x2x1 matrix mult is T[i,j,k] where i in {0,1}, j in {0,1}, k in {0}.
    # T[i,j,0] = 1 for all i,j.
    # A rank-1 tensor is u_i * v_j * w_k.
    # Since k=0 only, w_0 = 1.
    # So we need u_i * v_j = 1 for all i,j.
    # u = [1,1], v = [1,1] gives u_i*v_j = 1 for all i,j.
    # So R(2,2,1) = 1!
    # Yes! Because it's just the all-ones matrix, which has matrix rank 1.
    # So for 2x2x1, we only need 1 rank-1 term.

    # So for the third column of B1:
    # C_col2 = A1 * b_col2 where b_col2 = [b02, b12]^T
    # t = (a00 + a01 + a10 + a11) * ... no.
    # The rank-1 term: u_i * v_j where u = [1,1] for A rows, v = [1,1] for B rows.
    # U[i*3+j] for i in {0,1}, j in {0,1}: all get 1.
    # V[j*3+2] for j in {0,1}: both get 1.
    # W[2*2+i] for i in {0,1}: both get 1.

    # Factor 7: 2x2x1 for third column
    U[0*3+0][col] = 1  # A11
    U[0*3+1][col] = 1  # A12
    U[1*3+0][col] = 1  # A21
    U[1*3+1][col] = 1  # A22
    V[0*3+2][col] = 1  # B row 0, col 2
    V[1*3+2][col] = 1  # B row 1, col 2
    W[2*2+0][col] = 1  # C row 0, col 2
    W[2*2+1][col] = 1  # C row 1, col 2
    col += 1

    # Part 2: A2 * B2 where A2 is 2x1 (row 2 of A), B2 is 1x3 (row 2 of B)
    # C += A2 * B2 = outer product of A2 (2-vector) and B2 (3-vector)
    # Rank 3 (schoolbook)
    # Factor 8,9,10: for each column k of B2 (which is col 0,1,2 of B row 2)
    for k in range(3):
        U[2*3][col] = 1  # A row 2, col 0 (only element)
        V[2*3+k][col] = 1  # B row 2, col k
        W[k*2+0][col] = 1  # C row 0, col k
        W[k*2+1][col] = 1  # C row 1, col k
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
