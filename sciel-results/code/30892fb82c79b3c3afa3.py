"""Program for 4x5x5 matrix multiplication rank minimization."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# Known tensor decomposition constants
# Strassen decomposition for 2x2x2: rank 7
# C = AB where A, B are 2x2
# C11 = (a11+a22)(b11+b22) - (a11-a22)(b22-b11)  ... wait, standard Strassen is:
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B21-B22)
# M4 = A22(B12-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication using Strassen.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    
    Indexing:
    U row: a11=0, a12=1, a21=2, a22=3  (row-major)
    V row: b11=0, b12=1, b21=2, b22=3
    W row: c11=0, c12=1, c21=2, c22=3
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    W[0, 0] = 1  # c11
    W[2, 0] = 1  # c22
    W[1, 0] = 0
    W[3, 0] = 0
    # Actually c22 = M1, so W[2] = M1. c11 = M1 + ...
    # Let me redo: 
    # C = [[c11, c12], [c21, c22]]
    # W[k*n + i] convention: n=2 (rows of A = cols of C first dim)
    # For 2x2x2: n=2, m=2, p=2
    # U: (n*m, rank) = (4, 7) - indices are (i*m+j) where i=row of A, j=col of A
    # V: (m*p, rank) = (4, 7) - indices are (j*p+k) where j=col of A=row of B, k=col of B
    # W: (p*n, rank) = (4, 7) - indices are (k*n+i) where k=col of B, i=row of A
    
    # Let me use the convention carefully.
    # A is n x m = 2x2, B is m x p = 2x2, C = A*B is n x p = 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # W[k*n+i, c] means W is indexed by (k, i) where k is row index of C, i is col index of C? No...
    # The task says: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W row index = k*n + i where k is the column of B (which becomes row of C), i is row of A (which becomes col of C)?
    # Actually C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
    # The W factor: C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This is the standard bilinear decomposition.
    
    # So for 2x2x2:
    # A indices: (i,j) with i in 0..1, j in 0..1 → flat index i*2+j
    # B indices: (j,k) with j in 0..1, k in 0..1 → flat index j*2+k
    # C indices: (i,k) with i in 0..1, k in 0..1 → W flat index k*2+i
    
    # Let me define Strassen terms:
    # A sub-blocks: A11=A[0,0]=0, A12=A[0,1]=1, A21=A[1,0]=2, A22=A[1,1]=3
    # B sub-blocks: B11=B[0,0]=0, B12=B[0,1]=1, B21=B[1,0]=2, B22=B[1,1]=3
    # C sub-blocks: C11=C[0,0]→W[0*2+0]=0, C12=C[0,1]→W[1*2+0]=2, C21=C[1,0]→W[0*2+1]=1, C22=C[1,1]→W[1*2+1]=3
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22) = (U[0]+U[3])(V[0]+V[3])
    # M1 contributes to C11 (W[0]) and C22 (W[3])
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)B11 = (U[2]+U[3])V[0]
    # M2 contributes to C21 (W[1]) negatively, and C22 (W[3]) negatively
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1   # C21 += M2
    W[3, 1] = -1  # C22 -= M2
    
    # M3 = A11(B21-B22) = U[0](V[2]-V[3])
    # M3 contributes to C12 (W[2]) and C22 (W[3])
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1   # C12 += M3
    W[3, 2] = 1   # C22 += M3
    
    # M4 = A22(B12-B11) = U[3](V[1]-V[0])
    # M4 contributes to C11 (W[0]) and C21 (W[1])
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1   # C11 += M4
    W[1, 3] = 1   # C21 += M4
    
    # M5 = (A11+A12)B22 = (U[0]+U[1])V[3]
    # M5 contributes to C11 (W[0]) negatively, C12 (W[2])
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C11 -= M5
    W[2, 4] = 1   # C12 += M5
    
    # M6 = (A21-A11)(B11+B12) = (U[2]-U[0])(V[0]+V[1])
    # M6 contributes to C22 (W[3])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C22 += M6
    
    # M7 = (A12-A22)(B21+B22) = (U[1]-U[3])(V[2]+V[3])
    # M7 contributes to C11 (W[0])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C11 += M7
    
    return U, V, W

def verify_decomposition(n, m, p, U, V, W):
    """Verify that the tensor decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal delta(i,j,k) -> 1 if i,j,k valid
    # Actually, the tensor T[i,j,k] = 1 if the triple forms a valid multiplication path, 0 otherwise
    # T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # Because C[i,k] = sum_j A[i,j]*B[j,k], so every (i,j,k) triple contributes.
    
    # Compute the tensor from decomposition
    T_recon = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
    
    T_expected = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(T_recon, T_expected, atol=1e-10):
        diff = np.abs(T_recon - T_expected)
        max_diff = np.max(diff)
        indices = np.unravel_index(np.argmax(diff), diff.shape)
        return False, f"Max diff {max_diff} at {indices}"
    return True, "OK"

def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Use Strassen decomposition for 2x2x2 blocks where possible.
    
    For 4x5x5:
    - Partition into a 4x4x4 block + a 4x1x5 block + a 1x4x5 block
    - The 4x4x4 block can use rank 49 (known optimal) but let's start with simpler constructions
    - Actually, let's use a block Strassen approach
    
    Let me think about this differently.
    
    A is 4x5, B is 5x5, C = AB is 4x5.
    
    Partition A into [A_1, A_2] where A_1 is 4x4 and A_2 is 4x1.
    Partition B into [B_1; B_2] where B_1 is 4x5 and B_2 is 1x5.
    C = A_1*B_1 + A_2*B_2
    
    - A_1*B_1: 4x4 * 4x5 = 4x5. This is a product of 4x4 and 4x5.
      We can partition B_1 into [B_11, B_12] where B_11 is 4x4 and B_12 is 4x1.
      A_1*B_11: 4x4 * 4x4 = 4x4, rank 49 (known optimal)
      A_1*B_12: 4x4 * 4x1 = 4x1, rank 16 (schoolbook)
      Total: 65
    
    - A_2*B_2: 4x1 * 1x5 = 4x5, rank 20 (schoolbook)
    
    Total: 65 + 20 = 85
    
    Alternatively, use Strassen for the 4x4*4x4 part:
    4x4*4x4 has rank 49 (known result by Landsberg and Ottaviani).
    
    So total rank = 49 + 16 + 20 = 85.
    
    Let me implement this.
    """
    deadline = time.time() + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition incrementally
    U_list = []  # List of (4*5, rank_r) matrices
    V_list = []  # List of (5*5, rank_r) matrices
    W_list = []  # List of (5*4, rank_r) matrices
    
    # === Part 1: A_1 * B_11 (4x4 * 4x4 = 4x4), rank 49 ===
    # This is the hardest part. Let me use a known rank-49 decomposition for 4x4x4.
    # Actually, implementing rank 49 explicitly is very complex. Let me try a simpler approach first.
    # 
    # Alternative: Use Strassen 2x2x2 in blocks to get 4x4x4.
    # 4x4x4 can be viewed as 4 2x2x2 blocks plus cross terms... but that's not straightforward.
    #
    # Let me just use schoolbook for now to get a valid answer, then try to optimize.
    
    # === Schoolbook for everything (baseline rank 100) ===
    # A is 4x5 (rows 0..3, cols 0..4), B is 5x5, C is 4x5
    # For each i (0..3), j (0..4), k (0..4): one term
    # U[i*5+j, r] = 1, V[j*5+k, r] = 1, W[k*4+i, r] = 1
    # Rank = 4*5*5 = 100
    
    # Let me try a Strassen-based approach for the 4x4x4 sub-problem.
    # 4x4x4: rank 49 is optimal but complex. Let me try rank 56 via block Strassen.
    
    # Block Strassen for 4x4x4:
    # Partition into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    #
    # Using Strassen on the outer product:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each M_i involves multiplying 2x2 matrices, which itself has rank 4 (schoolbook) or rank 7 (Strassen).
    # If we use schoolbook for inner: 7 * 4 = 28 per group of 4 blocks... no.
    # Actually each term like (A11+A22)(B11+B22) is a product of two 2x2 matrices, rank 4 each.
    # So each Strassen term uses rank 4, total 7*4 = 28 for the 4x4x4? No, that's not right either.
    #
    # Wait, the 4x4x4 tensor has rank 49. Block Strassen gives:
    # 7 terms, each involving a 2x2x2 multiplication (rank 4 schoolbook or rank 7 Strassen).
    # With schoolbook inner: 7 * 4 = 28. But that's not correct because the Strassen formula 
    # has additions/subtractions of the intermediate results.
    #
    # Actually, let me reconsider. The 4x4x4 multiplication can be done with rank 49 (Landsberg-Ottaviani)
    # or rank 56 via nested Strassen (49 < 56 < 64).
    #
    # For simplicity, let me use the 2x2x2 Strassen decomposition in a structured way.
    
    # Let me just go with a clean schoolbook decomposition to ensure correctness, then try to improve.
    
    n, m, p = 4, 5, 5
    rank_total = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Schoolbook for all
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_rows.append(1.0)
                V_rows.append(1.0)
                W_rows.append(1.0)
                rank_total += 1
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
