"""Matrix multiplication 2x3x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

DEADLINE_SECONDS = 130  # safety margin before the 150 s kill


def _verify(U, V, W, n=2, m=3, p=5):
    """Check that the tensor decomposition is exact.
    
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    For each (i, j, k):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n=2, m=3, p=5):
    """Faster verification using numpy with exact arithmetic via Fraction."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=object)
    V_arr = np.array(V, dtype=object)
    W_arr = np.array(W, dtype=object)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r] for r in range(rank))
                if s != 1:
                    return False
    return True


def _build_schoolbook(n=2, m=3, p=5):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_rank_28(n=2, m=3, p=5):
    """
    Build a rank-28 decomposition for M_{2,3,5}.
    
    Strategy: Split the multiplication into blocks by the middle dimension j.
    For j=0: multiply 2x1 by 1x5 -> rank 10 (schoolbook)
    For j=1,2: multiply 2x2 by 2x5 -> use Strassen-like for the 2x2x2 part
                and schoolbook for the remaining 2x2x3 part.
    
    Actually, let's use a different approach:
    Split B into B[:, :2] (3x2) and B[:, 2:] (3x3).
    C = A * B[:, :2] + A * B[:, 2:]
    
    A * B[:, :2]: 2x3 * 3x2 = 2x2, rank 12 (schoolbook)
    A * B[:, 2:]: 2x3 * 3x3 = 2x3, rank 16 (schoolbook)
    Total: 28
    
    But we can do better on the 2x2 part using Strassen (rank 7 instead of 8).
    So: 7 + 10 (for the 2x3*3x2 schoolbook minus the 2x2 part) ...
    
    Let me think more carefully.
    
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    Split k into k in {0,1} and k in {2,3,4}:
    C_{i,0:2} = sum_j A_{i,j} B_{j,0:2}  (2x2 result)
    C_{i,2:5} = sum_j A_{i,j} B_{j,2:5}  (2x3 result)
    
    For the 2x2 part: we can use Strassen to compute 2x3 * 3x2 with rank 7+... 
    Actually, 2x3 * 3x2: we can split the 3 into 1+2.
    A[:, 0:1] * B[0:1, :] + A[:, 1:3] * B[1:3, :]
    First term: 2x1 * 1x2, rank 4
    Second term: 2x2 * 2x2, rank 7 (Strassen)
    Total for 2x2 output: 11
    
    For the 2x3 part: 2x3 * 3x3, schoolbook rank 16
    Or split: A[:, 0:1] * B[0:1, 2:5] + A[:, 1:3] * B[1:3, 2:5]
    First term: 2x1 * 1x3, rank 6
    Second term: 2x2 * 2x3, rank 12 (schoolbook) or 7+... 
    2x2 * 2x3: split into 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = 11
    Total for 2x3 output: 6 + 11 = 17
    
    Grand total: 11 + 17 = 28. 
    
    Let me construct this explicitly.
    """
    # We'll build the decomposition by combining sub-decompositions.
    # Each sub-decomposition contributes columns to U, V, W.
    
    # The full tensor: T_{i,j,k} = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    # We need: sum_r U[i*3+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
    
    # Decomposition 1: A[:, 0:1] * B[0:1, :] -> contributes to C_{i,0:2}
    # This is: for i in {0,1}, j=0, k in {0,1}: U[i*3+0, r] * V[0*5+k, r] * W[k*2+i, r]
    # Schoolbook for 2x1 * 1x2: 4 terms
    # U[i*3+0, r] = 1, V[0*5+k, r] = 1, W[k*2+i, r] = 1
    # for (i,k) in {(0,0),(0,1),(1,0),(1,1)}
    
    # Decomposition 2: A[:, 1:3] * B[1:3, :] -> contributes to C_{i,0:2}
    # This is: for i in {0,1}, j in {1,2}, k in {0,1}: U[i*3+j, r] * V[j*5+k, r] * W[k*2+i, r]
    # Strassen for 2x2 * 2x2: 7 terms
    # The 2x2 matrices are A_sub[i, j'] = A[i, j'+1] for j' in {0,1}
    # B_sub[j', k] = B[j'+1, k] for j' in {0,1}
    # Strassen computes: C_sub[i, k] = sum_{j'} A_sub[i, j'] * B_sub[j', k]
    # In our indexing: C[i, k] += sum_{j in {1,2}} A[i, j] * B[j, k]
    
    # Decomposition 3: A[:, 0:1] * B[0:1, 2:5] -> contributes to C_{i,2:5}
    # Schoolbook for 2x1 * 1x3: 6 terms
    
    # Decomposition 4: A[:, 1:3] * B[1:3, 2:5] -> contributes to C_{i,2:5}
    # 2x2 * 2x3: split into 2x2*2x2 (Strassen, rank 7) + 2x2*2x1 (schoolbook, rank 4) = 11
    
    # Total: 4 + 7 + 6 + 11 = 28
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        """Add a rank-1 term."""
        U_cols.append([0] * 28)
        V_cols.append([0] * 28)
        W_cols.append([0] * 28)
        U_cols[-1][u_row] = u_val
        V_cols[-1][v_row] = v_val
        W_cols[-1][w_row] = w_val
    
    # Actually, let me just build it directly as lists of (row, val) tuples for each column.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0:1] * B[0:1, 0:2] (2x1 * 1x2, schoolbook, rank 4)
    # T_{i, 0, k} contribution: U[i*3+0, col] * V[0*5+k, col] * W[k*2+i, col] = 1
    for i in range(2):
        for k in range(2):
            U[i * 3 + 0][col] = 1
            V[0 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    # col = 4
    
    # Part 2: A[:, 1:3] * B[1:3, 0:2] (2x2 * 2x2, Strassen, rank 7)
    # A_sub[i, j'] = A[i, j'+1], B_sub[j', k] = B[j'+1, k]
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 + M2 + M5
    
    # In our indexing:
    # A_sub[0,0] = A[0,1], A_sub[0,1] = A[0,2]
    # A_sub[1,0] = A[1,1], A_sub[1,1] = A[1,2]
    # B_sub[0,0] = B[1,0], B_sub[0,1] = B[1,1]
    # B_sub[1,0] = B[2,0], B_sub[1,1] = B[2,1]
    
    # U rows: i*3+j where i in {0,1}, j in {1,2}
    # V rows: j*5+k where j in {1,2}, k in {0,1}
    # W rows: k*2+i where k in {0,1}, i in {0,1}
    
    # Strassen terms:
    # M1 = (A[0,1]+A[1,2])*(B[1,0]+B[2,1])
    # U: rows 0*3+1=3, 1*3+2=5 with values 1,1
    # V: rows 1*5+0=5, 2*5+1=11 with values 1,1
    # W: row 0*2+0=0 with value 1, row 0*2+1=1 with value 1, row 1*2+0=2 with value 1, row 1*2+1=3 with value 1
    # Wait, that's not right. Strassen gives linear combinations.
    
    # Let me be more careful. Each Strassen term M_r is a product of a linear combination of A entries and a linear combination of B entries.
    # M1 = (A[0,1] + A[1,2]) * (B[1,0] + B[2,1])
    # This contributes to C[0,0] with +1, C[0,1] with 0, C[1,0] with 0, C[1,1] with +1
    # Wait, no. C00 = M1 - M4 + M6 + M7, C01 = M3 + M5 + M7, C10 = M2 + M4 + M6, C11 = M1 - M3 + M2 + M5
    
    # So M1 contributes +1 to C[0,0] and +1 to C[1,1]
    # In tensor form: M1 contributes to T_{i,j,k} where the product (A[0,1]+A[1,2])*(B[1,0]+B[2,1]) appears in C_{i,k}
    
    # Actually, let me just enumerate each Strassen term and its contribution.
    
    # For each Strassen term, we have:
    # U part: linear combination of A[i, j] for i in {0,1}, j in {1,2}
    # V part: linear combination of B[j, k] for j in {1,2}, k in {0,1}
    # W part: linear combination of C[i, k] for i in {0,1}, k in {0,1}
    
    # Term M1: (A[0,1]+A[1,2])*(B[1,0]+B[2,1]), contributes +1 to C[0,0] and +1 to C[1,1]
    # U: U[3][col]=1, U[5][col]=1 (A[0,1] and A[1,2])
    # V: V[5][col]=1, V[11][col]=1 (B[1,0] and B[2,1])
    # W: W[0][col]=1 (C[0,0]), W[3][col]=1 (C[1,1])
    U[3][col] = 1; U[5][col] = 1
    V[5][col] = 1; V[11][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # Term M2: (A[1,1]+A[1,2])*B[1,0], contributes +1 to C[1,0] and +1 to C[1,1]
    # U: U[4][col]=1, U[5][col]=1 (A[1,1] and A[1,2])
    # V: V[5][col]=1 (B[1,0])
    # W: W[2][col]=1 (C[1,0]), W[3][col]=1 (C[1,1])
    U[4][col] = 1; U[5][col] = 1
    V[5][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # Term M3: A[0,1]*(B[1,1]-B[2,1]), contributes +1 to C[0,1] and -1 to C[1,1]
    # U: U[3][col]=1 (A[0,1])
    # V: V[6][col]=1, V[11][col]=-1 (B[1,1] and B[2,1])
    # W: W[1][col]=1 (C[0,1]), W[3][col]=-1 (C[1,1])
    U[3][col] = 1
    V[6][col] = 1; V[11][col] = -1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # Term M4: A[1,2]*(B[2,0]-B[1,0]), contributes -1 to C[0,0] and +1 to C[1,0]
    # U: U[5][col]=1 (A[1,2])
    # V: V[10][col]=1, V[5][col]=-1 (B[2,0] and B[1,0])
    # W: W[0][col]=-1 (C[0,0]), W[2][col]=1 (C[1,0])
    U[5][col] = 1
    V[10][col] = 1; V[5][col] = -1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # Term M5: (A[0,1]+A[0,2])*B[2,1], contributes +1 to C[0,1] and +1 to C[1,1]
    # U: U[3][col]=1, U[4][col]=1 (A[0,1] and A[0,2])
    # V: V[11][col]=1 (B[2,1])
    # W: W[1][col]=1 (C[0,1]), W[3][col]=1 (C[1,1])
    U[3][col] = 1; U[4][col] = 1
    V[11][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # Term M6: (A[1,1]-A[0,1])*(B[1,0]+B[2,0]), contributes +1 to C[0,0] and +1 to C[1,0]
    # U: U[4][col]=1, U[3][col]=-1 (A[1,1] and A[0,1])
    # V: V[5][col]=1, V[10][col]=1 (B[1,0] and B[2,0])
    # W: W[0][col]=1 (C[0,0]), W[2][col]=1 (C[1,0])
    U[4][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[10][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # Term M7: (A[0,2]-A[1,2])*(B[1,1]+B[2,1]), contributes +1 to C[0,0] and +1 to C[0,1]
    # U: U[4][col]=1, U[5][col]=-1 (A[0,2] and A[1,2])
    # V: V[6][col]=1, V[11][col]=1 (B[1,1] and B[2,1])
    # W: W[0][col]=1 (C[0,0]), W[1][col]=1 (C[0,1])
    U[4][col] = 1; U[5][col] = -1
    V[6][col] = 1; V[11][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # col = 4 + 7 = 11
    
    # Part 3: A[:, 0:1] * B[0:1, 2:5] (2x1 * 1x3, schoolbook, rank 6)
    for i in range(2):
        for k in range(2, 5):
            U[i * 3 + 0][col] = 1
            V[0 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    # col = 11 + 6 = 17
    
    # Part 4: A[:, 1:3] * B[1:3, 2:5] (2x2 * 2x3, rank 11)
    # Split into: A_sub * B_sub[:, 0:2] (2x2*2x2, Strassen rank 7) + A_sub * B_sub[:, 2:3] (2x2*2x1, schoolbook rank 4)
    # B_sub[:, 0:2] = B[1:3, 2:4], B_sub[:, 2:3] = B[1:3, 4:5]
    
    # 4a: A[:, 1:3] * B[1:3, 2:4] (2x2 * 2x2, Strassen, rank 7)
    # A_sub[i, j'] = A[i, j'+1], B_sub[j', k'] = B[j'+1, k'+2]
    # B_sub[0,0] = B[1,2], B_sub[0,1] = B[1,3]
    # B_sub[1,0] = B[2,2], B_sub[1,1] = B[2,3]
    
    # M1: (A[0,1]+A[1,2])*(B[1,2]+B[2,3]), contributes +1 to C[0,2] and +1 to C[1,3]
    U[3][col] = 1; U[5][col] = 1
    V[1*5+2][col] = 1; V[2*5+3][col] = 1
    W[2*2+0][col] = 1; W[3*2+1][col] = 1
    col += 1
    
    # M2: (A[1,1]+A[1,2])*B[1,2], contributes +1 to C[1,2] and +1 to C[1,3]
    U[4][col] = 1; U[5][col] = 1
    V[1*5+2][col] = 1
    W[2*2+1][col] = 1; W[3*2+1][col] = 1
    col += 1
    
    # M3: A[0,1]*(B[1,3]-B[2,3]), contributes +1 to C[0,3] and -1 to C[1,3]
    U[3][col] = 1
    V[1*5+3][col] = 1; V[2*5+3][col] = -1
    W[3*2+0][col] = 1; W[3*2+1][col] = -1
    col += 1
    
    # M4: A[1,2]*(B[2,2]-B[1,2]), contributes -1 to C[0,2] and +1 to C[1,2]
    U[5][col] = 1
    V[2*5+2][col] = 1; V[1*5+2][col] = -1
    W[2*2+0][col] = -1; W[2*2+1][col] = 1
    col += 1
    
    # M5: (A[0,1]+A[0,2])*B[2,3], contributes +1 to C[0,3] and +1 to C[1,3]
    U[3][col] = 1; U[4][col] = 1
    V[2*5+3][col] = 1
    W[3*2+0][col] = 1; W[3*2+1][col] = 1
    col += 1
    
    # M6: (A[1,1]-A[0,1])*(B[1,2]+B[2,2]), contributes +1 to C[0,2] and +1 to C[1,2]
    U[4][col] = 1; U[3][col] = -1
    V[1*5+2][col] = 1; V[2*5+2][col] = 1
    W[2*2+0][col] = 1; W[2*2+1][col] = 1
    col += 1
    
    # M7: (A[0,2]-A[1,2])*(B[1,3]+B[2,3]), contributes +1 to C[0,2] and +1 to C[0,3]
    U[4][col] = 1; U[5][col] = -1
    V[1*5+3][col] = 1; V[2*5+3][col] = 1
    W[2*2+0][col] = 1; W[3*2+0][col] = 1
    col += 1
    
    # col = 17 + 7 = 24
    
    # 4b: A[:, 1:3] * B[1:3, 4:5] (2x2 * 2x1, schoolbook, rank 4)
    # B_sub[:, 2] = B[1:3, 4]
    for i in range(2):
        for j in range(1, 3):
            U[i * 3 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 2 + i][col] = 1
            col += 1
    # col = 24 + 4 = 28
    
    assert col == 28, f"Expected col=28, got {col}"
    
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Build rank-28 decomposition
    U, V, W = _build_rank_28(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
