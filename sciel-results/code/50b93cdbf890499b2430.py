"""
Rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
Uses a constructive approach to find a low-rank decomposition.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _validate(U, V, W, n=3, m=3, p=6):
    """Validate that the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] where T corresponds to:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal A[i,j]*B[j,k]
    # Actually, the tensor for matrix mult is: M[i,j,k] = A[i,j] * B[j,k]
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k]
    # Let's verify with random matrices
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        # Compute using decomposition
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # The term is u * v * w where the contraction is:
            # C_decomp[i,k] += sum_j u[i,j] * v[j,k] * w[k,i]... 
            # Wait, let me re-read the convention.
            # W[k*n+i, r] - so W is indexed by (k, i)
            # The standard tensor decomposition for matmul:
            # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Let's verify this formula
            C_decomp2 = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_decomp2[i, k] = s
            if not np.allclose(C_decomp2, C, atol=1e-9):
                return False
    return True

def _construct_strassen_2x2x2():
    """Construct the rank-7 Strassen decomposition for 2x2 * 2x2."""
    # Strassen's 7 products for 2x2 matmul
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U: 4 rows (a11,a12,a21,a22), 7 cols
    # V: 4 rows (b11,b12,b21,b22), 7 cols
    # W: 4 rows (c11,c12,c21,c22), 7 cols
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a11+a22)(b11+b22) -> contributes to C11(+1), C22(-1)
    U[0, 0] = 1; U[2, 0] = 1  # a11, a22
    V[0, 0] = 1; V[2, 0] = 1  # b11, b22
    W[0, 0] = 1; W[3, 0] = -1 # c11, c22
    
    # M2: (a21+a22)b11 -> contributes to C21(+1), C22(-1)
    U[2, 1] = 1; U[3, 1] = 1  # a21, a22
    V[0, 1] = 1  # b11
    W[2, 1] = 1; W[3, 1] = -1 # c21, c22
    
    # M3: a11(b12-b22) -> contributes to C12(+1), C22(+1)
    U[0, 2] = 1  # a11
    V[1, 2] = 1; V[3, 2] = -1  # b12, b22
    W[1, 2] = 1; W[3, 2] = 1  # c12, c22
    
    # M4: a22(b21-b11) -> contributes to C11(+1), C21(+1)
    U[3, 3] = 1  # a22
    V[2, 3] = 1; V[0, 3] = -1  # b21, b11
    W[0, 3] = 1; W[2, 3] = 1  # c11, c21
    
    # M5: (a11+a12)b22 -> contributes to C11(-1), C12(+1)
    U[0, 4] = 1; U[1, 4] = 1  # a11, a12
    V[3, 4] = 1  # b22
    W[0, 4] = -1; W[1, 4] = 1  # c11, c12
    
    # M6: (a21-a11)(b11+b12) -> contributes to C22(+1)
    U[2, 5] = 1; U[0, 5] = -1  # a21, a11
    V[0, 5] = 1; V[1, 5] = 1  # b11, b12
    W[3, 5] = 1  # c22
    
    # M7: (a12-a22)(b21+b22) -> contributes to C11(+1)
    U[1, 6] = 1; U[3, 6] = -1  # a12, a22
    V[2, 6] = 1; V[3, 6] = 1  # b21, b22
    W[0, 6] = 1  # c11
    
    return U, V, W

def _construct_3x3x3_decomposition():
    """
    Construct a low-rank decomposition for 3x3 * 3x3 matrix multiplication.
    Uses a partitioned approach with Strassen for 2x2 blocks.
    """
    n, m, p = 3, 3, 3
    # Split A into blocks: A = [[A11, A12], [A21, A22]]
    # A11: 2x2, A12: 2x1, A21: 1x2, A22: 1x1
    # Split B similarly: B = [[B11, B12], [B21, B22]]
    # B11: 2x2, B12: 2x1, B21: 1x2, B22: 1x1
    
    # C = A*B
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x1)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x1)
    
    # Terms:
    # 1. A11*B11: 2x2*2x2 -> rank 7 (Strassen)
    # 2. A12*B21: 2x1*1x2 -> rank 4 (outer product, can be done in 4)
    #    Actually, (2x1)*(1x2) is a 2x2 matrix. Rank is 4 (each element separately)
    #    But we can do: A12*B21 where A12=[a12_1, a12_2]^T, B21=[b21_1, b21_2]
    #    Result: [[a12_1*b21_1, a12_1*b21_2], [a12_2*b21_1, a12_2*b21_2]]
    #    This is rank 4 (4 independent products)
    # 3. A11*B12: 2x2*2x1 -> rank 4 (2x1 result, 4 products)
    # 4. A12*B22: 2x1*1x1 -> rank 2
    # 5. A21*B11: 1x2*2x2 -> rank 4
    # 6. A22*B21: 1x1*1x2 -> rank 2
    # 7. A21*B12: 1x2*2x1 -> rank 2 (scalar result: a21_1*b12_1 + a21_2*b12_2)
    #    Wait, this is a dot product, rank 2
    # 8. A22*B22: 1x1*1x1 -> rank 1
    
    # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
    
    # Let me build this more carefully.
    # U has 9 rows (a00,a01,a02,a10,a11,a12,a20,a21,a22), V has 9 rows, W has 9 rows
    
    U = []
    V = []
    W = []
    
    # Block A11 = [[a00, a01], [a10, a11]] (indices 0,1,3,4 in U)
    # Block A12 = [[a02], [a12]] (indices 2,5 in U)
    # Block A21 = [[a20, a21]] (indices 6,7 in U)
    # Block A22 = [[a22]] (index 8 in U)
    
    # Block B11 = [[b00, b01], [b10, b11]] (indices 0,1,3,4 in V)
    # Block B12 = [[b02], [b12]] (indices 2,5 in V)
    # Block B21 = [[b20, b21]] (indices 6,7 in V)
    # Block B22 = [[b22]] (index 8 in V)
    
    # C11 = [[c00, c01], [c10, c11]] (indices 0,1,3,4 in W)
    # C12 = [[c02], [c12]] (indices 2,5 in W)
    # C21 = [[c20, c21]] (indices 6,7 in W)
    # C22 = [[c22]] (index 8 in W)
    
    # 1. A11*B11 using Strassen (rank 7)
    U_s, V_s, W_s = _construct_strassen_2x2x2()
    # Map indices: A11 uses U rows 0,1,3,4; B11 uses V rows 0,1,3,4; C11 uses W rows 0,1,3,4
    a11_map = [0, 1, 3, 4]  # a00, a01, a10, a11
    b11_map = [0, 1, 3, 4]  # b00, b01, b10, b11
    c11_map = [0, 1, 3, 4]  # c00, c01, c10, c11
    
    for r in range(7):
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for i in range(4):
            u[a11_map[i]] += U_s[i, r]
            v[b11_map[i]] += V_s[i, r]
            w[c11_map[i]] += W_s[i, r]
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 2. A12*B21: (2x1)*(1x2) -> 2x2 result
    # A12 = [a02, a12]^T (U rows 2,5), B21 = [b20, b21] (V rows 6,7)
    # C11 += [[a02*b20, a02*b21], [a12*b20, a12*b21]]
    for i, (ai, vi) in enumerate([(2, 6), (2, 7), (5, 6), (5, 7)]):
        u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
        u[ai] = 1
        v[vi] = 1
        # Determine which C element
        if ai == 2 and vi == 6:
            w[0] = 1  # c00
        elif ai == 2 and vi == 7:
            w[1] = 1  # c01
        elif ai == 5 and vi == 6:
            w[3] = 1  # c10
        elif ai == 5 and vi == 7:
            w[4] = 1  # c11
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 3. A11*B12: (2x2)*(2x1) -> 2x1 result
    # A11 rows 0,1,3,4; B12 = [b02, b12]^T (V rows 2,5)
    # C12 += A11 * B12
    # C12[0] = a00*b02 + a01*b12
    # C12[1] = a10*b02 + a11*b12
    for (ai, bi, ci) in [(0,2,2), (1,5,2), (3,2,5), (4,5,5)]:
        u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
        u[ai] = 1
        v[bi] = 1
        w[ci] = 1
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 4. A12*B22: (2x1)*(1x1) -> 2x1 result
    # A12 = [a02, a12]^T (U rows 2,5), B22 = b22 (V row 8)
    # C12 += A12 * b22
    for ai, ci in [(2,2), (5,5)]:
        u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
        u[ai] = 1
        v[8] = 1
        w[ci] = 1
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 5. A21*B11: (1x2)*(2x2) -> 1x2 result
    # A21 = [a20, a21] (U rows 6,7), B11 cols
    # C21[0] = a20*b00 + a21*b10
    # C21[1] = a20*b01 + a21*b11
    for (ai, bi, ci) in [(6,0,6), (7,3,6), (6,1,7), (7,4,7)]:
        u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
        u[ai] = 1
        v[bi] = 1
        w[ci] = 1
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 6. A22*B21: (1x1)*(1x2) -> 1x2 result
    # A22 = a22 (U row 8), B21 = [b20, b21] (V rows 6,7)
    # C21 += a22 * B21
    for bi, ci in [(6,6), (7,7)]:
        u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
        u[8] = 1
        v[bi] = 1
        w[ci] = 1
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 7. A21*B12: (1x2)*(2x1) -> scalar
    # C22 += a20*b02 + a21*b12
    for ai, bi in [(6,2), (7,5)]:
        u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
        u[ai] = 1
        v[bi] = 1
        w[8] = 1  # c22
        U.append(u)
        V.append(v)
        W.append(w)
    
    # 8. A22*B22: scalar
    u = np.zeros(9); v = np.zeros(9); w = np.zeros(9)
    u[8] = 1
    v[8] = 1
    w[8] = 1
    U.append(u)
    V.append(v)
    W.append(w)
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    """
    n, m, p = 3, 3, 6
    
    # Build a rank-26 decomposition for 3x3x3, then duplicate for 3x3x6
    # with shifted indices in B and C
    U_333, V_333, W_333 = _construct_3x3x3_decomposition()
    rank_333 = U_333.shape[1]
    
    # For 3x3x6: split B into two 3x3 blocks B1 (cols 0-2) and B2 (cols 3-5)
    # C has 6 columns: C1 (cols 0-2) from A*B1, C2 (cols 3-5) from A*B2
    
    # U is the same for both blocks (9 rows, 2*rank_333 cols)
    # V: first rank_333 cols for B1, next rank_333 for B2
    # W: first rank_333 cols for C1 (3x3), next rank_333 for C2 (3x3)
    
    total_rank = 2 * rank_333
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Block 1: B cols 0-2, C cols 0-2
    U[:, :rank_333] = U_333
    V[:m*3, :rank_333] = V_333  # B rows for cols 0-2
    W[:3*n, :rank_333] = W_333  # C rows for cols 0-2
    
    # Block 2: B cols 3-5, C cols 3-5
    U[:, rank_333:] = U_333
    # B rows for cols 3-5: row j in B_333 corresponds to row j*p + (k+3) in full B
    # V_333 has rows indexed as j*3+k for j in 0-2, k in 0-2
    # In full B, this is j*6 + (k+3)
    for r in range(rank_333):
        for row in range(9):
            j = row // 3
            k = row % 3
            full_row = j * 6 + (k + 3)
            V[full_row, rank_333 + r] = V_333[row, r]
    
    # C rows for cols 3-5: W_333 has rows indexed as k*3+i for k in 0-2, i in 0-2
    # In full C, this is (k+3)*3 + i
    for r in range(rank_333):
        for row in range(9):
            k = row // 3
            i = row % 3
            full_row = (k + 3) * 3 + i
            W[full_row, rank_333 + r] = W_333[row, r]
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[float(U[i, j]) for j in range(total_rank)] for i in range(n*m)]
    V_list = [[float(V[i, j]) for j in range(total_rank)] for i in range(m*p)]
    W_list = [[float(W[i, j]) for j in range(total_rank)] for i in range(p*n)]
    
    # Round to nearest half-integer if close
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                mat[i][j] = round(mat[i][j] * 2) / 2
    
    return {"U": U_list, "V": V_list, "W": W_list}
