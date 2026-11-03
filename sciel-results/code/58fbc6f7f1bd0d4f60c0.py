"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 138  # seconds, leave 12s margin for 150s limit


def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic."""
    rank = len(U[0]) if U else 0
    def to_int2(x):
        v = int(round(x * 2))
        if abs(x * 2 - v) > 1e-9:
            return None
        return v
    
    U2 = [[to_int2(x) for x in row] for row in U]
    V2 = [[to_int2(x) for x in row] for row in V]
    W2 = [[to_int2(x) for x in row] for row in W]
    
    for mat in [U2, V2, W2]:
        for row in mat:
            if any(x is None for x in row):
                return False
    
    U3 = np.array(U2).reshape(n, m, rank)
    V3 = np.array(V2).reshape(m, p, rank)
    W3 = np.array(W2).reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 4:
                    return False
    return True


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7) with entries in {-1, 0, 1}.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Row/col indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1  # C00, C11... wait
    # C11 = M1 + M4 - M5 + M7, C22 = M1 - M2 + M3 + M6
    # W[k*2+i]: W[0*2+0]=W[0] for C[0,0], W[0*2+1]=W[1] for C[0,1]
    #           W[1*2+0]=W[2] for C[1,0], W[1*2+1]=W[3] for C[1,1]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W[0, 0] = 1   # C00 gets M1
    W[3, 0] = 1   # C11 gets M1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1   # C10 gets M2
    W[3, 1] = -1  # C11 gets -M2
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1   # C01 gets M3
    W[3, 2] = 1   # C11 gets M3
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1   # C00 gets M4
    W[2, 3] = 1   # C10 gets M4
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00 gets -M5
    W[1, 4] = 1   # C01 gets M5
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C11 gets M6
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00 gets M7
    
    return U, V, W


def _build_block_strassen(n, m, p):
    """Build block-structured decomposition using Strassen for 2x2 sub-blocks.
    
    For 2x4x7: split the 4 middle indices into two groups of 2.
    Group 1: j in {0,1}, Group 2: j in {2,3}
    
    For output columns k=0..3: use Strassen on each 2x2 block (7+7=14 terms)
    For output columns k=4..6: use schoolbook (2*4*3=24 terms)
    Total rank: 14 + 24 = 38
    """
    n, m, p = 2, 4, 7
    U_s, V_s, W_s = _strassen_2x2x2()  # Each (4, 7)
    
    rank = 14 + 24  # 38
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # For k in {0,1,2,3}: use Strassen decomposition
    # The 2x4x4 part: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # First sum is 2x2x2 (i in 0,1; j in 0,1; k in 0,3) -- but k only goes 0..3
    # Actually for k=0..3, we need B[j,k] for j=0,1 and j=2,3
    
    # For the first block (j=0,1): it's a 2x2x4 multiplication
    # A[i, j] for i=0,1; j=0,1 multiplied by B[j, k] for j=0,1; k=0,3
    # This is NOT a simple 2x2x2. Let me reconsider.
    
    # Actually, let me use a different approach.
    # For each pair of output columns, use Strassen.
    # C[i, 2k] and C[i, 2k+1] for k=0,1:
    #   C[i, 2k] = sum_j A[i,j] B[j, 2k]
    #   C[i, 2k+1] = sum_j A[i,j] B[j, 2k+1]
    # This is a 2x4x2 multiplication, which can be done with 2x2x2 Strassen
    # by grouping j into {0,1} and {2,3}.
    
    # For a 2x4x2 product: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k], i=0,1; k=0,1
    # Split j: j1 = j mod 2, j2 = j // 2
    # C[i,k] = sum_{j2=0}^{1} sum_{j1=0}^{1} A[i, j2*2+j1] B[j2*2+j1, k]
    # For each j2, this is a 2x2x2 product: A_block[i,j1] * B_block[j1,k]
    # Use Strassen for each j2 block: 7 terms per block, 14 total for 2x4x2
    
    # Let's verify Strassen for 2x2x2 first
    U3 = U_s.reshape(2, 2, 7)
    V3 = V_s.reshape(2, 2, 7)
    W3 = W_s.reshape(2, 2, 7)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U3[i,j,c] * V3[j,k,c] * W3[k,i,c] for c in range(7))
                assert s == 1, f"Strassen failed at ({i},{j},{k}): got {s}"
    
    # Now build the full decomposition
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # For each pair of output columns (2k, 2k+1), k=0,1:
    # Use two Strassen decompositions (one for j-block 0, one for j-block 1)
    for k_pair in range(2):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for j_block in range(2):
            # j_block=0: j in {0,1}, j_block=1: j in {2,3}
            # Strassen: A[i, j_block*2+j1] * B[j_block*2+j1, k] for k in {k0, k1}
            # This is 2x2x2 with:
            #   A' [i, j1] = A[i, j_block*2 + j1]
            #   B' [j1, k'] = B[j_block*2 + j1, k0 + k']
            #   C' [i, k'] = C[i, k0 + k']
            for c in range(7):
                # U part: A[i, j_block*2+j1]
                for i in range(2):
                    for j1 in range(2):
                        j = j_block * 2 + j1
                        U_list.append([0] * rank + [U_s[i * 2 + j1, c]])
                # V part: B[j_block*2+j1, k0+k']
                for j1 in range(2):
                    for k_prime in range(2):
                        j = j_block * 2 + j1
                        k = k0 + k_prime
                        V_list.append([0] * rank + [V_s[j1 * 2 + k_prime, c]])
                # W part: C[i, k0+k']
                for k_prime in range(2):
                    for i in range(2):
                        k = k0 + k_prime
                        W_list.append([0] * rank + [W_s[k_prime * 2 + i, c]])
                rank += 1
    
    # For output columns k=4,5,6: schoolbook
    for k in range(4, 7):
        for i in range(2):
            for j in range(4):
                U_list.append([0] * rank + [1])
                V_list.append([0] * rank + [1])
                W_list.append([0] * rank + [1])
                rank += 1
    
    U = np.array(U_list).T  # (rank, n*m) -> transpose to (n*m, rank)
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_strassen_full(n, m, p):
    """Build decomposition using Strassen for all 2x2 sub-blocks.
    
    For 2x4x7:
    - Split middle index into 2 blocks of 2
    - Split output into: first 4 columns use Strassen (14 terms), last 3 use schoolbook (24 terms)
    - Total: 38
    """
    n, m, p = 2, 4, 7
    U_s, V_s, W_s = _strassen_2x2x2()
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Strassen part for k=0..3 (two 2x4x2 sub-problems)
    for k_pair in range(2):
        k0 = 2 * k_pair
        k1 = 2 * k_pair + 1
        for j_block in range(2):
            for c in range(7):
                U_row = [0] * rank
                V_row = [0] * rank
                W_row = [0] * rank
                for i in range(2):
                    for j1 in range(2):
                        j = j_block * 2 + j1
                        U_row[i * m + j] = U_s[i * 2 + j1, c]
                for j1 in range(2):
                    for kp in range(2):
                        j = j_block * 2 + j1
                        k = k0 + kp
                        V_row[j * p + k] = V_s[j1 * 2 + kp, c]
                for kp in range(2):
                    for i in range(2):
                        k = k0 + kp
                        W_row[k * n + i] = W_s[kp * 2 + i, c]
                U_rows.append(U_row)
                V_rows.append(V_row)
                W_rows.append(W_row)
                rank += 1
    
    # Schoolbook part for k=4,5,6
    for k in range(4, 7):
        for i in range(2):
            for j in range(4):
                U_row = [0] * rank
                V_row = [0] * rank
                W_row = [0] * rank
                U_row[i * m + j] = 1
                V_row[j * p + k] = 1
                W_row[k * n + i] = 1
                U_rows.append(U_row)
                V_rows.append(V_row)
                W_rows.append(W_row)
                rank += 1
    
    return {"U": U_rows, "V": V_rows, "W": W_rows}


def _build_schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = 2, 4, 7
    
    # Try the block Strassen construction (rank 38)
    result = _build_strassen_full(n, m, p)
    if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook (rank 56)
    return _build_schoolbook(n, m, p)
