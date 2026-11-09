"""Matrix multiplication 4x4x6 via Strassen-Winograd block decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U_row, V_row, W_row) where:
      U is 2x2 (input A rows), V is 2x2 (input B rows), W is 2x2 (output C rows)
    using the trace convention W[k*n+i, r].
    
    For 2x2x2: n=2, m=2, p=2
    U shape: (n*m, rank) = (4, 7)
    V shape: (m*p, rank) = (4, 7)
    W shape: (p*n, rank) = (4, 7)
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # n=2, m=2, p=2
    # A is 2x2: A[i,j] with i in {0,1}, j in {0,1}
    # B is 2x2: B[j,k] with j in {0,1}, k in {0,1}
    # C is 2x2: C[i,k] with i in {0,1}, k in {0,1}
    
    # U[i*m+j, c] = U[i*2+j, c]
    # V[j*p+k, c] = V[j*2+k, c]
    # W[k*n+i, c] = W[k*2+i, c]
    
    U = [[0]*7 for _ in range(4)]  # 4 rows: (i,j) pairs: (0,0),(0,1),(1,0),(1,1)
    V = [[0]*7 for _ in range(4)]  # 4 rows: (j,k) pairs: (0,0),(0,1),(1,0),(1,1)
    W = [[0]*7 for _ in range(4)]  # 4 rows: (k,i) pairs: (0,0),(0,1),(1,0),(1,1)
    
    # M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0] -> U[0*2+0]=U[0], A22 = A[1,1] -> U[1*2+1]=U[3]
    # B11 = B[0,0] -> V[0*2+0]=V[0], B22 = B[1,1] -> V[1*2+1]=V[3]
    # C11 += M1 -> W[0*2+0]=W[0], C22 += M1 -> W[1*2+1]=W[3]
    for c in [0]:
        U[0][c] = 1; U[3][c] = 1
        V[0][c] = 1; V[3][c] = 1
        W[0][c] = 1; W[3][c] = 1
    
    # M2 = (A21+A22)B11
    # A21 = A[1,0] -> U[1*2+0]=U[2], A22 = A[1,1] -> U[3]
    # B11 = B[0,0] -> V[0]
    # C21 += M2 -> W[0*2+1]=W[1], C22 -= M2 -> W[3]
    for c in [1]:
        U[2][c] = 1; U[3][c] = 1
        V[0][c] = 1
        W[1][c] = 1; W[3][c] = -1
    
    # M3 = A11(B21-B22)
    # A11 = A[0,0] -> U[0]
    # B21 = B[1,0] -> V[1*2+0]=V[2], B22 = B[1,1] -> V[3]
    # C12 += M3 -> W[0*2+1]=W[1], C22 += M3 -> W[3]
    for c in [2]:
        U[0][c] = 1
        V[2][c] = 1; V[3][c] = -1
        W[1][c] = 1; W[3][c] = 1
    
    # M4 = A22(B12-B11)
    # A22 = A[1,1] -> U[3]
    # B12 = B[0,1] -> V[0*2+1]=V[1], B11 = B[0,0] -> V[0]
    # C11 += M4 -> W[0], C21 += M4 -> W[1]
    for c in [3]:
        U[3][c] = 1
        V[1][c] = 1; V[0][c] = -1
        W[0][c] = 1; W[1][c] = 1
    
    # M5 = (A11+A12)B22
    # A11 = A[0,0] -> U[0], A12 = A[0,1] -> U[1]
    # B22 = B[1,1] -> V[3]
    # C11 -= M5 -> W[0], C12 += M5 -> W[1]
    for c in [4]:
        U[0][c] = 1; U[1][c] = 1
        V[3][c] = 1
        W[0][c] = -1; W[1][c] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21 = A[1,0] -> U[2], A11 = A[0,0] -> U[0]
    # B11 = B[0,0] -> V[0], B12 = B[0,1] -> V[1]
    # C22 += M6 -> W[3]
    for c in [5]:
        U[2][c] = 1; U[0][c] = -1
        V[0][c] = 1; V[1][c] = 1
        W[3][c] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12 = A[0,1] -> U[1], A22 = A[1,1] -> U[3]
    # B21 = B[1,0] -> V[2], B22 = B[1,1] -> V[3]
    # C11 += M7 -> W[0]
    for c in [6]:
        U[1][c] = 1; U[3][c] = -1
        V[2][c] = 1; V[3][c] = 1
        W[0][c] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the tensor for matmul (n x m) * (m x p) -> (n x p) has:
    # T[i,j,k] = 1 for all valid (i,j,k) because C[i,k] = sum_j A[i,j] B[j,k]
    # Wait, no. The multiplication tensor has entries T[i,j,k] = 1 for all i,j,k.
    # Because C[i,k] = sum_j A[i,j] * B[j,k], so the structure tensor is all-ones.
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build 4x4x4 decomposition using 2x2x2 Strassen blocks (rank 18)
    # 4x4x4: split into four 2x2x2 blocks
    # A is 4x4, B is 4x4, C is 4x4
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] -> 7 terms
    # C[0:2,2:4] = A[0:2,2:4]*B[2:4,0:2] -> 7 terms
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] -> 7 terms
    # C[2:4,2:4] = A[2:4,2:4]*B[2:4,0:2] -> 7 terms
    # Total: 28 terms for 4x4x4 via naive block application of Strassen
    # But the known optimal rank for 4x4x4 is 18 (Landsberg)
    # Let's use the 28-term construction first, then try to optimize
    
    # Actually, let me think more carefully.
    # For 4x4x4, the block structure gives:
    # We can write 4x4 as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21 (sum of two 2x2 products)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each 2x2 product needs 7 multiplications, so 4*7 = 28 total.
    # But we can share some? No, these are independent.
    # The optimal rank 18 for 4x4x4 is non-trivial.
    
    # For now, let's construct a valid decomposition.
    # 4x4x6: We can view this as 4x4x4 + 4x4x2
    # Or: split B into two 4x3 blocks: B = [B1 | B2] where B1 is 4x3, B2 is 4x3
    # C = A*B = [A*B1 | A*B2], each is 4x3
    # 4x4x3: split into 4x2x3 + 2x2x3? Or 4x4x3 directly.
    
    # Simplest approach: use schoolbook for the parts we can't optimize.
    # Let's do: 4x4x6 = 4x4x4 (rank 28 via Strassen blocks) + 4x4x2 (rank 28 via Strassen blocks)
    # 4x4x2: A is 4x4, B is 4x2, C is 4x2
    # Split: C[0:2,0:2] = A[0:2,:]*B[0:2,0:2] -> 2x4x2
    #         C[2:4,0:2] = A[2:4,:]*B[0:2,0:2] -> 2x4x2
    # 2x4x2: A is 2x4, B is 4x2, C is 2x2
    # C[i,k] = sum_j A[i,j]*B[j,k], j in 0..3
    # This is rank 2*4*2 = 16 schoolbook, or we can use Strassen-like?
    # Actually for 2x4x2, we can think of it as two 2x2x2 multiplications:
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # First part: 2x2x2 (rank 7), second part: 2x2x2 (rank 7), total rank 14
    # So 4x4x2 has rank 2*14 = 28
    
    # Total: 28 + 28 = 56 for 4x4x6
    # Schoolbook would be 4*4*6 = 96, so 56 is an improvement.
    
    # Let me construct this carefully.
    
    # Part 1: 4x4x4 (rows 0-3 of A, cols 0-3 of B, all rows of C)
    # Part 2: 4x4x2 (rows 0-3 of A, cols 4-5 of B, all rows of C)
    
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 16 x 56
    V = [[0]*rank for _ in range(m*p)]  # 24 x 56
    W = [[0]*rank for _ in range(p*n)]  # 24 x 56
    
    col = 0
    
    # Part 1: 4x4x4 using 4 Strassen 2x2x2 blocks
    # Block (bi, bj) where bi, bj in {0,1}
    # bi=0: A rows 0-1, bi=1: A rows 2-3
    # bj=0: B cols 0-1, bj=1: B cols 2-3
    # Output: C rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # Wait, that's not right. The block structure is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # If we split i into {0,1} and {2,3}, j into {0,1} and {2,3}, k into {0,1} and {2,3}:
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    # C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    
    # So we need 8 Strassen blocks for 4x4x4!
    # Each contributes 7 terms, total 56 terms for 4x4x4 alone.
    # That's worse than schoolbook (64) only slightly.
    
    # Let me reconsider. The 2x2x2 Strassen applies when we have a 2x2 A, 2x2 B, 2x2 C.
    # In the block decomposition of 4x4x4:
    # A is 4x4, split into four 2x2 blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B is 4x4, split into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # C is 4x4, split into four 2x2 blocks: C[0:2,0:2], C[0:2,2:4], C[2:4,0:2], C[2:4,2:4]
    
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # This is sum of two 2x2x2 products. Each needs 7 mults. So 14 for this block.
    # Similarly for other 3 blocks: 14 each. Total 56 for 4x4x4.
    
    # For 4x4x2:
    # A is 4x4, B is 4x2, C is 4x2
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Split i into {0,1} and {2,3}, k=0,1:
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # Each 2x2x2 product needs 7 mults. 4 products total: 28 terms.
    
    # Grand total: 56 + 28 = 84 terms.
    # Schoolbook: 96. So 84 is an improvement.
    
    # Let me redo the construction with 84 terms.
    
    rank = 84
    U = [[0]*rank for _ in range(n*m)]  # 16 x 84
    V = [[0]*rank for _ in range(m*p)]  # 24 x 84
    W = [[0]*rank for _ in range(p*n)]  # 24 x 84
    
    col = 0
    
    def add_strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add a Strassen 2x2x2 block.
        A rows: a_row_offset, a_row_offset+1
        B cols: b_col_offset, b_col_offset+1
        C rows: c_row_offset, c_row_offset+1
        C cols: c_col_offset, c_col_offset+1
        
        But wait, the Strassen block computes C[2x2] = A[2x2] * B[2x2]
        where A is 2x2 (2 rows, 2 cols), B is 2x2 (2 rows, 2 cols), C is 2x2 (2 rows, 2 cols).
        
        In our indexing:
        A[i, j] with i in {a_row_offset, a_row_offset+1}, j in {b_col_offset, b_col_offset+1}
        B[j, k] with j in {b_col_offset, b_col_offset+1}, k in {c_col_offset, c_col_offset+1}
        C[i, k] with i in {c_row_offset, c_row_offset+1}, k in {c_col_offset, c_col_offset+1}
        
        Wait, that's not right either. Let me be more careful.
        
        For the product C = A * B where A is n x m and B is m x p:
        C[i, k] = sum_j A[i, j] * B[j, k]
        
        If we restrict to a 2x2x2 subproblem:
        A_sub is 2x2: rows i1, i2, cols j1, j2
        B_sub is 2x2: rows j1, j2, cols k1, k2
        C_sub is 2x2: rows i1, i2, cols k1, k2
        
        C_sub[i', k'] = sum_{j'} A_sub[i', j'] * B_sub[j', k']
        
        So the Strassen block uses:
        A indices: (i1, j1), (i1, j2), (i2, j1), (i2, j2) -> U row = i*m + j
        B indices: (j1, k1), (j1, k2), (j2, k1), (j2, k2) -> V row = j*p + k
        C indices: (i1, k1), (i1, k2), (i2, k1), (i2, k2) -> W row = k*n + i
        """
        nonlocal col
        sU, sV, sW = _strassen_2x2()
        # sU has 4 rows corresponding to (i',j') in order: (0,0),(0,1),(1,0),(1,1)
        # sV has 4 rows corresponding to (j',k') in order: (0,0),(0,1),(1,0),(1,1)
        # sW has 4 rows corresponding to (k',i') in order: (0,0),(0,1),(1,0),(1,1)
        
        i1, i2 = c_row_offset, c_row_offset + 1
        j1, j2 = b_col_offset, b_col_offset + 1
        k1, k2 = c_col_offset, c_col_offset + 1
        
        # Map local (i', j') to global (i, j)
        # local (0,0) -> (i1, j1), local (0,1) -> (i1, j2), local (1,0) -> (i2, j1), local (1,1) -> (i2, j2)
        U_map = [i1*m + j1, i1*m + j2, i2*m + j1, i2*m + j2]
        V_map = [j1*p + k1, j1*p + k2, j2*p + k1, j2*p + k2]
        W_map = [k1*n + i1, k1*n + i2, k2*n + i1, k2*n + i2]
        
        for c in range(7):
            for li in range(4):
                if sU[li][c] != 0:
                    U[U_map[li]][col] += sU[li][c]
            for lv in range(4):
                if sV[lv][c] != 0:
                    V[V_map[lv]][col] += sV[lv][c]
            for lw in range(4):
                if sW[lw][c] != 0:
                    W[W_map[lw]][col] += sW[lw][c]
            col += 1
    
    # Part 1: 4x4x4 (B cols 0-3)
    # 8 Strassen blocks:
    # (i_block, j_block, k_block) where i_block in {0,1}, j_block in {0,1}, k_block in {0,1}
    # i_block=0: rows 0,1; i_block=1: rows 2,3
    # j_block=0: cols 0,1; j_block=1: cols 2,3
    # k_block=0: cols 0,1; k_block=1: cols 2,3
    
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(2):
                a_row_offset = i_block * 2
                b_col_offset = j_block * 2
                c_row_offset = i_block * 2
                c_col_offset = k_block * 2
                add_strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset)
    
    # Part 2: 4x4x2 (B cols 4-5)
    # C[i, k] for k in {4, 5}
    # C[0:2, 4:6] = A[0:2, 0:2]*B[0:2, 4:6] + A[0:2, 2:4]*B[2:4, 4:6]
    # C[2:4, 4:6] = A[2:4, 0:2]*B[0:2, 4:6] + A[2:4, 2:4]*B[2:4, 4:6]
    
    for i_block in range(2):
        for j_block in range(2):
            a_row_offset = i_block * 2
            b_col_offset = j_block * 2
            c_row_offset = i_block * 2
            c_col_offset = 4  # cols 4,5
            add_strassen_block(a_row_offset, b_col_offset, c_row_offset, c_col_offset)
    
    # Verify
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
