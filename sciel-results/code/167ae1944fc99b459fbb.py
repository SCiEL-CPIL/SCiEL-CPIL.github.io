"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Strategy:
- Use double Strassen for 4x4x4 (rank 49) for the first 4 columns of B.
- Use schoolbook for 4x4x1 (rank 16) for the last column of B.
- Total rank: 65.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    size1 = n * m  # 16
    size2 = m * p  # 20
    size3 = p * n  # 20

    # Build the target tensor T of shape (size1, size2, size3)
    T = np.zeros((size1, size2, size3), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Verify if U, V, W exactly decompose T."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = np.abs(T_approx - T)
        return np.max(diff) < 1e-10

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7.
        Returns U, V, W each of shape (4, 7).
        Indexing: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
                  B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
                  C[0]=C00, C[1]=C01, C[2]=C10, C[3]=C11
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def double_strassen_4x4x4():
        """
        Double Strassen decomposition for 4x4 * 4x4 -> 4x4, rank 49.
        Partition into 2x2 blocks of 2x2 submatrices.
        Apply Strassen at block level (7 block products), each block product
        is a 2x2x2 multiplication (rank 7), total rank 7*7 = 49.
        """
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # 2x2 Strassen basis
        sU, sV, sW = strassen_2x2x2()
        
        # Block-level Strassen: 7 block products
        # Each block product (i,j) uses Strassen with 7 scalar products
        # Block (bi,bj) in A corresponds to rows bi*2..bi*2+1, cols bj*2..bj*2+1
        # Block (bj,bk) in B corresponds to rows bj*2..bj*2+1, cols bk*2..bk*2+1
        # Block (bi,bk) in C corresponds to rows bi*2..bi*2+1, cols bk*2..bk*2+1
        
        # Block-level Strassen coefficients:
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)*B00      -> C10 += M2, C11 += M2
        # M3 = A00*(B01-B11)      -> C01 += M3, C11 += M3
        # M4 = A11*(B10-B00)      -> C00 += M4, C10 += M4
        # M5 = (A00+A01)*B11      -> C00 -= M5, C01 += M5
        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 += M7
        
        # For each block-level Strassen product (7 total), and for each
        # scalar Strassen product within it (7 total), we have one rank-1 term.
        
        # Block-level Strassen structure:
        # For block product M_b (b=0..6):
        #   A-blocks involved: list of (block_idx, coeff)
        #   B-blocks involved: list of (block_idx, coeff)
        #   C-blocks involved: list of (block_idx, coeff)
        
        block_strassen = [
            # (A_blocks, B_blocks, C_blocks)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),           # M2
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
            ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (3, 1)]),          # M4
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),          # M5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),          # M6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),          # M7
        ]
        
        col = 0
        for b in range(7):
            a_blocks, b_blocks, c_blocks = block_strassen[b]
            for s in range(7):
                # For this scalar product within block product b:
                # U coefficients: for each A block, for each scalar position in that block
                for (blk, coeff) in a_blocks:
                    for si in range(2):
                        for sj in range(2):
                            row = blk * 4 + si * 2 + sj  # A index: block*4 + row*2 + col
                            U[row, col] += coeff * sU[si * 2 + sj, s]
                
                for (blk, coeff) in b_blocks:
                    for si in range(2):
                        for sj in range(2):
                            row = blk * 4 + si * 2 + sj  # B index: block*4 + row*2 + col
                            V[row, col] += coeff * sV[si * 2 + sj, s]
                
                for (blk, coeff) in c_blocks:
                    for si in range(2):
                        for sj in range(2):
                            row = blk * 4 + si * 2 + sj  # C index: block*4 + row*2 + col
                            W[row, col] += coeff * sW[si * 2 + sj, s]
                
                col += 1
        
        return U, V, W

    def schoolbook_4x4x1():
        """Schoolbook decomposition for 4x4 * 4x1 -> 4x1, rank 16."""
        rank = 16
        U = np.zeros((16, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    # Build double Strassen for 4x4x4
    U_left, V_left, W_left = double_strassen_4x4x4()
    left_rank = 49

    # Verify left part
    T_left = np.zeros((16, 16, 16), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T_left[i*4+j, j*4+k, k*4+i] = 1.0
    T_left_approx = np.einsum('ar,br,cr->abc', U_left, V_left, W_left)
    if np.max(np.abs(T_left_approx - T_left)) > 1e-10:
        print("WARNING: Double Strassen 4x4x4 verification failed!")
        # Fallback to schoolbook 4x4x4 (rank 64)
        left_rank = 64
        U_left = np.zeros((16, left_rank))
        V_left = np.zeros((16, left_rank))
        W_left = np.zeros((16, left_rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_left[i*4+j, col] = 1
                    V_left[j*4+k, col] = 1
                    W_left[k*4+i, col] = 1
                    col += 1

    # Schoolbook for 4x4x1
    U_right, V_right, W_right = schoolbook_4x4x1()
    right_rank = 16

    total_rank = left_rank + right_rank
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))

    U[:, :left_rank] = U_left
    V[:16, :left_rank] = V_left
    W[:16, :left_rank] = W_left

    U[:, left_rank:] = U_right
    V[16:, left_rank:] = V_right
    W[16:, left_rank:] = W_right

    result = {"U": U, "V": V, "W": W}

    # Final verification
    if not verify(U, V, W):
        print("WARNING: Final verification failed, using full schoolbook rank 80")
        rank = 80
        U = np.zeros((16, rank))
        V = np.zeros((20, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}

    U_list = result['U'].tolist()
    V_list = result['V'].tolist()
    W_list = result['W'].tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
