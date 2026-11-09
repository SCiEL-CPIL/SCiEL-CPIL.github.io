"""Matrix multiplication 3x3x5 with rank-minimal exact tensor decomposition."""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication.
    
    Strategy: Block decomposition using Strassen for 2x2x2 blocks.
    Split n=3 into {0,1}|{2}, m=3 into {0,1}|{2}, p=5 into {0,1}|{2,3,4}
    
    Sub-blocks and their ranks:
    - (2,2,2): rank 7 (Strassen)
    - (2,2,3): rank 12 (schoolbook)
    - (2,1,2): rank 4 (schoolbook)
    - (2,1,3): rank 6 (schoolbook)
    - (1,2,2): rank 4 (schoolbook)
    - (1,2,3): rank 6 (schoolbook)
    - (1,1,2): rank 2 (schoolbook)
    - (1,1,3): rank 3 (schoolbook)
    
    Total: 7 + 12 + 4 + 6 + 4 + 6 + 2 + 3 = 44
    
    Actually, let me try a better split: p=5 into {0,1}|{2,3}|{4}
    Split n=3 into {0,1}|{2}, m=3 into {0,1}|{2}, p=5 into {0,1}|{2,3}|{4}
    
    Sub-blocks:
    - (2,2,2): rank 7 (Strassen) for i in {0,1}, j in {0,1}, k in {0,1}
    - (2,2,2): rank 7 (Strassen) for i in {0,1}, j in {0,1}, k in {2,3}
    - (2,2,1): rank 4 (schoolbook) for i in {0,1}, j in {0,1}, k in {4}
    - (2,1,2): rank 4 (schoolbook) for i in {0,1}, j in {2}, k in {0,1}
    - (2,1,2): rank 4 (schoolbook) for i in {0,1}, j in {2}, k in {2,3}
    - (2,1,1): rank 2 (schoolbook) for i in {0,1}, j in {2}, k in {4}
    - (1,2,2): rank 4 (schoolbook) for i in {2}, j in {0,1}, k in {0,1}
    - (1,2,2): rank 4 (schoolbook) for i in {2}, j in {0,1}, k in {2,3}
    - (1,2,1): rank 2 (schoolbook) for i in {2}, j in {0,1}, k in {4}
    - (1,1,2): rank 2 (schoolbook) for i in {2}, j in {2}, k in {0,1}
    - (1,1,2): rank 2 (schoolbook) for i in {2}, j in {2}, k in {2,3}
    - (1,1,1): rank 1 (schoolbook) for i in {2}, j in {2}, k in {4}
    
    Total: 7 + 7 + 4 + 4 + 4 + 2 + 4 + 4 + 2 + 2 + 2 + 1 = 43
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Strassen decomposition for 2x2x2
    def strassen_222():
        """
        Strassen's algorithm for 2x2 matrix multiplication.
        Returns U, V, W for the 2x2x2 tensor with rank 7.
        
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        
        The 7 Strassen products:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B22)
        M4 = A22(B11-B12)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 4 = 2*2
        V = [[0]*rank for _ in range(4)]  # 4 = 2*2
        W = [[0]*rank for _ in range(4)]  # 4 = 2*2
        
        # Term 0: M1 = (A11+A22)(B11+B22)
        # U: A11 (row 0), A22 (row 3)
        U[0][0] = 1
        U[3][0] = 1
        # V: B11 (col 0), B21 (col 2)
        V[0][0] = 1
        V[2][0] = 1
        # W: C11 (row 0), C22 (row 3)
        W[0][0] = 1
        W[3][0] = 1
        
        # Term 1: M2 = (A21+A22)B11
        # U: A21 (row 2), A22 (row 3)
        U[2][1] = 1
        U[3][1] = 1
        # V: B11 (col 0)
        V[0][1] = 1
        # W: C21 (row 2)
        W[2][1] = 1
        
        # Term 2: M3 = A11(B21-B22)
        # U: A11 (row 0)
        U[0][2] = 1
        # V: B21 (col 2), B22 (col 3)
        V[2][2] = 1
        V[3][2] = -1
        # W: C12 (row 1), C22 (row 3)
        W[1][2] = 1
        W[3][2] = 1
        
        # Term 3: M4 = A22(B11-B12)
        # U: A22 (row 3)
        U[3][3] = 1
        # V: B11 (col 0), B12 (col 1)
        V[0][3] = 1
        V[1][3] = -1
        # W: C11 (row 0), C21 (row 2)
        W[0][3] = 1
        W[2][3] = 1
        
        # Term 4: M5 = (A11+A12)B22
        # U: A11 (row 0), A12 (row 1)
        U[0][4] = 1
        U[1][4] = 1
        # V: B22 (col 3)
        V[3][4] = 1
        # W: C11 (row 0), C12 (row 1)
        W[0][4] = -1
        W[1][4] = 1
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        # U: A21 (row 2), A11 (row 0)
        U[2][5] = 1
        U[0][5] = -1
        # V: B11 (col 0), B12 (col 1)
        V[0][5] = 1
        V[1][5] = 1
        # W: C22 (row 3)
        W[3][5] = 1
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        # U: A12 (row 1), A22 (row 3)
        U[1][6] = 1
        U[3][6] = -1
        # V: B21 (col 2), B22 (col 3)
        V[2][6] = 1
        V[3][6] = 1
        # W: C11 (row 0)
        W[0][6] = 1
        
        return U, V, W
    
    # Schoolbook decomposition for (ni, mj, pk)
    def schoolbook(ni, mj, pk):
        """Return U, V, W for schoolbook decomposition of ni x mj x pk matrix mult."""
        rank = ni * mj * pk
        U = [[0]*rank for _ in range(ni*mj)]
        V = [[0]*rank for _ in range(mj*pk)]
        W = [[0]*rank for _ in range(pk*ni)]
        col = 0
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    U[i*mj+j][col] = 1
                    V[j*pk+k][col] = 1
                    W[k*ni+i][col] = 1
                    col += 1
        return U, V, W
    
    # Define the block structure
    # n=3: i in {0,1} | {2}
    # m=3: j in {0,1} | {2}
    # p=5: k in {0,1} | {2,3} | {4}
    
    blocks = [
        # (i_range, j_range, k_range, ni, mj, pk, use_strassen)
        ((0,2), (0,2), (0,2), 2, 2, 2, True),    # Strassen
        ((0,2), (0,2), (2,4), 2, 2, 2, True),    # Strassen
        ((0,2), (0,2), (4,5), 2, 2, 1, False),   # Schoolbook
        ((0,2), (2,3), (0,2), 2, 1, 2, False),   # Schoolbook
        ((0,2), (2,3), (2,4), 2, 1, 2, False),   # Schoolbook
        ((0,2), (2,3), (4,5), 2, 1, 1, False),   # Schoolbook
        ((2,3), (0,2), (0,2), 1, 2, 2, False),   # Schoolbook
        ((2,3), (0,2), (2,4), 1, 2, 2, False),   # Schoolbook
        ((2,3), (0,2), (4,5), 1, 2, 1, False),   # Schoolbook
        ((2,3), (2,3), (0,2), 1, 1, 2, False),   # Schoolbook
        ((2,3), (2,3), (2,4), 1, 1, 2, False),   # Schoolbook
        ((2,3), (2,3), (4,5), 1, 1, 1, False),   # Schoolbook
    ]
    
    # Calculate total rank
    total_rank = 0
    for block in blocks:
        ni, mj, pk = block[3], block[4], block[5]
        if block[6]:  # use Strassen
            total_rank += 7
        else:
            total_rank += ni * mj * pk
    
    # Build the full decomposition
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for block in blocks:
        i_start, i_end, j_start, j_end, k_start, k_end, ni, mj, pk, use_strassen = block
        
        if use_strassen:
            sub_U, sub_V, sub_W = strassen_222()
            sub_rank = 7
        else:
            sub_U, sub_V, sub_W = schoolbook(ni, mj, pk)
            sub_rank = ni * mj * pk
        
        # Map local indices to global indices
        for li in range(ni):
            for lj in range(mj):
                for lk in range(pk):
                    local_u_row = li * mj + lj
                    local_v_row = lj * pk + lk
                    local_w_row = lk * ni + li
                    
                    global_i = i_start + li
                    global_j = j_start + lj
                    global_k = k_start + lk
                    
                    global_u_row = global_i * m + global_j
                    global_v_row = global_j * p + global_k
                    global_w_row = global_k * n + global_i
                    
                    for c in range(sub_rank):
                        U[global_u_row][col_offset + c] = sub_U[local_u_row][c]
                        V[global_v_row][col_offset + c] = sub_V[local_v_row][c]
                        W[global_w_row][col_offset + c] = sub_W[local_w_row][c]
        
        col_offset += sub_rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
