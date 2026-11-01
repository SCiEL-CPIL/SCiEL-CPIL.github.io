import numpy as np
import time

def construct():
    """
    Rank-50 decomposition for 2x3 * 3x9 matrix multiplication.
    
    Strategy:
    - Split the 3x9 matrix B into three 3x3 blocks: B = [B1, B2, B3]
    - For each block, compute C_k = A * B_k where A is 2x3 and B_k is 3x3
    - Split A into A_left (2x2) and A_right (2x1)
    - Split B_k into B_k_left (3x2) and B_k_right (3x1)
    - C_k = A_left * B_k_left + A_right * B_k_right
    - For A_left * B_k_left (2x2 * 2x3): use Strassen-like rank-10 decomposition
    - For A_right * B_k_right (2x1 * 3x1): use schoolbook rank-6 decomposition
    - Total per block: 10 + 6 = 16, total for 3 blocks: 48
    - Wait, let me recalculate: 2x2 * 2x3 has rank 10? Let me use a safer approach.
    
    Actually, let me use a simpler approach:
    - Use Strassen for 2x2 * 2x2 (rank 7)
    - For the remaining columns, use schoolbook
    
    Let me try a different decomposition:
    - A is 2x3, B is 3x9
    - Split B into 3 blocks of 3x3
    - For each block, use a rank-16 decomposition for 2x3 * 3x3
    
    For 2x3 * 3x3:
    - A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    - B = [B1; B2] where B1 is 2x3, B2 is 1x3
    - C = A1*B1 + A2*B2
    - A1*B1 is 2x2 * 2x3: can be decomposed as (2x2*2x2) + (2x2*2x1)
      = rank 7 + rank 4 = rank 11
    - A2*B2 is 2x1 * 1x3 = 2x3: rank 6 (each element is a product)
    - Total: 11 + 6 = 17 per block, 51 total
    
    Let me use rank 51.
    """
    n, m, p = 2, 3, 9
    rank = 51
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    for block in range(3):
        # For each 3x3 block of B
        # A is 2x3, B_block is 3x3, C_block is 2x3
        
        # Part 1: 2x2 * 2x2 part using Strassen (rank 7)
        # A1 = A[:, 0:2] (2x2), B1 = B_block[0:2, :] (2x3)
        # We decompose B1 = [B1a, B1b] where B1a is 2x2, B1b is 2x1
        # A1 * B1a: 2x2 * 2x2, rank 7
        # A1 * B1b: 2x2 * 2x1, rank 4
        # A2 * B2: 2x1 * 1x3, rank 6
        
        # Strassen for 2x2 * 2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)f
        # m3 = a(e-g)
        # m4 = d(h-g)
        # m5 = (a+b)h
        # m6 = (c-d)(e+f)
        # m7 = (b-c)(g+h)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # For A1 * B1a where B1a is 2x2:
        # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B1a = [[B[0, block*3+0], B[0, block*3+1]], [B[1, block*3+0], B[1, block*3+1]]]
        
        # m1 = (A[0,0] + A[1,1]) * (B[0, block*3+0] + B[1, block*3+1])
        U[0, col] = 1  # A[0,0]
        U[3, col] = 1  # A[1,1]
        V[0 * 9 + block*3+0, col] = 1  # B[0, block*3+0]
        V[1 * 9 + block*3+1, col] = 1  # B[1, block*3+1]
        W[(block*3+0) * 2 + 0, col] = 1  # C[0, block*3+0]
        W[(block*3+0) * 2 + 1, col] = 1  # C[1, block*3+0]
        W[(block*3+1) * 2 + 0, col] = 1  # C[0, block*3+1]
        W[(block*3+1) * 2 + 1, col] = 1  # C[1, block*3+1]
        col += 1
        
        # m2 = (A[1,0] + A[1,1]) * B[0, block*3+1]
        U[3, col] = 1  # A[1,0]
        U[4, col] = 1  # A[1,1]
        V[0 * 9 + block*3+1, col] = 1  # B[0, block*3+1]
        W[(block*3+0) * 2 + 1, col] = 1  # C[0, block*3+1]
        W[(block*3+1) * 2 + 1, col] = 1  # C[1, block*3+1]
        col += 1
        
        # m3 = A[0,0] * (B[0, block*3+0] - B[1, block*3+0])
        U[0, col] = 1  # A[0,0]
        V[0 * 9 + block*3+0, col] = 1  # B[0, block*3+0]
        V[1 * 9 + block*3+0, col] = -1  # B[1, block*3+0]
        W[(block*3+0) * 2 + 0, col] = 1  # C[0, block*3+0]
        W[(block*3+0) * 2 + 1, col] = 1  # C[0, block*3+1]
        W[(block*3+1) * 2 + 1, col] = 1  # C[1, block*3+1]
        col += 1
        
        # m4 = A[1,1] * (B[1, block*3+1] - B[1, block*3+0])
        U[4, col] = 1  # A[1,1]
        V[1 * 9 + block*3+1, col] = 1  # B[1, block*3+1]
        V[1 * 9 + block*3+0, col] = -1  # B[1, block*3+0]
        W[(block*3+0) * 2 + 0, col] = 1  # C[0, block*3+0]
        W[(block*3+1) * 2 + 0, col] = 1  # C[1, block*3+0]
        W[(block*3+1) * 2 + 1, col] = 1  # C[1, block*3+1]
        col += 1
        
        # m5 = (A[0,0] + A[0,1]) * B[1, block*3+1]
        U[0, col] = 1  # A[0,0]
        U[1, col] = 1  # A[0,1]
        V[1 * 9 + block*3+1, col] = 1  # B[1, block*3+1]
        W[(block*3+0) * 2 + 0, col] = -1  # C[0, block*3+0]
        W[(block*3+0) * 2 + 1, col] = 1  # C[0, block*3+1]
        col += 1
        
        # m6 = (A[1,0] - A[0,0]) * (B[0, block*3+0] + B[0, block*3+1])
        U[3, col] = 1  # A[1,0]
        U[0, col] = -1  # A[0,0]
        V[0 * 9 + block*3+0, col] = 1  # B[0, block*3+0]
        V[0 * 9 + block*3+1, col] = 1  # B[0, block*3+1]
        W[(block*3+1) * 2 + 1, col] = 1  # C[1, block*3+1]
        col += 1
        
        # m7 = (A[0,1] - A[1,0]) * (B[1, block*3+0] + B[1, block*3+1])
        U[1, col] = 1  # A[0,1]
        U[3, col] = -1  # A[1,0]
        V[1 * 9 + block*3+0, col] = 1  # B[1, block*3+0]
        V[1 * 9 + block*3+1, col] = 1  # B[1, block*3+1]
        W[(block*3+0) * 2 + 0, col] = 1  # C[0, block*3+0]
        col += 1
        
        # Part 2: 2x2 * 2x1 part (rank 4)
        # A1 * B1b where B1b is 2x1: [[B[0, block*3+2]], [B[1, block*3+2]]]
        # C[:, block*3+2] = A1 * B1b
        # C[0, block*3+2] = A[0,0]*B[0, block*3+2] + A[0,1]*B[1, block*3+2]
        # C[1, block*3+2] = A[1,0]*B[0, block*3+2] + A[1,1]*B[1, block*3+2]
        
        # m8 = A[0,0] * B[0, block*3+2]
        U[0, col] = 1
        V[0 * 9 + block*3+2, col] = 1
        W[(block*3+2) * 2 + 0, col] = 1
        col += 1
        
        # m9 = A[0,1] * B[1, block*3+2]
        U[1, col] = 1
        V[1 * 9 + block*3+2, col] = 1
        W[(block*3+2) * 2 + 0, col] = 1
        col += 1
        
        # m10 = A[1,0] * B[0, block*3+2]
        U[3, col] = 1
        V[0 * 9 + block*3+2, col] = 1
        W[(block*3+2) * 2 + 1, col] = 1
        col += 1
        
        # m11 = A[1,1] * B[1, block*3+2]
        U[4, col] = 1
        V[1 * 9 + block*3+2, col] = 1
        W[(block*3+2) * 2 + 1, col] = 1
        col += 1
        
        # Part 3: 2x1 * 1x3 part (rank 6)
        # A2 * B2 where A2 = [[A[0,2]], [A[1,2]]], B2 = [[B[2, block*3+0], B[2, block*3+1], B[2, block*3+2]]]
        # C[0, block*3+j] += A[0,2] * B[2, block*3+j]
        # C[1, block*3+j] += A[1,2] * B[2, block*3+j]
        
        for j in range(3):
            # m = A[0,2] * B[2, block*3+j]
            U[2, col] = 1
            V[2 * 9 + block*3+j, col] = 1
            W[(block*3+j) * 2 + 0, col] = 1
            col += 1
            
            # m = A[1,2] * B[2, block*3+j]
            U[5, col] = 1
            V[2 * 9 + block*3+j, col] = 1
            W[(block*3+j) * 2 + 1, col] = 1
            col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
