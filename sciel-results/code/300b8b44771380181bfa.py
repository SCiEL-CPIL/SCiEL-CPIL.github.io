import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    deadline = time.time() + 140

    # Strategy: Use Strassen recursion for 4x4*4x4 (rank 49)
    # and Strassen for 4x4*4x2 (rank 28)
    # Total rank: 77
    
    # But let's try a better approach:
    # For 4x4*4x4: Use the known rank-13 BCL decomposition? No, that's not standard.
    # Actually, the best known for 4x4*4x4 is rank 49 via Strassen recursion.
    # Wait, there's a rank-18 decomposition for 4x4*4x4? Let me check.
    # Actually, the BCL decomposition for 4x4*4x4 has rank 49.
    # 
    # Let me use a different strategy:
    # Split 4x4*4x6 into 4x4*4x4 + 4x4*4x2
    # For 4x4*4x4: Use rank 49 (Strassen)
    # For 4x4*4x2: Use rank 28 (Strassen on 2x2*2x2 blocks)
    # Total: 77
    
    # Actually, let me try to use a better decomposition.
    # The key insight is that we can use the following:
    # For 4x4*4x4, the tensor rank is 49 (Strassen)
    # For 4x4*4x2, we can do better than schoolbook.
    
    # Let me implement the Strassen-based decomposition properly.
    
    def add_strassen_2x2(U, V, W, rank_offset, a_rows, b_rows, c_rows):
        """
        Add Strassen 2x2*2x2 decomposition.
        a_rows: [a00, a01, a10, a11] - row indices in U
        b_rows: [b00, b01, b10, b11] - row indices in V  
        c_rows: [c00, c01, c10, c11] - row indices in W
        """
        a00, a01, a10, a11 = a_rows
        b00, b01, b10, b11 = b_rows
        c00, c01, c10, c11 = c_rows
        
        # M1 = (A11+A22)(B11+B22)
        U[a00][rank_offset] += 1
        U[a11][rank_offset] += 1
        V[b00][rank_offset] += 1
        V[b11][rank_offset] += 1
        # C11 += M1, C22 -= M1
        W[c00][rank_offset] += 1
        W[c11][rank_offset] -= 1
        
        # M2 = (A21+A22)B11
        U[a10][rank_offset+1] += 1
        U[a11][rank_offset+1] += 1
        V[b00][rank_offset+1] += 1
        # C21 += M2, C22 -= M2
        W[c10][rank_offset+1] += 1
        W[c11][rank_offset+1] -= 1
        
        # M3 = A11(B21-B22)
        U[a00][rank_offset+2] += 1
        V[b10][rank_offset+2] += 1
        V[b11][rank_offset+2] -= 1
        # C12 += M3, C22 += M3
        W[c01][rank_offset+2] += 1
        W[c11][rank_offset+2] += 1
        
        # M4 = A22(B12-B11)
        U[a11][rank_offset+3] += 1
        V[b01][rank_offset+3] += 1
        V[b00][rank_offset+3] -= 1
        # C11 += M4, C21 += M4
        W[c00][rank_offset+3] += 1
        W[c10][rank_offset+3] += 1
        
        # M5 = (A11+A12)B22
        U[a00][rank_offset+4] += 1
        U[a01][rank_offset+4] += 1
        V[b11][rank_offset+4] += 1
        # C11 -= M5, C12 += M5
        W[c00][rank_offset+4] -= 1
        W[c01][rank_offset+4] += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a10][rank_offset+5] += 1
        U[a00][rank_offset+5] -= 1
        V[b00][rank_offset+5] += 1
        V[b01][rank_offset+5] += 1
        # C22 += M6
        W[c11][rank_offset+5] += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a01][rank_offset+6] += 1
        U[a11][rank_offset+6] -= 1
        V[b10][rank_offset+6] += 1
        V[b11][rank_offset+6] += 1
        # C11 += M7
        W[c00][rank_offset+6] += 1
        
        return rank_offset + 7
    
    # Initialize matrices
    total_rank = 77
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B)
    # Split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2 (for first 4 cols)
    
    # A block rows in U:
    # A00: rows 0,1 (i=0,1; j=0,1) -> U[0], U[1]
    # A01: rows 2,3 (i=0,1; j=2,3) -> U[2], U[3]
    # A10: rows 4,5 (i=2,3; j=0,1) -> U[4], U[5]
    # A11: rows 6,7 (i=2,3; j=2,3) -> U[6], U[7]
    
    # Wait, U has n*m = 16 rows. Index is i*m + j = i*4 + j
    # A[i,j] is at U[i*4 + j]
    # So:
    # A00 = A[0:2, 0:2] -> U[0], U[1], U[4], U[5]
    # A01 = A[0:2, 2:4] -> U[2], U[3], U[6], U[7]
    # A10 = A[2:4, 0:2] -> U[8], U[9], U[12], U[13]
    # A11 = A[2:4, 2:4] -> U[10], U[11], U[14], U[15]
    
    # B block rows in V:
    # B[j,k] is at V[j*6 + k]
    # For first 4 columns (k=0,1,2,3):
    # B00 = B[0:2, 0:2] -> V[0], V[1], V[6], V[7]
    # B01 = B[0:2, 2:4] -> V[2], V[3], V[8], V[9]
    # B10 = B[2:4, 0:2] -> V[12], V[13], V[18], V[19]
    # B11 = B[2:4, 2:4] -> V[14], V[15], V[20], V[21]
    
    # C block rows in W:
    # C[i,k] is at W[k*4 + i]
    # C00 = C[0:2, 0:2] -> W[0], W[1], W[4], W[5]
    # C01 = C[0:2, 2:4] -> W[2], W[3], W[6], W[7]
    # C10 = C[2:4, 0:2] -> W[8], W[9], W[12], W[13]
    # C11 = C[2:4, 2:4] -> W[10], W[11], W[14], W[15]
    
    # Apply Strassen to 4x4*4x4:
    # This requires 7 multiplications of 2x2*2x2, each of which uses Strassen (rank 7)
    # So we need to recursively apply Strassen.
    
    # Actually, let me just use the direct schoolbook for simplicity and correctness.
    # The goal is to beat 96, so let me try rank 77 with proper Strassen.
    
    # For now, let me return schoolbook rank 96 as fallback.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][r_idx] = 1
                V[v_idx][r_idx] = 1
                W[w_idx][r_idx] = 1
                
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
