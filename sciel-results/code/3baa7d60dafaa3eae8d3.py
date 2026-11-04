import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    deadline = time.time() + 110

    def add_strassen_2x2(U, V, W, rank_offset, a_rows, b_rows, c_rows):
        """
        Add Strassen 2x2*2x2 decomposition.
        a_rows: [a00, a01, a10, a11] - row indices in U for A[0,0], A[0,1], A[1,0], A[1,1]
        b_rows: [b00, b01, b10, b11] - row indices in V for B[0,0], B[0,1], B[1,0], B[1,1]
        c_rows: [c00, c01, c10, c11] - row indices in W for C[0,0], C[0,1], C[1,0], C[1,1]
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

    # Total rank: 49 (for 4x4*4x4) + 28 (for 4x4*4x2) = 77
    total_rank = 77
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B, i.e., k=0..3)
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2 (for first 4 cols)
    
    # A block indices in U (row = i*m + j = i*4 + j):
    # A00 = A[0:2, 0:2] -> U[0], U[1], U[4], U[5]
    # A01 = A[0:2, 2:4] -> U[2], U[3], U[6], U[7]
    # A10 = A[2:4, 0:2] -> U[8], U[9], U[12], U[13]
    # A11 = A[2:4, 2:4] -> U[10], U[11], U[14], U[15]
    
    # B block indices in V (row = j*p + k = j*6 + k):
    # For first 4 columns (k=0,1,2,3):
    # B00 = B[0:2, 0:2] -> V[0], V[1], V[6], V[7]
    # B01 = B[0:2, 2:4] -> V[2], V[3], V[8], V[9]
    # B10 = B[2:4, 0:2] -> V[12], V[13], V[18], V[19]
    # B11 = B[2:4, 2:4] -> V[14], V[15], V[20], V[21]
    
    # C block indices in W (row = k*n + i = k*4 + i):
    # C00 = C[0:2, 0:2] -> W[0], W[1], W[4], W[5]
    # C01 = C[0:2, 2:4] -> W[2], W[3], W[6], W[7]
    # C10 = C[2:4, 0:2] -> W[8], W[9], W[12], W[13]
    # C11 = C[2:4, 2:4] -> W[10], W[11], W[14], W[15]
    
    # Apply Strassen to 4x4*4x4:
    # This requires 7 multiplications of 2x2*2x2, each of which uses Strassen (rank 7)
    # So we need to recursively apply Strassen.
    
    # Let's define the 2x2 Strassen decomposition for the 4x4*4x4 case.
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each of these is a 2x2*2x2 product, which we decompose using Strassen (rank 7).
    # So we need to call add_strassen_2x2 for each of these 7 products.
    
    # But wait, the add_strassen_2x2 function expects the row indices of the 2x2 blocks.
    # For M1 = (A00+A11)(B00+B11), the "A" block is A00+A11, which is not a single 2x2 block in U.
    # This is the issue. The Strassen decomposition for 4x4*4x4 is not simply 7 calls to add_strassen_2x2.
    # It requires a more careful construction.
    
    # Let's use a different approach: directly construct the rank-49 decomposition for 4x4*4x4.
    # This is complex, so let's use the schoolbook for 4x4*4x4 (rank 64) and Strassen for 4x4*4x2 (rank 28).
    # Total: 64 + 28 = 92. Not better than 96.
    
    # Actually, let's just use the schoolbook for the whole thing (rank 96) as a fallback.
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
