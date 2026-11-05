import time
import numpy as np

def construct():
    deadline = time.time() + 110
    n, m, p = 2, 4, 8
    
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # We use 8 Strassen blocks, each of rank 7.
    # Total rank = 56.
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    # Blocks definition:
    # A is 2x4. Split into A1 (cols 0-1) and A2 (cols 2-3).
    # B is 4x8. Split into 8 blocks of 2x2:
    # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    # B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    # C is 2x8.
    # C[:, 0:2] = A1 * B[0:2, 0:2] + A2 * B[2:4, 0:2]
    # C[:, 2:4] = A1 * B[0:2, 2:4] + A2 * B[2:4, 2:4]
    # C[:, 4:6] = A1 * B[0:2, 4:6] + A2 * B[2:4, 4:6]
    # C[:, 6:8] = A1 * B[0:2, 6:8] + A2 * B[2:4, 6:8]
    
    # For each of the 4 column-pairs in C, we have a sum of two 2x2x2 products.
    # We handle each 2x2x2 product with Strassen (rank 7).
    # So for each C-column-pair, we have 14 multiplications.
    # Total 4 * 14 = 56.
    
    col = 0
    for c_col_pair in range(4):
        # c_col_pair 0 -> cols 0,1
        # c_col_pair 1 -> cols 2,3
        # c_col_pair 2 -> cols 4,5
        # c_col_pair 3 -> cols 6,7
        
        c_k0 = 2 * c_col_pair
        c_k1 = 2 * c_col_pair + 1
        
        # Product 1: A1 * B_top
        # A1 is A[0:2, 0:2] -> indices 0,1,4,5
        # B_top is B[0:2, c_k0:c_k1+1] -> indices c_k0, c_k0+1, 8+c_k0, 8+c_k0+1
        a1_00 = 0; a1_01 = 1; a1_10 = 4; a1_11 = 5
        b1_00 = c_k0; b1_01 = c_k0+1; b1_10 = 8+c_k0; b1_11 = 8+c_k0+1
        c_00 = c_k0*2 + 0; c_01 = c_k0*2 + 1 # C[0, c_k0]
        c_10 = c_k0*2 + 1; c_11 = c_k0*2 + 2 # Wait, W index is k*n+i = k*2+i
        # C[0, k] is at index k*2 + 0
        # C[1, k] is at index k*2 + 1
        c_00_idx = c_k0 * 2
        c_01_idx = (c_k0 + 1) * 2
        c_10_idx = c_k0 * 2 + 1
        c_11_idx = (c_k0 + 1) * 2 + 1
        
        # Strassen for A1 * B1
        # P1 = (a00+a11)(b00+b11)
        U[a1_00, col] += 1; U[a1_11, col] += 1
        V[b1_00, col] += 1; V[b1_11, col] += 1
        W[c_00_idx, col] += 1; W[c_11_idx, col] += 1
        col += 1
        
        # P2 = (a10+a11)b00
        U[a1_10, col] += 1; U[a1_11, col] += 1
        V[b1_00, col] += 1
        W[c_10_idx, col] += 1; W[c_11_idx, col] -= 1
        col += 1
        
        # P3 = a00(b01-b11)
        U[a1_00, col] += 1
        V[b1_01, col] += 1; V[b1_11, col] -= 1
        W[c_01_idx, col] += 1; W[c_11_idx, col] += 1
        col += 1
        
        # P4 = a11(b10-b00)
        U[a1_11, col] += 1
        V[b1_10, col] += 1; V[b1_00, col] -= 1
        W[c_00_idx, col] += 1; W[c_10_idx, col] += 1
        col += 1
        
        # P5 = (a00+a01)b11
        U[a1_00, col] += 1; U[a1_01, col] += 1
        V[b1_11, col] += 1
        W[c_00_idx, col] -= 1; W[c_01_idx, col] += 1
        col += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[a1_10, col] += 1; U[a1_00, col] -= 1
        V[b1_00, col] += 1; V[b1_01, col] += 1
        W[c_11_idx, col] += 1
        col += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[a1_01, col] += 1; U[a1_11, col] -= 1
        V[b1_10, col] += 1; V[b1_11, col] += 1
        W[c_00_idx, col] += 1
        col += 1
        
        # Product 2: A2 * B_bot
        # A2 is A[0:2, 2:4] -> indices 2,3,6,7
        # B_bot is B[2:4, c_k0:c_k1+1] -> indices 16+c_k0, 16+c_k0+1, 24+c_k0, 24+c_k0+1
        a2_00 = 2; a2_01 = 3; a2_10 = 6; a2_11 = 7
        b2_00 = 16 + c_k0; b2_01 = 16 + c_k0 + 1
        b2_10 = 24 + c_k0; b2_11 = 24 + c_k0 + 1
        
        # C indices are the same: c_00_idx, c_01_idx, c_10_idx, c_11_idx
        
        # Strassen for A2 * B2
        # P8 = (a00+a11)(b00+b11)
        U[a2_00, col] += 1; U[a2_11, col] += 1
        V[b2_00, col] += 1; V[b2_11, col] += 1
        W[c_00_idx, col] += 1; W[c_11_idx, col] += 1
        col += 1
        
        # P9 = (a10+a11)b00
        U[a2_10, col] += 1; U[a2_11, col] += 1
        V[b2_00, col] += 1
        W[c_10_idx, col] += 1; W[c_11_idx, col] -= 1
        col += 1
        
        # P10 = a00(b01-b11)
        U[a2_00, col] += 1
        V[b2_01, col] += 1; V[b2_11, col] -= 1
        W[c_01_idx, col] += 1; W[c_11_idx, col] += 1
        col += 1
        
        # P11 = a11(b10-b00)
        U[a2_11, col] += 1
        V[b2_10, col] += 1; V[b2_00, col] -= 1
        W[c_00_idx, col] += 1; W[c_10_idx, col] += 1
        col += 1
        
        # P12 = (a00+a01)b11
        U[a2_00, col] += 1; U[a2_01, col] += 1
        V[b2_11, col] += 1
        W[c_00_idx, col] -= 1; W[c_01_idx, col] += 1
        col += 1
        
        # P13 = (a10-a00)(b00+b01)
        U[a2_10, col] += 1; U[a2_00, col] -= 1
        V[b2_00, col] += 1; V[b2_01, col] += 1
        W[c_11_idx, col] += 1
        col += 1
        
        # P14 = (a01-a11)(b10+b11)
        U[a2_01, col] += 1; U[a2_11, col] -= 1
        V[b2_10, col] += 1; V[b2_11, col] += 1
        W[c_00_idx, col] += 1
        col += 1
        
        if time.time() > deadline - 5:
            break
    
    assert col == 56, f"Col mismatch: {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
