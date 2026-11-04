"""Matrix multiplication tensor decomposition for 2x4 by 4x6."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """Return rank-42 tensor decomposition for multiplying a 2x4 by 4x6 matrix.
    
    Uses block structure: split the 4x6 matrix into three 4x2 blocks along columns.
    For each 4x2 block, multiply (2x2) x (2x2) using Strassen's algorithm (rank 7).
    Total rank = 3 * 7 = 21 per half-block... wait, need to be more careful.
    
    Actually: A is 2x4, B is 4x6.
    Split A into A1 (2x2) and A2 (2x2) along columns.
    Split B into B1 (4x2), B2 (4x2), B3 (4x2) along columns.
    C = A1*B1 + A2*B2 (for cols 0-1)
    C = A1*B2 + A2*B3 (for cols 2-3) -- wait, this isn't right either.
    
    Let me reconsider. C[i,k] = sum_j A[i,j] * B[j,k].
    j goes from 0 to 3.
    k goes from 0 to 5.
    i goes from 0 to 1.
    
    Split j into j1 in {0,1} and j2 in {2,3}.
    Split k into k1 in {0,1}, k2 in {2,3}, k3 in {4,5}.
    
    C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    C[:, 4:6] = A[:, 0:2] @ B[0:2, 4:6] + A[:, 2:4] @ B[2:4, 4:6]
    
    Each term is a (2x2) x (2x2) product. Using Strassen, each needs rank 7.
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen's 7 multiplications for 2x2 x 2x2
    # For matrices X (2x2) and Y (2x2):
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X11-X10)*(Y10+Y11)
    # Result:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # We need to construct U, V, W for the full 2x4 * 4x6 product.
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen forms for X (left factor, 2x2) and Y (right factor, 2x2)
    # U encodes linear combinations of X entries
    # V encodes linear combinations of Y entries
    # W encodes how M1..M7 combine to give Z entries
    
    # For a single Strassen block multiplying X (rows r1,r2) and Y (cols c1,c2):
    # X entry (a,b) where a in {0,1}, b in {0,1} maps to U index a*2+b (within the 2x2 block)
    # Y entry (a,b) where a in {0,1}, b in {0,1} maps to V index a*2+b (within the 2x2 block)
    # Z entry (a,b) maps to W index a*p+b = a*6+b... wait, W[k*n+i] where k is col, i is row
    # So W index for output C[i,k] is k*n + i = k*2 + i
    
    # Strassen left forms (coefficients for X entries):
    # M1: X00+X11 -> coeffs [1, 0, 0, 1] for X00,X01,X10,X11
    # M2: X10+X11 -> [0, 0, 1, 1]
    # M3: X00      -> [1, 0, 0, 0]
    # M4: X11      -> [0, 0, 0, 1]
    # M5: X00+X01 -> [1, 1, 0, 0]
    # M6: X10-X00 -> [-1, 0, 1, 0]
    # M7: X11-X10 -> [0, 0, -1, 1]
    
    strassen_left = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 0, -1, 1],  # M7
    ]
    
    # Strassen right forms (coefficients for Y entries):
    # M1: Y00+Y11 -> [1, 0, 0, 1]
    # M2: Y00     -> [1, 0, 0, 0]
    # M3: Y01-Y11 -> [0, 1, 0, -1]
    # M4: Y10-Y00 -> [-1, 0, 1, 0]
    # M5: Y11     -> [0, 0, 0, 1]
    # M6: Y00+Y01 -> [1, 1, 0, 0]
    # M7: Y10+Y11 -> [0, 0, 1, 1]
    
    strassen_right = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # Strassen output combinations (how M1..M7 give Z00, Z01, Z10, Z11):
    # Z00 = M1+M4-M5+M7 -> [1, 0, 0, 1, -1, 0, 1]
    # Z01 = M3+M5       -> [0, 0, 1, 0, 1, 0, 0]
    # Z10 = M2+M4       -> [0, 1, 0, 1, 0, 0, 0]
    # Z11 = M1-M2+M3+M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    strassen_out = [
        [1, 0, 0, 1, -1, 0, 1],   # Z00
        [0, 0, 1, 0, 1, 0, 0],    # Z01
        [0, 1, 0, 1, 0, 0, 0],    # Z10
        [1, -1, 1, 0, 0, 1, 0],   # Z11
    ]
    
    # Now, for each of the 6 Strassen blocks:
    # Block (j_half, k_block) where j_half in {0,1}, k_block in {0,1,2}
    # j_half=0: A[:, 0:2] and B[0:2, k_block*2:(k_block+1)*2]
    # j_half=1: A[:, 2:4] and B[2:4, k_block*2:(k_block+1)*2]
    
    # U index: i*m + j where i in {0,1}, j in {0,1,2,3}
    # For j_half=0: j in {0,1}, so U indices: i*4+0, i*4+1
    # For j_half=1: j in {2,3}, so U indices: i*4+2, i*4+3
    
    # V index: j*p + k where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # For j_half=0: j in {0,1}, k_block maps to k in {0,1}, {2,3}, {4,5}
    #   V indices: j*6 + k for j in {0,1}, k in the block
    # For j_half=1: j in {2,3}, same k blocks
    #   V indices: j*6 + k for j in {2,3}, k in the block
    
    # W index: k*n + i where k in {0,1,2,3,4,5}, i in {0,1}
    # For k_block: k in {k_block*2, k_block*2+1}
    # W indices: k*2 + i for k in the block, i in {0,1}
    
    col = 0
    for j_half in range(2):
        for k_block in range(3):
            # Determine U row offsets
            # U rows for A[:, 2*j_half : 2*j_half+2]
            # A[i, 2*j_half + a] -> U index i*4 + 2*j_half + a
            u_rows = []
            for a in range(2):  # a = 0,1 corresponds to column within the 2x2 block
                for i in range(2):  # i = 0,1
                    u_rows.append(i * 4 + 2 * j_half + a)
            # u_rows[0] = A[0, 2*j_half+0], u_rows[1] = A[0, 2*j_half+1]
            # u_rows[2] = A[1, 2*j_half+0], u_rows[3] = A[1, 2*j_half+1]
            # Actually let me be more careful:
            # u_rows should be in order [X00, X01, X10, X11]
            # X00 = A[0, 2*j_half], X01 = A[0, 2*j_half+1]
            # X10 = A[1, 2*j_half], X11 = A[1, 2*j_half+1]
            u_rows = [
                0 * 4 + 2 * j_half + 0,  # X00
                0 * 4 + 2 * j_half + 1,  # X01
                1 * 4 + 2 * j_half + 0,  # X10
                1 * 4 + 2 * j_half + 1,  # X11
            ]
            
            # Determine V row offsets
            # V rows for B[2*j_half : 2*j_half+2, k_block*2 : k_block*2+2]
            # B[2*j_half + a, k_block*2 + b] -> V index (2*j_half + a)*6 + k_block*2 + b
            # V in order [Y00, Y01, Y10, Y11]:
            # Y00 = B[2*j_half, k_block*2], Y01 = B[2*j_half, k_block*2+1]
            # Y10 = B[2*j_half+1, k_block*2], Y11 = B[2*j_half+1, k_block*2+1]
            v_rows = [
                (2 * j_half + 0) * 6 + k_block * 2 + 0,  # Y00
                (2 * j_half + 0) * 6 + k_block * 2 + 1,  # Y01
                (2 * j_half + 1) * 6 + k_block * 2 + 0,  # Y10
                (2 * j_half + 1) * 6 + k_block * 2 + 1,  # Y11
            ]
            
            # Determine W row offsets
            # W rows for C[:, k_block*2 : k_block*2+2]
            # C[i, k_block*2 + b] -> W index (k_block*2 + b)*2 + i
            # W in order [Z00, Z01, Z10, Z11]:
            # Z00 = C[0, k_block*2], Z01 = C[0, k_block*2+1]
            # Z10 = C[1, k_block*2], Z11 = C[1, k_block*2+1]
            w_rows = [
                (k_block * 2 + 0) * 2 + 0,  # Z00
                (k_block * 2 + 1) * 2 + 0,  # Z01
                (k_block * 2 + 0) * 2 + 1,  # Z10
                (k_block * 2 + 1) * 2 + 1,  # Z11
            ]
            
            for s in range(7):
                # Set U coefficients
                for a in range(4):
                    U[u_rows[a]][col] = strassen_left[s][a]
                # Set V coefficients
                for a in range(4):
                    V[v_rows[a]][col] = strassen_right[s][a]
                # Set W coefficients
                for z in range(4):
                    W[w_rows[z]][col] = strassen_out[z][s]
                col += 1
    
    return {"U": U, "V": V, "W": W}
