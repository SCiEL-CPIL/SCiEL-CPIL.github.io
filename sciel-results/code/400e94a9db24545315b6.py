"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 88 decomposition
    # Top part (2x4x8) uses Strassen for 2x2x2 blocks.
    # Bottom part (1x4x8) uses Schoolbook.
    
    U = np.zeros((n * m, 88), dtype=np.float64)
    V = np.zeros((m * p, 88), dtype=np.float64)
    W = np.zeros((p * n, 88), dtype=np.float64)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    # Strassen 2x2x2 decomposition
    # Local indices:
    # A: [a00, a01, a10, a11] -> flat [0, 1, 2, 3]
    # B: [b00, b01, b10, b11] -> flat [0, 1, 2, 3]
    # C: [c00, c01, c10, c11] -> flat [0, 1, 2, 3]
    # 
    # 7 terms:
    # 1. (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    #    U: [1, 0, 0, 1], V: [1, 0, 0, 1], W: [1, 0, 0, 1]
    # 2. (a10+a11)b00 -> c10 += 1, c11 -= 1
    #    U: [0, 0, 1, 1], V: [1, 0, 0, 0], W: [0, 0, 1, -1]
    # 3. a00(b01-b11) -> c01 += 1, c11 += 1
    #    U: [1, 0, 0, 0], V: [0, 1, 0, -1], W: [0, 1, 0, 1]
    # 4. a11(b10-b00) -> c00 += 1, c10 += 1
    #    U: [0, 0, 0, 1], V: [0, 0, 1, -1], W: [1, 0, 1, 0]
    # 5. (a00+a01)b11 -> c00 -= 1, c01 += 1
    #    U: [1, 1, 0, 0], V: [0, 0, 0, 1], W: [-1, 1, 0, 0]
    # 6. (a10-a00)(b00+b01) -> c11 += 1, c01 += 1
    #    U: [-1, 0, 1, 0], V: [1, 1, 0, 0], W: [0, 1, 0, 1]
    # 7. (a01-a11)(b10+b11) -> c00 += 1, c10 -= 1
    #    U: [0, 1, 0, -1], V: [0, 0, 1, 1], W: [1, 0, -1, 0]

    strassen_terms = [
        ([(0, 1), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]),      # T1
        ([(1, 0), (1, 1)], [(0, 0)], [(1, 0), (1, 1)]),              # T2 (c10+=1, c11-=1 -> W[1,0]=1, W[1,1]=-1)
        ([(0, 0)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),              # T3
        ([(1, 1)], [(1, 0), (0, 0)], [(0, 0), (1, 0)]),              # T4
        ([(0, 0), (0, 1)], [(1, 1)], [(0, 0), (0, 1)]),              # T5 (c00-=1, c01+=1 -> W[0,0]=-1, W[0,1]=1)
        ([(1, 0), (0, 0)], [(0, 0), (0, 1)], [(1, 1), (0, 1)]),      # T6
        ([(0, 1), (1, 1)], [(1, 0), (1, 1)], [(0, 0), (1, 0)]),      # T7
    ]
    
    # Correcting the W signs for the specific terms based on Strassen formulas:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5 + M6
    # c10 = M2 + M4 - M7
    # c11 = M1 - M2 + M3 + M6
    
    # Let's re-derive the exact (U, V, W) contributions for each term.
    # The term is U_dot_a * V_dot_b * (W_dot_c).
    # So we add 1 to W[k] if c_k gets +1, -1 if c_k gets -1.
    
    # T1: (a00+a11)(b00+b11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 0, 0, 1] (c00+=1, c11+=1)
    
    # T2: (a10+a11)b00
    # U: [0, 0, 1, 1]
    # V: [1, 0, 0, 0]
    # W: [0, 0, 1, -1] (c10+=1, c11-=1)
    
    # T3: a00(b01-b11)
    # U: [1, 0, 0, 0]
    # V: [0, 1, 0, -1]
    # W: [0, 1, 0, 1] (c01+=1, c11+=1)
    
    # T4: a11(b10-b00)
    # U: [0, 0, 0, 1]
    # V: [0, 0, 1, -1]
    # W: [1, 0, 1, 0] (c00+=1, c10+=1)
    
    # T5: (a00+a01)b11
    # U: [1, 1, 0, 0]
    # V: [0, 0, 0, 1]
    # W: [-1, 1, 0, 0] (c00-=1, c01+=1)
    
    # T6: (a10-a00)(b00+b01)
    # U: [-1, 0, 1, 0]
    # V: [1, 1, 0, 0]
    # W: [0, 1, 0, 1] (c01+=1, c11+=1) -> Wait, c11 gets +1?
    # Formula: c11 = M1 - M2 + M3 + M6. Yes, +1.
    # c01 = M3 + M5 + M6. Yes, +1.
    
    # T7: (a01-a11)(b10+b11)
    # U: [0, 1, 0, -1]
    # V: [0, 0, 1, 1]
    # W: [1, 0, -1, 0] (c00+=1, c10-=1)

    # Let's hardcode these to be safe.
    # Format: (U_vec, V_vec, W_vec) where vectors are dicts or lists.
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        
        # Mapping local (r,c) to global index
        def A_idx(r, c): return (i0 + r) * m + (j0 + c)
        def B_idx(r, c): return (j0 + r) * p + (k0 + c)
        def C_idx(c, r): return (k0 + c) * n + (i0 + r) # Note: W index is k*n + i
        
        # T1
        U[ A_idx(0,0), col] = 1; U[ A_idx(1,1), col] = 1
        V[ B_idx(0,0), col] = 1; V[ B_idx(1,1), col] = 1
        W[ C_idx(0,0), col] = 1; W[ C_idx(1,1), col] = 1
        col += 1
        
        # T2
        U[ A_idx(1,0), col] = 1; U[ A_idx(1,1), col] = 1
        V[ B_idx(0,0), col] = 1
        W[ C_idx(0,1), col] = 1; W[ C_idx(1,1), col] = -1 # c10 is row 1, col 0 -> C_idx(0,1)? No.
        # C_idx(c, r) = (k0+c)*n + (i0+r).
        # c10 is c=0, r=1. So C_idx(0, 1).
        # c11 is c=1, r=1. So C_idx(1, 1).
        # Correcting T2 W: c10 += 1, c11 -= 1.
        W[ C_idx(0,1), col] = 1; W[ C_idx(1,1), col] = -1
        col += 1
        
        # T3
        U[ A_idx(0,0), col] = 1
        V[ B_idx(0,1), col] = 1; V[ B_idx(1,1), col] = -1
        W[ C_idx(1,0), col] = 1; W[ C_idx(1,1), col] = 1 # c01 is c=1, r=0. C_idx(1,0). c11 is c=1, r=1. C_idx(1,1).
        col += 1
        
        # T4
        U[ A_idx(1,1), col] = 1
        V[ B_idx(1,0), col] = 1; V[ B_idx(0,0), col] = -1
        W[ C_idx(0,0), col] = 1; W[ C_idx(0,1), col] = 1 # c00 is c=0, r=0. C_idx(0,0). c10 is c=0, r=1. C_idx(0,1).
        col += 1
        
        # T5
        U[ A_idx(0,0), col] = 1; U[ A_idx(0,1), col] = 1
        V[ B_idx(1,1), col] = 1
        W[ C_idx(0,0), col] = -1; W[ C_idx(1,0), col] = 1 # c00 -= 1, c01 += 1.
        col += 1
        
        # T6
        U[ A_idx(1,0), col] = 1; U[ A_idx(0,0), col] = -1
        V[ B_idx(0,0), col] = 1; V[ B_idx(0,1), col] = 1
        W[ C_idx(1,0), col] = 1; W[ C_idx(1,1), col] = 1 # c01 += 1, c11 += 1.
        col += 1
        
        # T7
        U[ A_idx(0,1), col] = 1; U[ A_idx(1,1), col] = -1
        V[ B_idx(1,0), col] = 1; V[ B_idx(1,1), col] = 1
        W[ C_idx(0,0), col] = 1; W[ C_idx(0,1), col] = -1 # c00 += 1, c10 -= 1.
        col += 1

    # 1. Bottom part: Row 2 of A (1x4) * B (4x8). Rank 32.
    for j in range(m):
        for k in range(p):
            # A[2, j] * B[j, k] -> C[2, k]
            add_term(2 * m + j, j * p + k, k * n + 2, 1.0)
    
    # 2. Top part: Rows 0,1 of A (2x4) * B (4x8). Rank 56.
    # Split A into A1 (cols 0,1) and A2 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # A1 * B1 (2x2 A, 2x8 B) -> Rank 28.
    # A2 * B1 (2x2 A, 2x8 B) -> Rank 28.
    # A1 * B2 (2x2 A, 2x8 B) -> Rank 28.
    # A2 * B2 (2x2 A, 2x8 B) -> Rank 28.
    # Total 112? 
    # Wait. 2x2 A * 2x8 B.
    # Split B into B_left (2x4) and B_right (2x4).
    # A * B_left (2x2x4) -> Rank 14 (2 * Strassen 2x2x2).
    # A * B_right (2x2x4) -> Rank 14.
    # Total 28.
    # So 4 blocks * 28 = 112.
    # Plus bottom 32 = 144?
    # The previous calculation said 88. Let's re-verify.
    # Ah, I see. The "inspiration" said 88.
    # "Using Strassen's algorithm on 2x2x2 blocks reduces the rank of the 2x2x8 subproblems from 32 to 28"
    # Wait. 2x2x8 schoolbook is 2*2*8 = 32.
    # Strassen 2x2x2 is 7.
    # 2x2x4 is two 2x2x2 blocks? No, 2x2 A * 2x4 B.
    # If we split B into 2x2 blocks, we have two 2x2x2 multiplications.
    # Rank 7 + 7 = 14.
    # 2x2x8 is two 2x2x4 multiplications. Rank 14 + 14 = 28.
    # So 2x2x8 rank is 28.
    
    # Now, how many 2x2x8 subproblems are there in 2x4x8?
    # A is 2x4. B is 4x8.
    # Split A into A1 (2x2), A2 (2x2).
    # Split B into B1 (2x8), B2 (2x8).
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2x8. Rank 28.
    # A2*B2 is 2x2x8. Rank 28.
    # Total Top Rank = 56.
    # Bottom Rank = 32.
    # Total Rank = 88.
    
    # Okay, my previous manual count of 4 blocks was wrong.
    # A is 2x4. Splitting A into 2x2s means we use the columns of A.
    # But B is 4x8. The rows of B correspond to the columns of A.
    # So A1 uses cols 0,1 of A and rows 0,1 of B.
    # A2 uses cols 2,3 of A and rows 2,3 of B.
    # So there are only 2 blocks of type 2x2x8.
    # Not 4.
    
    # Block 1: A cols 0,1. B rows 0,1.
    # A1 is 2x2. B1 is 2x8.
    # Split B1 into B1a (cols 0-3) and B1b (cols 4-7).
    # A1 * B1a is 2x2x4. Rank 14.
    # A1 * B1b is 2x2x4. Rank 14.
    # Total 28.
    
    # Block 2: A cols 2,3. B rows 2,3.
    # A2 is 2x2. B2 is 2x8.
    # Split B2 into B2a (cols 0-3) and B2b (cols 4-7).
    # A2 * B2a is 2x2x4. Rank 14.
    # A2 * B2b is 2x2x4. Rank 14.
    # Total 28.
    
    # Total Top = 56.
    # Total = 56 + 32 = 88.
    
    # Implementation:
    
    # Block 1: i in [0,1], j in [0,1], k in [0..7]
    # k in [0..3]:
    add_strassen_block(0, 0, 0) # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 2) # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # k in [4..7]:
    add_strassen_block(0, 0, 4) # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 0, 6) # A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    
    # Block 2: i in [0,1], j in [2,3], k in [0..7]
    # k in [0..3]:
    add_strassen_block(0, 2, 0) # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 2) # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # k in [4..7]:
    add_strassen_block(0, 2, 4) # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 2, 6) # A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
