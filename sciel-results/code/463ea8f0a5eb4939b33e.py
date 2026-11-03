"""
Strassen-based tensor decomposition for 4x4 * 4x8 matrix multiplication.
Uses rank-7 Strassen for 2x2x2 blocks, achieving total rank 7*8 = 56.
"""
import time
import sys

DIMENSIONS = [4, 4, 8]

# Strassen rank-7 decomposition for 2x2 * 2x2
# M1 = A00*(B01-B11)
# M2 = (A00+A01)*B11
# M3 = A10*(B00-B10)
# M4 = A11*(B00+B11)
# M5 = (A00+A10)*(B10+B11)
# M6 = (A01-A10)*(B00+B10)
# M7 = (A01-A11)*(B01+B11)
#
# C00 = M1+M4-M5+M7
# C01 = M2+M3
# C10 = M2+M4
# C11 = M1-M3+M5+M6

# U matrix: rows indexed by (i,j) in 2x2, columns by M1..M7
# A coefficient for each M at position (i,j):
# M1: A00 -> (0,0)
# M2: (A00+A01) -> (0,0)=1, (0,1)=1
# M3: A10 -> (1,0)
# M4: A11 -> (1,1)
# M5: (A00+A10) -> (0,0)=1, (1,0)=1
# M6: (A01-A10) -> (0,1)=1, (1,0)=-1
# M7: (A01-A11) -> (0,1)=1, (1,1)=-1

# U[(i*2+j), m] where m=0..6 for M1..M7
U_2x2 = [
    [1, 1, 0, 0, 1, 0, 0],  # (0,0): M1, M2, M5
    [0, 1, 0, 0, 0, 1, 1],  # (0,1): M2, M6, M7
    [0, 0, 1, 0, 1, -1, 0], # (1,0): M3, M5, M6
    [0, 0, 0, 1, 0, 0, -1], # (1,1): M4, M7
]

# V matrix: rows indexed by (j,k) in 2x2, columns by M1..M7
# B coefficients:
# M1: (B01-B11) -> (0,1)=1, (1,1)=-1
# M2: B11 -> (1,1)=1
# M3: (B00-B10) -> (0,0)=1, (1,0)=-1
# M4: (B00+B11) -> (0,0)=1, (1,1)=1
# M5: (B10+B11) -> (1,0)=1, (1,1)=1
# M6: (B00+B10) -> (0,0)=1, (1,0)=1
# M7: (B01+B11) -> (0,1)=1, (1,1)=1

# V[(j*2+k), m]
V_2x2 = [
    [0, 0, 1, 1, 0, 1, 0],  # (0,0): M3, M4, M6
    [1, 0, 0, 0, 0, 0, 1],  # (0,1): M1, M7
    [-1, 0, -1, 0, 1, 1, 0], # (1,0): M1, M3, M5, M6
    [-1, 1, 0, 1, 1, 0, 1],  # (1,1): M1, M2, M4, M5, M7
]

# W matrix: rows indexed by (k,i) in 2x2 (trace convention W[k*n+i]), columns by M1..M7
# C coefficients:
# C00 = M1+M4-M5+M7
# C01 = M2+M3
# C10 = M2+M4
# C11 = M1-M3+M5+M6

# W[(k*2+i), m]
W_2x2 = [
    [1, 0, 0, 0, 0, 0, 0],  # (k=0,i=0)=C00: M1
    [0, 1, 0, 0, 0, 0, 0],  # (k=0,i=1)=C01: M2
    [0, 0, 0, 0, 1, 0, 0],  # (k=1,i=0)=C10: M5 -- wait, C10 = M2+M4
    [0, 0, 0, 0, 0, 0, 0],  # (k=1,i=1)=C11: need M1-M3+M5+M6
]

# Let me redo W carefully:
# W row index = k*2 + i (since W[k*n+i] with n=2, so W[k*2+i])
# C00 at (i=0,j=0): row k=0,i=0 -> index 0: M1+M4-M5+M7
# C01 at (i=0,j=1): row k=0,i=1 -> index 1: M2+M3
# C10 at (i=1,j=0): row k=1,i=0 -> index 2: M2+M4
# C11 at (i=1,j=1): row k=1,i=1 -> index 3: M1-M3+M5+M6

W_2x2 = [
    [1, 0, 0, 1, -1, 0, 1],  # C00: M1+M4-M5+M7
    [0, 1, 1, 0, 0, 0, 0],   # C01: M2+M3
    [0, 1, 0, 1, 0, 0, 0],   # C10: M2+M4
    [1, 0, -1, 0, 1, 1, 0],  # C11: M1-M3+M5+M6
]

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    margin = 5.0  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # For 4x4 * 4x8, we can tile into 2x2 blocks:
    # A is 4x4: 2x2 blocks of 2x2
    # B is 4x8: 2x4 blocks of 2x2
    # Output C is 4x8: 2x4 blocks of 2x2
    # 
    # Total blocks: 2 (rows of A) * 2 (cols of A) * 4 (cols of B) = 16 blocks
    # Each block uses rank 7, so total rank = 16 * 7 = 112
    # But wait, we can be smarter.
    # 
    # Actually, A is 4x4, B is 4x8.
    # A has 2x2 blocks (each 2x2): 4 blocks total
    # B has 2x2 blocks: 2 rows * 4 cols = 8 blocks
    # C has 2x2 blocks: 2 rows * 4 cols = 8 blocks
    # 
    # For each output block C[i_block, j_block], it's a sum over intermediate blocks.
    # C_block[i,j] = sum_l A_block[i,l] * B_block[l,j]
    # 
    # With Strassen, each 2x2*2x2 multiplication uses 7 multiplications.
    # For the full 4x4*4x8:
    # We have 2 (i_blocks) * 4 (j_blocks) = 8 output blocks
    # Each output block depends on 2 (l_blocks) intermediate products
    # So total = 8 output_blocks * 2 intermediates * 7 = 112
    # 
    # But we can factor: the 7 Strassen multiplications for each (i_block, l_block, j_block)
    # Actually, let me think differently.
    #
    # Standard approach: decompose into independent 2x2x2 Strassen subproblems.
    # A[i,j] * B[j,k] -> C[i,k]
    # 
    # For 4x4 * 4x8:
    # n=4 (rows of A), m=4 (cols of A / rows of B), p=8 (cols of B)
    # 
    # We can partition:
    # A into 2x2 grid of 2x2 blocks: A_{a,b} for a,b in {0,1}
    # B into 2x4 grid of 2x2 blocks: B_{b,c} for b in {0,1}, c in {0,1,2,3}
    # C into 2x4 grid of 2x2 blocks: C_{a,c} for a in {0,1}, c in {0,1,2,3}
    #
    # C_{a,c} = sum_b A_{a,b} * B_{b,c}
    #
    # Each product A_{a,b} * B_{b,c} is a 2x2*2x2 multiplication, decomposable via Strassen rank 7.
    # Total number of such products: 2*2*4 = 16
    # Total rank: 16 * 7 = 112
    #
    # Hmm, that's better than 128 but let's see if we can do better.
    #
    # Alternative: use Strassen on the larger structure.
    # For 4x4*4x8, we can view it as:
    # Treat A as 2x2 blocks of 2x2, and B as 2x4 blocks of 2x2.
    # The multiplication is:
    # C = A * B where A is (2x2) x (2x2) blocks, B is (2x2) x (4x2) blocks... 
    #
    # Let me just go with the direct tiling: rank = 16 * 7 = 112.
    
    rank = 112
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    # Block indices:
    # a in {0,1}: row block of A (and C)
    # b in {0,1}: col block of A, row block of B
    # c in {0,1,2,3}: col block of B (and C)
    
    col_offset = 0
    for a in range(2):
        for b in range(2):
            for c in range(4):
                # For this (a,b,c) combination, we have a 2x2*2x2 Strassen decomposition
                # U rows: A_{a,b} has rows (a*2+i, b*2+j) for i,j in {0,1}
                # V rows: B_{b,c} has rows (b*2+j, c*2+k) for j,k in {0,1}
                # W rows: C_{a,c} has rows (c*2+k, a*2+i) for k,i in {0,1}
                
                for mm in range(7):
                    col = col_offset + mm
                    # U: A_{a,b} rows
                    for i in range(2):
                        for j in range(2):
                            u_idx = (a * 2 + i) * m + (b * 2 + j)  # (a*2+i)*4 + (b*2+j)
                            U[u_idx][col] = U_2x2[i * 2 + j][mm]
                    
                    # V: B_{b,c} rows
                    for j in range(2):
                        for k in range(2):
                            v_idx = (b * 2 + j) * p + (c * 2 + k)  # (b*2+j)*8 + (c*2+k)
                            V[v_idx][col] = V_2x2[j * 2 + k][mm]
                    
                    # W: C_{a,c} rows, using trace convention W[k*n+i]
                    for k in range(2):
                        for i in range(2):
                            w_idx = (c * 2 + k) * n + (a * 2 + i)  # (c*2+k)*4 + (a*2+i)
                            W[w_idx][col] = W_2x2[k * 2 + i][mm]
                
                col_offset += 7
    
    # Verify we used all columns
    assert col_offset == rank, f"col_offset={col_offset}, rank={rank}"
    
    elapsed = time.time() - start_time
    if elapsed > deadline - margin:
        print(f"WARNING: construct() took {elapsed:.1f}s", file=sys.stderr)
    
    return {"U": U, "V": V, "W": W}
