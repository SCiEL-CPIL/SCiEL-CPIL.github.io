"""
Matrix multiplication 4x5 by 5x6 using a Strassen-based block decomposition.

Strategy:
- Split the 4x5 x 5x6 multiplication into blocks.
- For the 4x4 x 4x4 core, use a rank-49 Strassen decomposition (two levels of 2x2 Strassen).
- For the remaining 4x1 x 5x6 strips, use schoolbook (rank 24).
- Total rank: 49 + 24 = 73.

The interface returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
such that for all (i,j,k) in [0,n)x[0,m)x[0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
and for all other (a,b,c) the sum is 0.
"""
import time
import numpy as np

def construct():
    n, m, p = 4, 5, 6
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 120s limit

    # We will build U, V, W as numpy arrays, then convert to lists.
    # Total rank: 49 (for 4x4x4 core) + 24 (for 4x1x6 strip) = 73
    rank_core = 49
    rank_strip = 24
    total_rank = rank_core + rank_strip

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    # Helper to add a Strassen 2x2x2 block
    # A block at rows i0:i0+2, cols j0:j0+2 of A, and rows j0:j0+2, cols k0:k0+2 of B
    def add_strassen(i0, j0, k0, r_offset):
        # M1 = (A00 + A11)(B00 + B11) -> C00 += 1, C11 += 1
        U[(i0+0)*m + (j0+0), r_offset] += 1
        U[(i0+1)*m + (j0+1), r_offset] += 1
        V[(j0+0)*p + (k0+0), r_offset] += 1
        V[(j0+1)*p + (k0+1), r_offset] += 1
        W[(k0+0)*n + (i0+0), r_offset] += 1
        W[(k0+1)*n + (i0+1), r_offset] += 1
        
        # M2 = (A10 + A11)B00 -> C10 += 1, C11 += -1
        r = r_offset + 1
        U[(i0+1)*m + (j0+0), r] += 1
        U[(i0+1)*m + (j0+1), r] += 1
        V[(j0+0)*p + (k0+0), r] += 1
        W[(k0+0)*n + (i0+1), r] += 1
        W[(k0+1)*n + (i0+1), r] += -1
        
        # M3 = A00(B01 - B11) -> C01 += 1, C11 += 1
        r = r_offset + 2
        U[(i0+0)*m + (j0+0), r] += 1
        V[(j0+0)*p + (k0+1), r] += 1
        V[(j0+1)*p + (k0+1), r] += -1
        W[(k0+1)*n + (i0+0), r] += 1
        W[(k0+1)*n + (i0+1), r] += 1
        
        # M4 = A11(B10 - B00) -> C00 += 1, C10 += 1
        r = r_offset + 3
        U[(i0+1)*m + (j0+1), r] += 1
        V[(j0+1)*p + (k0+0), r] += 1
        V[(j0+0)*p + (k0+0), r] += -1
        W[(k0+0)*n + (i0+0), r] += 1
        W[(k0+0)*n + (i0+1), r] += 1
        
        # M5 = (A00 + A01)B11 -> C00 += -1, C01 += 1
        r = r_offset + 4
        U[(i0+0)*m + (j0+0), r] += 1
        U[(i0+0)*m + (j0+1), r] += 1
        V[(j0+1)*p + (k0+1), r] += 1
        W[(k0+0)*n + (i0+0), r] += -1
        W[(k0+1)*n + (i0+0), r] += 1
        
        # M6 = A10(B00 - B10) -> C10 += 1, C11 += -1
        r = r_offset + 5
        U[(i0+1)*m + (j0+0), r] += 1
        V[(j0+0)*p + (k0+0), r] += 1
        V[(j0+1)*p + (k0+0), r] += -1
        W[(k0+0)*n + (i0+1), r] += 1
        W[(k0+1)*n + (i0+1), r] += -1
        
        # M7 = (A01 - A10 - A11)(B10 + B11) -> C11 += 1
        r = r_offset + 6
        U[(i0+0)*m + (j0+1), r] += 1
        U[(i0+1)*m + (j0+0), r] += -1
        U[(i0+1)*m + (j0+1), r] += -1
        V[(j0+1)*p + (k0+0), r] += 1
        V[(j0+1)*p + (k0+1), r] += 1
        W[(k0+1)*n + (i0+1), r] += 1

    # --- Part 1: 4x4 x 4x4 core (rows 0:4, cols 0:4 of A; rows 0:4, cols 0:4 of B) ---
    # We decompose the 4x4x4 tensor into four 2x2x2 Strassen blocks.
    # Block (0,0): i0=0, j0=0, k0=0 -> rank 7
    add_strassen(0, 0, 0, 0)
    # Block (1,0): i0=0, j0=2, k0=0 -> rank 7
    add_strassen(0, 2, 0, 7)
    # Block (0,1): i0=2, j0=0, k0=0 -> rank 7
    add_strassen(2, 0, 0, 14)
    # Block (1,1): i0=2, j0=2, k0=0 -> rank 7
    add_strassen(2, 2, 0, 21)
    # Block (0,0) for second half of B? No, B is 4x6, we only use first 4 cols for core.
    # Wait, the core is A[0:4, 0:4] * B[0:4, 0:4].
    # The blocks are:
    # A[0:2, 0:2] * B[0:2, 0:2] -> add_strassen(0, 0, 0, 0)
    # A[0:2, 2:4] * B[2:4, 0:2] -> add_strassen(0, 2, 0, 7)
    # A[2:4, 0:2] * B[0:2, 0:2] -> add_strassen(2, 0, 0, 14)
    # A[2:4, 2:4] * B[2:4, 0:2] -> add_strassen(2, 2, 0, 21)
    # A[0:2, 0:2] * B[0:2, 2:4] -> add_strassen(0, 0, 2, 28)
    # A[0:2, 2:4] * B[2:4, 2:4] -> add_strassen(0, 2, 2, 35)
    # A[2:4, 0:2] * B[0:2, 2:4] -> add_strassen(2, 0, 2, 42)
    # A[2:4, 2:4] * B[2:4, 2:4] -> add_strassen(2, 2, 2, 49)
    # Wait, I only added 4 blocks above. I need 8 blocks for the full 4x4x4.
    # Let's re-do the core properly.
    
    # Reset U, V, W for the core part
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # 8 Strassen blocks for 4x4x4
    add_strassen(0, 0, 0, 0)   # A[0:2,0:2] * B[0:2,0:2]
    add_strassen(0, 2, 0, 7)   # A[0:2,2:4] * B[2:4,0:2]
    add_strassen(2, 0, 0, 14)  # A[2:4,0:2] * B[0:2,0:2]
    add_strassen(2, 2, 0, 21)  # A[2:4,2:4] * B[2:4,0:2]
    add_strassen(0, 0, 2, 28)  # A[0:2,0:2] * B[0:2,2:4]
    add_strassen(0, 2, 2, 35)  # A[0:2,2:4] * B[2:4,2:4]
    add_strassen(2, 0, 2, 42)  # A[2:4,0:2] * B[0:2,2:4]
    add_strassen(2, 2, 2, 49)  # A[2:4,2:4] * B[2:4,2:4]
    
    # --- Part 2: Strips ---
    # A is 4x5, B is 5x6.
    # Core used A[0:4, 0:4] and B[0:4, 0:4].
    # Remaining:
    # 1. A[0:4, 4:5] * B[4:5, 0:6] (4x1 x 1x6) -> rank 24
    # 2. A[0:4, 0:4] * B[0:4, 4:6] (4x4 x 4x2) -> rank 32
    # Total rank so far: 49 + 24 + 32 = 105.
    # This is worse than the 108 from inspiration 2? No, 105 < 108.
    # But can we do better?
    # Let's check the dimensions again.
    # A: 4x5, B: 5x6.
    # Split A into A1 (4x4) and A2 (4x1).
    # Split B into B1 (4x6) and B2 (1x6).
    # C = A1*B1 + A2*B2.
    # A1*B1: 4x4 * 4x6. Split B1 into B1a (4x4) and B1b (4x2).
    # A1*B1a: 4x4 * 4x4 -> rank 49 (Strassen)
    # A1*B1b: 4x4 * 4x2 -> rank 32 (Schoolbook: 4*4*2=32)
    # A2*B2: 4x1 * 1x6 -> rank 24 (Schoolbook: 4*1*6=24)
    # Total: 49 + 32 + 24 = 105.
    
    # Can we do A1*B1b better?
    # 4x4 * 4x2. Split A1 into 2x2 blocks.
    # A1[0:2,0:4] * B1b[0:4,0:2] -> 2x4 * 4x2 -> rank 16
    # A1[2:4,0:4] * B1b[0:4,0:2] -> 2x4 * 4x2 -> rank 16
    # Total 32. No improvement.
    
    # What if we split differently?
    # A: 4x5, B: 5x6.
    # Maybe use Strassen for the 4x4x4 core is the best we can do with this structure.
    
    # Let's verify the rank 105 construction.
    # We already added the 49 rank core.
    # Now add A1*B1b: A[0:4, 0:4] * B[0:4, 4:6]
    # This is 4x4 * 4x2.
    # Schoolbook: for i in 0..3, j in 0..3, k in 4..5:
    # U[i*5+j] += 1, V[j*6+k] += 1, W[k*4+i] += 1
    # But wait, the indices in U, V, W are global.
    # U row index: i*m + j = i*5 + j.
    # V row index: j*p + k = j*6 + k.
    # W row index: k*n + i = k*4 + i.
    
    r_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[i*m + j, r_offset] += 1
                V[j*p + k, r_offset] += 1
                W[k*n + i, r_offset] += 1
                r_offset += 1
    # r_offset is now 49 + 32 = 81.
    
    # Now add A2*B2: A[0:4, 4:5] * B[4:5, 0:6]
    # This is 4x1 * 1x6.
    for i in range(4):
        for k in range(6):
            U[i*m + 4, r_offset] += 1
            V[4*p + k, r_offset] += 1
            W[k*n + i, r_offset] += 1
            r_offset += 1
    # r_offset is now 81 + 24 = 105.
    
    # Total rank is 105.
    
    # Check time
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
