"""Matrix multiplication 3x4x8 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

# Deadline guard: 150s wall-clock limit, use 145s as safety margin
DEADLINE = 145.0

def _elapsed():
    return time.time() - _START

_START = time.time()

def construct():
    global _START
    _START = time.time()
    
    n, m, p = 3, 4, 8
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 96
    
    # Strategy: Use Strassen's algorithm on 2x2 blocks where applicable
    # For 3x4 * 4x8:
    # Partition A (3x4) into rows: A = [A1(1x4); A2(1x4); A3(1x4)]
    # Partition B (4x8) into columns: B = [B1(4x4) B2(4x4)]
    # 
    # Actually, let's think more carefully.
    # A is 3x4, B is 4x8, C is 3x8.
    # 
    # We can partition the 4 middle dimension.
    # Let's try: split the 4 into 2+2.
    # A = [A11(3x2) A12(3x2)]
    # B = [B21(2x8); B22(2x8)]
    # C = A11*B21 + A12*B22
    # 
    # Each product is 3x2 * 2x8 = 3x8, rank 3*2*8=48 each, total 96. No gain.
    #
    # Alternative: Use the known result that for 2x2x2, Strassen gives rank 7.
    # For our problem, we need to find a good decomposition.
    #
    # Let's try a different partition:
    # A (3x4) = [A1(3x2) | A2(3x2)]
    # B (4x8) = [B1(2x8); B2(2x8)]
    # C = A1*B1 + A2*B2
    #
    # Now, A1*B1 is 3x2 * 2x8. We can further decompose.
    # Actually, let's think of it as:
    # C[i,k] = sum_j A[i,j] B[j,k] for j=0..3
    # 
    # Let me try a block approach with Strassen on the 2x2 core:
    # Partition the 4x4 middle into four 2x2 blocks:
    # A = [A11 A12] where A11, A12 are 3x2
    # B = [B11; B21] where B11, B21 are 2x8... wait, B is 4x8.
    # B = [B1; B2] where B1, B2 are 2x8
    # C = A1*B1 + A2*B2
    #
    # Each term A1*B1: 3x2 * 2x8. Rank of this tensor is 3*2*8=48.
    # Total: 96. No improvement.
    #
    # Let's try a smarter approach. The key insight from the knowledge:
    # "Using Strassen's algorithm on 2x2x2 blocks reduces the rank of the 2x2x8 subproblems from 32 to 28"
    # 
    # So if we can decompose our problem into subproblems of type 2x2x8, we can use rank 28 for each.
    #
    # 3x4 * 4x8: 
    # Let's partition A into 3 row-blocks of 1x4 each, and B into 4 column-blocks of 4x2 each? No...
    #
    # Better: Partition the middle 4 into 2+2.
    # A = [A_left(3x2) | A_right(3x2)]
    # B = [B_top(2x8); B_bottom(2x8)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Now A_left * B_top is 3x2 * 2x8. 
    # Can we decompose 3x2*2x8 more efficiently?
    # 3x2*2x8: partition the 3 into 1+2.
    # A_left = [a1(1x2); a2(2x2)]
    # B_top is 2x8.
    # a1 * B_top: 1x2 * 2x8, rank 16.
    # a2 * B_top: 2x2 * 2x8, can use Strassen-like: rank 28 (from knowledge).
    # Total for A_left*B_top: 16 + 28 = 44.
    # Similarly A_right*B_bottom: 44.
    # Total: 88. This matches the measured result!
    #
    # So the plan is:
    # 1. Implement the 2x2*2x8 decomposition with rank 28 (using Strassen on 2x2x2 blocks).
    # 2. Implement the 1x2*2x8 decomposition with rank 16 (schoolbook).
    # 3. Combine them.
    #
    # Let's work out the 2x2*2x8 decomposition with rank 28.
    # 2x2*2x8: C[i,k] = sum_j A[i,j] B[j,k], i=0,1; j=0,1; k=0..7.
    # Partition k into four groups of 2: k=0,1; k=2,3; k=4,5; k=6,7.
    # For each group of 2 columns, we have a 2x2*2x2 subproblem.
    # Using Strassen on each 2x2*2x2: rank 7 each.
    # 4 groups * 7 = 28. 
    #
    # Strassen for 2x2*2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[0,1]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[1,1]-A[0,1])*(B[0,1]+B[1,1])
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M2 + M3 - M5 + M7
    # C[1,0] = M2 + M3 + M1 - M4
    # C[1,1] = M1 - M2 + M4 + M6
    #
    # Now let's build the full decomposition.
    
    # We'll build U, V, W as lists of lists (or numpy arrays), then convert.
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # Let's use a list to accumulate columns.
    # Each "rank-1" component is (u, v, w) where:
    # u is a vector of length n*m=12
    # v is a vector of length m*p=32
    # w is a vector of length p*n=24
    
    components = []
    
    def add_component(u, v, w):
        """Add a rank-1 component."""
        components.append((u, v, w))
    
    # Helper: zero vectors
    def zero_u():
        return [0] * (n * m)
    def zero_v():
        return [0] * (m * p)
    def zero_w():
        return [0] * (p * n)
    
    # Indexing:
    # U[i*m + j, c] for i in 0..n-1, j in 0..m-1
    # V[j*p + k, c] for j in 0..m-1, k in 0..p-1
    # W[k*n + i, c] for k in 0..p-1, i in 0..n-1
    
    # Now, the decomposition:
    # C = A_left * B_top + A_right * B_bottom
    # where A_left = A[:, 0:2], A_right = A[:, 2:4]
    #       B_top = B[0:2, :], B_bottom = B[2:4, :]
    #
    # A_left * B_top: 3x2 * 2x8
    #   = a1 * B_top + a2 * B_top
    #   where a1 = A[0, 0:2] (1x2), a2 = A[1:3, 0:2] (2x2)
    #
    # a1 * B_top: 1x2 * 2x8, schoolbook rank 16
    # a2 * B_top: 2x2 * 2x8, Strassen rank 28
    #
    # Similarly for A_right * B_bottom.
    
    # Let's define a function to add a 1x2 * 2x8 schoolbook component
    def add_1x2_x_2x8_schoolbook(row_idx, j_offset, k_offset):
        """
        Add schoolbook decomposition for A[row_idx, j_offset:j_offset+2] * B[j_offset:j_offset+2, k_offset:k_offset+8]
        row_idx: which row of A (0, 1, or 2)
        j_offset: starting j index in the 4-dim (0 or 2)
        k_offset: starting k index in the 8-dim (always 0 for full B)
        """
        for j in range(2):
            for k in range(8):
                u = zero_u()
                v = zero_v()
                w = zero_w()
                u[row_idx * m + (j_offset + j)] = 1
                v[(j_offset + j) * p + (k_offset + k)] = 1
                w[(k_offset + k) * n + row_idx] = 1
                add_component(u, v, w)
    
    # Let's define a function to add a 2x2 * 2x8 Strassen component
    def add_2x2_x_2x8_strassen(row_start, j_offset, k_offset):
        """
        Add Strassen decomposition for A[row_start:row_start+2, j_offset:j_offset+2] * B[j_offset:j_offset+2, k_offset:k_offset+8]
        row_start: starting row of A (0 or 1)
        j_offset: starting j index (0 or 2)
        k_offset: starting k index (0)
        """
        # For each group of 2 columns in k
        for kg in range(4):
            k_start = k_offset + kg * 2
            # Strassen 2x2 * 2x2
            # M1 = (A[i1,j1]+A[i2,j2])*(B[j1,k1]+B[j2,k2])
            # where i1=row_start, i2=row_start+1, j1=j_offset, j2=j_offset+1
            #       k1=k_start, k2=k_start+1
            
            # M1
            u = zero_u(); v = zero_v(); w = zero_w()
            u[row_start * m + j_offset] = 1
            u[(row_start + 1) * m + (j_offset + 1)] = 1
            v[j_offset * p + k_start] = 1
            v[(j_offset + 1) * p + (k_start + 1)] = 1
            # W: C[i,k] = M1 + ...
            # C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1
            # Wait, let me re-check Strassen:
            # C[0,0] = M1 + M4 - M5 + M7
            # C[0,1] = M2 + M3 - M5 + M7
            # C[1,0] = M2 + M3 + M1 - M4
            # C[1,1] = M1 - M2 + M4 + M6
            # So M1 contributes to C[0,0], C[1,0], C[1,1]
            w[k_start * n + row_start] = 1       # C[row_start, k_start]
            w[k_start * n + (row_start + 1)] = 1  # C[row_start+1, k_start]
            w[(k_start + 1) * n + (row_start + 1)] = 1  # C[row_start+1, k_start+1]
            add_component(u, v, w)
            
            # M2 = (A[i1,j2]+A[i2,j2])*B[j1,k1]
            u = zero_u(); v = zero_v(); w = zero_w()
            u[row_start * m + (j_offset + 1)] = 1
            u[(row_start + 1) * m + (j_offset + 1)] = 1
            v[j_offset * p + k_start] = 1
            # C[0,1] += M2, C[1,0] += M2
            w[(k_start + 1) * n + row_start] = 1
            w[k_start * n + (row_start + 1)] = 1
            add_component(u, v, w)
            
            # M3 = A[i1,j1]*(B[j2,k1]-B[j2,k2])
            u = zero_u(); v = zero_v(); w = zero_w()
            u[row_start * m + j_offset] = 1
            v[(j_offset + 1) * p + k_start] = 1
            v[(j_offset + 1) * p + (k_start + 1)] = -1
            # C[0,1] += M3, C[1,0] += M3
            w[(k_start + 1) * n + row_start] = 1
            w[k_start * n + (row_start + 1)] = 1
            add_component(u, v, w)
            
            # M4 = A[i2,j2]*(B[j1,k2]-B[j1,k1])
            u = zero_u(); v = zero_v(); w = zero_w()
            u[(row_start + 1) * m + (j_offset + 1)] = 1
            v[j_offset * p + (k_start + 1)] = 1
            v[j_offset * p + k_start] = -1
            # C[0,0] += M4, C[1,0] -= M4, C[1,1] += M4
            w[k_start * n + row_start] = 1
            w[k_start * n + (row_start + 1)] = -1
            w[(k_start + 1) * n + (row_start + 1)] = 1
            add_component(u, v, w)
            
            # M5 = (A[i1,j1]+A[i1,j2])*B[j2,k2]
            u = zero_u(); v = zero_v(); w = zero_w()
            u[row_start * m + j_offset] = 1
            u[row_start * m + (j_offset + 1)] = 1
            v[(j_offset + 1) * p + (k_start + 1)] = 1
            # C[0,0] -= M5, C[0,1] -= M5
            w[k_start * n + row_start] = -1
            w[(k_start + 1) * n + row_start] = -1
            add_component(u, v, w)
            
            # M6 = (A[i2,j1]-A[i1,j1])*(B[j1,k1]+B[j1,k2])
            u = zero_u(); v = zero_v(); w = zero_w()
            u[(row_start + 1) * m + j_offset] = 1
            u[row_start * m + j_offset] = -1
            v[j_offset * p + k_start] = 1
            v[j_offset * p + (k_start + 1)] = 1
            # C[1,1] += M6
            w[(k_start + 1) * n + (row_start + 1)] = 1
            add_component(u, v, w)
            
            # M7 = (A[i2,j2]-A[i1,j2])*(B[j1,k2]+B[j2,k2])
            u = zero_u(); v = zero_v(); w = zero_w()
            u[(row_start + 1) * m + (j_offset + 1)] = 1
            u[row_start * m + (j_offset + 1)] = -1
            v[j_offset * p + (k_start + 1)] = 1
            v[(j_offset + 1) * p + (k_start + 1)] = 1
            # C[0,0] += M7, C[0,1] += M7
            w[k_start * n + row_start] = 1
            w[(k_start + 1) * n + row_start] = 1
            add_component(u, v, w)
    
    # Now build the full decomposition:
    # Part 1: A_left * B_top
    #   A_left = A[:, 0:2], B_top = B[0:2, :]
    #   a1 * B_top: row 0, j_offset=0
    add_1x2_x_2x8_schoolbook(0, 0, 0)
    #   a2 * B_top: rows 1-2, j_offset=0
    add_2x2_x_2x8_strassen(1, 0, 0)
    
    # Part 2: A_right * B_bottom
    #   A_right = A[:, 2:4], B_bottom = B[2:4, :]
    #   a1 * B_bottom: row 0, j_offset=2
    add_1x2_x_2x8_schoolbook(0, 2, 0)
    #   a2 * B_bottom: rows 1-2, j_offset=2
    add_2x2_x_2x8_strassen(1, 2, 0)
    
    rank = len(components)
    
    # Build U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(components):
        for i in range(n * m):
            if u[i] != 0:
                U[i][c] = u[i]
        for i in range(m * p):
            if v[i] != 0:
                V[i][c] = v[i]
        for i in range(p * n):
            if w[i] != 0:
                W[i][c] = w[i]
    
    return {"U": U, "V": V, "W": W}
