import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 3, 10

    # We will build a decomposition using Strassen for 2x2x2 blocks where possible.
    # The matrix A is 2x3, B is 3x10.
    # We can view B as having 3 rows. The first 2 rows form a 2x10 matrix B12.
    # The 3rd row is a 1x10 vector B3.
    # A is 2x3. Let A12 be the first 2 columns of A (2x2) and A3 be the 3rd column (2x1).
    # C = A * B = [A12 | A3] * [B12; B3] = A12 * B12 + A3 * B3.
    
    # A12 * B12 is 2x2 * 2x10.
    # We can decompose B12 into 5 blocks of 2x2.
    # For each block, use Strassen (rank 7).
    # A3 * B3 is 2x1 * 1x10.
    # This can be viewed as 5 blocks of 2x1 * 1x2.
    # The rank of 2x1 * 1x2 is 2 (standard: 2*1*2=4? No. 
    # Let's check: C[i,k] = A[i] * B[k]. 
    # Tensor: sum_i sum_k e_i e_k. Rank is 2*1*2 = 4? 
    # Wait. For (2,1,2), lower bound is max(2*1, 1*2, 2*2) = max(2,2,4)=4.
    # Is rank 4 achievable? Yes, schoolbook is 4. Can we do better?
    # Strassen is for 2x2x2. Here one dim is 1.
    # Actually, for (2,1,2), the multiplication is:
    # c00 = a0*b0
    # c01 = a0*b1
    # c10 = a1*b0
    # c11 = a1*b1
    # This requires 4 multiplications. No sharing possible because a0 only multiplies b0,b1 and a1 only multiplies b0,b1.
    # So rank is 4 for each 2x1 * 1x2 block.
    # There are 5 such blocks. Total rank 5 * 4 = 20.
    
    # Total rank = 5 * 7 (Strassen) + 5 * 4 (Standard) = 35 + 20 = 55.
    
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*b00
    # m2 = a00*(b01-b11)
    # m3 = a11*(b10-b00)
    # m4 = (a00+a01)*b11
    # m5 = (a10-a00)*(b00+b10)
    # m6 = (a01-a11)*(b01+b11)
    # C = [[m0+m3-m4+m6, m2+m4], [m1+m3, m0-m1+m2+m5]]
    
    # Indices in U, V, W for a block
    # A indices: i in {0,1}, j in {0,1}. Global A index: i*m + j.
    # B indices: j in {0,1}, k in {block_start, block_start+1}. Global B index: j*p + k.
    # C indices: i in {0,1}, k in {block_start, block_start+1}. Global C index: k*n + i.
    
    # We process 5 blocks for the A12 * B12 part.
    # A12 uses columns 0,1 of A. So j=0,1.
    # B12 uses rows 0,1 of B. So j=0,1.
    # k ranges from 0 to 9. Blocks are [0,1], [2,3], [4,5], [6,7], [8,9].
    
    for block in range(5):
        k_start = block * 2
        k0 = k_start
        k1 = k_start + 1
        
        # m0: (a00+a11)*(b00+b11)
        # U: a00 (0,0) -> 1, a11 (1,1) -> 1. Global: 0*m+0=0, 1*m+1=4.
        # V: b00 (0,k0) -> 1, b11 (1,k1) -> 1. Global: 0*p+k0, 1*p+k1.
        # W: c00 (k0,0) -> 1, c01 (k1,0) -> 1, c10 (k0,1) -> 1, c11 (k1,1) -> -1.
        # W indices: k0*n+0, k1*n+0, k0*n+1, k1*n+1.
        
        U[0][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # m1: (a10+a11)*b00
        # U: a10 (1,0) -> 1, a11 (1,1) -> 1. Global: 1*m+0=3, 1*m+1=4.
        # V: b00 (0,k0) -> 1. Global: 0*p+k0.
        # W: c10 (k0,1) -> 1, c11 (k1,1) -> -1.
        
        U[3][col] = 1
        U[4][col] = 1
        V[k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # m2: a00*(b01-b11)
        # U: a00 (0,0) -> 1. Global: 0.
        # V: b01 (0,k1) -> 1, b11 (1,k1) -> -1. Global: 0*p+k1, 1*p+k1.
        # W: c01 (k1,0) -> 1, c11 (k1,1) -> 1.
        
        U[0][col] = 1
        V[k1][col] = 1
        V[p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m3: a11*(b10-b00)
        # U: a11 (1,1) -> 1. Global: 4.
        # V: b10 (1,k0) -> 1, b00 (0,k0) -> -1. Global: 1*p+k0, 0*p+k0.
        # W: c00 (k0,0) -> 1, c10 (k0,1) -> 1.
        
        U[4][col] = 1
        V[p + k0][col] = 1
        V[k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # m4: (a00+a01)*b11
        # U: a00 (0,0) -> 1, a01 (0,1) -> 1. Global: 0, 1.
        # V: b11 (1,k1) -> 1. Global: 1*p+k1.
        # W: c00 (k0,0) -> -1, c01 (k1,0) -> 1.
        
        U[0][col] = 1
        U[1][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # m5: (a10-a00)*(b00+b10)
        # U: a10 (1,0) -> 1, a00 (0,0) -> -1. Global: 3, 0.
        # V: b00 (0,k0) -> 1, b10 (1,k0) -> 1. Global: 0*p+k0, 1*p+k0.
        # W: c11 (k1,1) -> 1.
        
        U[3][col] = 1
        U[0][col] = -1
        V[k0][col] = 1
        V[p + k0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m6: (a01-a11)*(b01+b11)
        # U: a01 (0,1) -> 1, a11 (1,1) -> -1. Global: 1, 4.
        # V: b01 (0,k1) -> 1, b11 (1,k1) -> 1. Global: 0*p+k1, 1*p+k1.
        # W: c00 (k0,0) -> 1.
        
        U[1][col] = 1
        U[4][col] = -1
        V[k1][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

    # Now process A3 * B3.
    # A3 is column 2 of A (j=2). Indices: (0,2)->5, (1,2)->6.
    # B3 is row 2 of B (j=2). Indices: (2,k) -> 2*p+k.
    # C is all of it.
    # For each block of 2 columns in B3 (k0, k1):
    # We need to compute:
    # C[0, k0] += A[0,2] * B[2, k0]
    # C[0, k1] += A[0,2] * B[2, k1]
    # C[1, k0] += A[1,2] * B[2, k0]
    # C[1, k1] += A[1,2] * B[2, k1]
    
    # This requires 4 multiplications per block.
    # 1. A[0,2] * B[2, k0] -> C[0, k0]
    # 2. A[0,2] * B[2, k1] -> C[0, k1]
    # 3. A[1,2] * B[2, k0] -> C[1, k0]
    # 4. A[1,2] * B[2, k1] -> C[1, k1]
    
    for block in range(5):
        k_start = block * 2
        k0 = k_start
        k1 = k_start + 1
        
        # Term 1: A[0,2] * B[2, k0]
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,2] * B[2, k1]
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,2] * B[2, k0]
        U[6][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,2] * B[2, k1]
        U[6][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
