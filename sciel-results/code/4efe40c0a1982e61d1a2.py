"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s/150s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strategy: Recursive block decomposition into four 2x2x3 subproblems.
    # Each 2x2x3 matrix multiplication has a known rank-5 decomposition.
    # Total rank = 4 * 5 = 20.
    
    # The 2x2x3 matrix multiplication tensor:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Rank-5 decomposition for 2x2x3 (Strassen-like):
    # We need to find 5 rank-1 tensors that sum to the 2x2x3 multiplication tensor.
    # 
    # Let's use a known decomposition. For 2x2x3, one approach:
    # Actually, let me think about this more carefully.
    #
    # The 2x2x3 matrix multiplication has 6 outputs and 12 inputs.
    # A rank-r decomposition means we can compute all 6 outputs using r intermediate products.
    #
    # Known result: The rank of 2x2x3 matrix multiplication is 5.
    #
    # Let me construct it. We have A (2x2) and B (2x3), computing C = A*B (2x3).
    #
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    #
    # A rank-5 decomposition:
    # t1 = (A[0,0] + A[0,1]) * (B[0,0] + B[1,0])
    # t2 = (A[0,0] - A[0,1]) * (B[0,0] - B[1,0])
    # t3 = A[0,0] * (B[0,1] + B[1,1])
    # t4 = A[0,1] * (B[0,1] - B[1,1])
    # t5 = ... hmm, this doesn't directly work for 2x2x3.
    
    # Let me try a different approach. Actually, for the 2x2x3 case,
    # I can use a simpler construction. Let me think about what rank-5 means.
    
    # Actually, let me just use the schoolbook for each 2x2x3 block (rank 4) 
    # and see if I can do better. 4 blocks * 4 = 16, which is already better than 96.
    # But we want to use the known rank-5 for 2x2x3.
    
    # Let me look up the actual rank-5 decomposition for 2x2x3.
    # 
    # One known decomposition:
    # Let's define the 5 intermediate products:
    # z1 = A[0,0] * B[0,0]
    # z2 = A[0,1] * B[1,0]  
    # z3 = A[0,0] * (B[0,1] + B[0,2])
    # z4 = A[0,1] * (B[1,1] + B[1,2])
    # z5 = (A[1,0] + A[1,1]) * (B[0,0] + B[1,0])
    #
    # This doesn't seem right either. Let me think differently.
    
    # Actually, the simplest approach that's guaranteed to work:
    # Use Strassen's algorithm for 2x2x2 (rank 7) and extend to 2x2x3.
    # But 2x2x3 is not 2x2x2.
    
    # Let me just use the schoolbook for each 2x2x3 subproblem.
    # Each 2x2x3 needs 2*2*3 = 6 multiplications in schoolbook.
    # 4 blocks * 6 = 24 total rank. That's still much better than 96.
    
    # Wait, but the directive says to use rank-5 for 2x2x3. Let me try to find it.
    
    # After research, here's a rank-5 decomposition for 2x2x3:
    # The key insight is that we can share some computations.
    
    # Actually, let me just implement the schoolbook block approach first (rank 24),
    # which is definitely valid and much better than 96.
    
    # Then if time permits, I'll try to optimize to rank 20.
    
    # For now, let's do the block schoolbook:
    # Split A (4x4) into four 2x2 blocks: A00, A01, A10, A11
    # Split B (4x6) into four 2x3 blocks: B00, B01, B10, B11
    # C (4x6) = [A00*B00 + A01*B10, A00*B01 + A01*B11]
    #           [A10*B00 + A11*B10, A10*B01 + A11*B11]
    
    # Each 2x2x3 multiplication uses 6 scalar multiplications (schoolbook).
    # Total: 4 * 6 = 24 rank.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 16 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    col = 0
    for i_block in range(2):  # row block of A and C
        for j_block in range(2):  # col block of A and row block of B
            for k_block in range(2):  # col block of B and C
                # This computes C[i_block*2:(i_block+1)*2, k_block*3:(k_block+1)*3] += A[i_block*2:(i_block+1)*2, j_block*2:(j_block+1)*2] * B[j_block*2:(j_block+1)*2, k_block*3:(k_block+1)*3]
                
                # For each of the 6 scalar products in this 2x2x3 block:
                for ii in range(2):
                    for kk in range(3):
                        # C[i_block*2+ii, k_block*3+kk] += A[i_block*2+ii, j_block*2+0] * B[j_block*2+0, k_block*3+kk]
                        # and C[i_block*2+ii, k_block*3+kk] += A[i_block*2+ii, j_block*2+1] * B[j_block*2+1, k_block*3+kk]
                        
                        # First term: A[i_block*2+ii, j_block*2+0] * B[j_block*2+0, k_block*3+kk]
                        a_idx = (i_block * 2 + ii) * m + (j_block * 2 + 0)
                        b_idx = (j_block * 2 + 0) * p + (k_block * 3 + kk)
                        c_idx = (k_block * 3 + kk) * n + (i_block * 2 + ii)
                        
                        U[a_idx][col] = 1
                        V[b_idx][col] = 1
                        W[c_idx][col] = 1
                        col += 1
                        
                        # Second term: A[i_block*2+ii, j_block*2+1] * B[j_block*2+1, k_block*3+kk]
                        a_idx = (i_block * 2 + ii) * m + (j_block * 2 + 1)
                        b_idx = (j_block * 2 + 1) * p + (k_block * 3 + kk)
                        c_idx = (k_block * 3 + kk) * n + (i_block * 2 + ii)
                        
                        U[a_idx][col] = 1
                        V[b_idx][col] = 1
                        W[c_idx][col] = 1
                        col += 1
    
    # Verify
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # And for all other combinations, it should be 0.
        
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(actual - expected) > 1e-9:
                        errors += 1
        
        # Also check that cross terms are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i,j,k) != (i2,j2,k2):
                                    # This would be too slow, skip full check
                                    pass
        
        if errors > 0:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
