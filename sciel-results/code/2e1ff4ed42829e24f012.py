import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # 1. Define Strassen factors for 2x2x2
    U2 = np.zeros((2, 2, 7))
    V2 = np.zeros((2, 2, 7))
    W2 = np.zeros((2, 2, 7))
    
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1
    
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1
    
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1
    
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1
    
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1
    
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1
    
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # 2. Build 4x4x4 factors using recursion
    U4 = np.zeros((4, 4, 49))
    V4 = np.zeros((4, 4, 49))
    W4 = np.zeros((4, 4, 49))
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2*k1 + k0
                    i = 2*i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7*s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # 3. Build 4x4x8 factors - try to share between two halves
    # First, build the standard rank-98 decomposition
    rank_std = 98
    U_std = np.zeros((16, rank_std))
    V_std = np.zeros((32, rank_std))
    W_std = np.zeros((32, rank_std))
    
    for i in range(4):
        for j in range(4):
            U_std[i*4 + j, :49] = U4[i, j, :]
            U_std[i*4 + j, 49:] = U4[i, j, :]
            
    for j in range(4):
        for k in range(4):
            V_std[j*8 + k, :49] = V4[j, k, :]
            V_std[j*8 + k + 4, 49:] = V4[j, k, :]
            
    for k in range(4):
        for i in range(4):
            W_std[k*4 + i, :49] = W4[k, i, :]
            W_std[(k+4)*4 + i, 49:] = W4[k, i, :]
    
    # Try to find shared structure: 
    # The two halves have identical U columns. 
    # We can try to combine: for each of the 49 rank terms in half1 and half2,
    # check if we can merge them by solving for a shared U.
    # 
    # Approach: Try to find a decomposition with rank < 98 by exploiting
    # the fact that U columns are duplicated.
    # 
    # Alternative: Use a different factorization of 4x4x8 directly.
    # Think of it as: split A into 2x2 blocks, B into 2x4 blocks.
    # This gives 4 sub-problems of 2x2x4.
    # Each 2x2x4 can be done with rank 14 (two Strassen 2x2x2).
    # Total: 4 * 14 = 56? No, that's not how it works.
    #
    # Actually, let's try: 4x4x8 = combine 2x4x8 and 2x4x8?
    # Or: use the fact that 4x4x8 tensor = (4x4x4) ⊗_k (identity)
    #
    # Let me try a direct approach: build 2x2x4 factors, then combine.
    
    # Build 2x2x4 factors (rank 14: two copies of 2x2x2 Strassen)
    U24 = np.zeros((2, 2, 14))
    V24 = np.zeros((2, 4, 14))
    W24 = np.zeros((4, 2, 14))
    
    for s in range(7):
        for i in range(2):
            for j in range(2):
                U24[i, j, s] = U2[i, j, s]
                U24[i, j, 7+s] = U2[i, j, s]
            for k in range(2):
                V24[j, k, s] = V2[j, k, s]
                V24[j, k+2, 7+s] = V2[j, k, s]
            for k in range(2):
                W24[k, i, s] = W2[k, i, s]
                W24[k+2, i, 7+s] = W2[k, i, s]
    
    # Now combine four 2x2x4 blocks for 4x4x8
    # A is 4x4: split into 2x2 blocks A[a1,j1]
    # B is 4x8: split into 2x4 blocks B[j1,k1] where k1 in {0,1,2,3}
    # C is 4x8: C[i1,k1] = sum_j1 A[i1,j1] B[j1,k1]
    #
    # For each (i1,j1) pair in A and (j1,k1) pair in B, we have a 2x2x4 product.
    # Total blocks: 2*2 for i1,j1 times 2*4 for k1... 
    # Actually: C[i1,k1] = sum_{j1=0,1} A[i1,j1] * B[j1,k1]
    # Each term is (2x2) * (2x4) = 2x4 result
    # We have 2 choices for j1, and for each, a 2x2x4 multiplication.
    #
    # For fixed (i1,k1), C[i1,k1] = A[i1,0]*B[0,k1] + A[i1,1]*B[1,k1]
    # Each product is 2x2 * 2x4 -> 2x4
    #
    # Using rank-14 for each 2x2x4:
    # Total rank: 2 (for j1=0,1) * 14 = 28 per (i1,k1) block?
    # No, the factors share structure.
    #
    # Let me build it properly:
    # U: shape (16, rank) - indexed by (i1,i0,j1,j0) = (i1,j1,i0,j0)
    # V: shape (32, rank) - indexed by (j1,j0,k1,k0) = (j1,k1,j0,k0)  
    # W: shape (32, rank) - indexed by (k1,k0,i1,i0) = (k1,i1,k0,i0)
    #
    # For the j1=0 part: use U24[i0,j0,s], V24[j0,k0,s], W24[k0,i0,s] for s in 0..13
    # For the j1=1 part: use U24[i0,j0,s], V24[j0,k0,s], W24[k0,i0,s] for s in 0..13
    #
    # But U depends on (i1,j1), and for j1=0 vs j1=1, the U factors are the same!
    # So we can potentially merge.
    
    # Build 4x4x8 with the 2x2x4 building blocks
    rank_224 = 14
    total_rank = 2 * rank_224  # 28? That seems too low...
    
    # Wait, let me reconsider. For each (i1, k1) block of C:
    # C[i1,k1] = A[i1,0]*B[0,k1] + A[i1,1]*B[1,k1]
    # The first product uses U factors from A[i1,0] (which is 2x2) and V from B[0,k1] (2x4)
    # The second uses U from A[i1,1] and V from B[1,k1]
    #
    # U for first product: U24[i0, j0, s] where j0 is the local index in the 2x2 block A[i1,0]
    #   Global U index: (i1, 0, i0, j0) -> flat: i1*8 + 0*4 + i0*2 + j0 = i1*8 + i0*2 + j0
    # U for second product: U24[i0, j0, s] where j0 is local in A[i1,1]
    #   Global U index: (i1, 1, i0, j0) -> flat: i1*8 + 1*4 + i0*2 + j0 = i1*8 + 4 + i0*2 + j0
    #
    # So U factors ARE different (different j1). No sharing possible this way.
    #
    # Total rank would be: for each of the 4 values of (i1,k1), we need rank 28
    # But U,V,W are global... Let me think again.
    #
    # Actually the tensor is: T[i,j,k] = delta_{i,j,k} for matmul
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] is the standard matmul tensor.
    #
    # The rank-14 2x2x4 decomposition works for a single 2x2x4 sub-tensor.
    # For 4x4x8, I need to tile this.
    #
    # Let me just verify the standard rank-98 is correct and try a different angle.
    
    # Alternative: try to reduce rank by finding linear dependencies
    # among the 98 rank-1 terms.
    
    # Check if the 98 columns of U are linearly independent
    U_mat = U_std  # 16 x 98
    V_mat = V_std  # 32 x 98
    W_mat = W_std  # 32 x 98
    
    # The rank-1 terms are: u_r * v_r * w_r (outer products)
    # They're linearly independent as vectors in R^{16*32*32}
    # But we can check if some combination of columns gives the same tensor
    
    # For a valid decomposition, we need:
    # sum_r U[:,r] * V[:,r] * W[:,r] = T (the matmul tensor)
    # where the product is element-wise outer product
    
    # If we have two decompositions with the same T, we can try to find
    # a lower-rank one.
    
    # Let me try: can we find a 97-rank decomposition?
    # This would require finding a linear dependency among the rank-1 terms
    # that still sums to T.
    
    # The key observation: U columns for half1 and half2 are identical.
    # U_std[:, :49] == U_std[:, 49:]
    # 
    # So for the tensor: T = sum_{r=0}^{48} u_r * v1_r * w1_r + sum_{r=0}^{48} u_r * v2_r * w2_r
    # = sum_{r=0}^{48} u_r * (v1_r * w1_r + v2_r * w2_r)  -- NO, this isn't right
    # because v1_r and v2_r have different supports (different k ranges)
    
    # Actually v1_r has support in k=0..3 and v2_r has support in k=4..7
    # So they're on disjoint supports. The sum v1_r*w1_r + v2_r*w2_r is NOT a
    # valid rank-1 term in the V,W space.
    
    # But we could try: for each r, define v_combined_r = v1_r + v2_r (in R^32)
    # and w_combined_r = w1_r + w2_r (in R^32)
    # Then u_r * v_combined_r * w_combined_r = u_r * (v1_r*w1_r + v1_r*w2_r + v2_r*w1_r + v2_r*w2_r)
    # This has cross terms v1_r*w2_r and v2_r*w1_r that we don't want.
    
    # So simple merging doesn't work. We need a more sophisticated approach.
    
    # Let me try: use the fact that the problem is 4x4x8 and try a
    # non-recursive decomposition.
    
    # Known results: 
    # - 2x2x2: rank 7 (Strassen)
    # - 2x2x4: rank 14 (two Strassen)
    # - 4x4x4: rank 49 (Strassen^2)
    # - 4x4x8: ? 
    #
    # The trivial bound is 2*49=98. Can we do better?
    #
    # One idea: use a 2x4x4 decomposition with rank < 49/2 = 24.5
    # If 2x4x4 has rank 24, then 4x4x8 = 2*(2x4x4) could have rank 48.
    # But 2x4x4: A is 2x4, B is 4x4, C is 2x4.
    # This is equivalent to two 2x2x4 problems (splitting the 4 columns of A
    # into two 2-column groups, or the 4 columns of B).
    # Each 2x2x4 has rank 14, so 2x4x4 has rank at most 28.
    # Then 4x4x8 = 2*(2x4x4) has rank at most 56.
    
    # Let me build this!
    # 2x4x4: A is 2x4, B is 4x4, C is 2x4
    # Split A into A[:, :2] and A[:, 2:] (two 2x2 matrices)
    # C = A[:, :2] * B[:, :2] + A[:, 2:] * B[:, 2:]
    # Each term is 2x2 * 2x4 -> 2x4
    # Using rank-14 for each: total rank 28
    
    # Build 2x4x4 factors
    U244 = np.zeros((2, 4, 28))
    V244 = np.zeros((4, 4, 28))
    W244 = np.zeros((4, 2, 28))
    
    for s in range(14):
        # First half: A[:, :2] * B[:, :2]
        for i in range(2):
            for j in range(2):
                U244[i, j, s] = U24[i, j, s]
        for j in range(2):
            for k in range(4):
                V244[j, k, s] = V24[j, k, s]
        for k in range(4):
            for i in range(2):
                W244[k, i, s] = W24[k, i, s]
        
        # Second half: A[:, 2:] * B[:, 2:]
        for i in range(2):
            for j in range(2):
                U244[i, j+2, 14+s] = U24[i, j, s]
        for j in range(2):
            for k in range(4):
                V244[j+2, k, 14+s] = V24[j, k, s]
        for k in range(4):
            for i in range(2):
                W244[k, i, 14+s] = W24[k, i, s]
    
    # Now build 4x4x8 from two 2x4x4 blocks
    # A is 4x4: split into A[:2, :] and A[2:, :] (each 2x4)
    # C is 4x8: C[:2, :] = A[:2,:] * B, C[2:, :] = A[2:,:] * B
    # B is 4x8 (shared)
    
    # For C[:2, :] = A[:2,:] * B:
    #   Use U244[i, j, s] for i in 0..1, j in 0..3, s in 0..27
    #   V244[j, k, s] for j in 0..3, k in 0..7... 
    #   Wait, B is 4x8, but V244 expects B to be 4x4.
    #   I need to split B into B[:, :4] and B[:, 4:]
    
    # Let me redo: 2x4x8 instead of 2x4x4
    # A is 2x4, B is 4x8, C is 2x8
    # C = A * B = A[:, :2] * B[:, :2]... no, B is 4x8
    # C[i, k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # Split j into 0,1 and 2,3:
    # C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # Each term: (2x2) * (2x8) -> 2x8
    # 2x2x8: rank? Two copies of 2x2x4, each rank 14, total 28.
    # Or: 2x2x8 = 2x2x(2+2+2+2), four 2x2x2, rank 4*7=28.
    
    # Build 2x2x8 factors
    U228 = np.zeros((2, 2, 28))
    V228 = np.zeros((2, 8, 28))
    W228 = np.zeros((8, 2, 28))
    
    for s in range(7):
        for i in range(2):
            for j in range(2):
                U228[i, j, s] = U2[i, j, s]
            for k in range(2):
                V228[j, k, s] = V2[j, k, s]
            for k in range(2):
                W228[k, i, s] = W2[k, i, s]
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    U228[i, j, 7+s] = U2[i, j, s]
            for k in range(2):
                V228[j, k+2, 7+s] = V2[j, k, s]
            for k in range(2):
                W228[k+2, i, 7+s] = W2[k, i, s]
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    U228[i, j, 14+s] = U2[i, j, s]
            for k in range(2):
                V228[j, k+4, 14+s] = V2[j, k, s]
            for k in range(2):
                W228[k+4, i, 14+s] = W2[k, i, s]
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    U228[i, j, 21+s] = U2[i, j, s]
            for k in range(2):
                V228[j, k+6, 21+s] = V2[j, k, s]
            for k in range(2):
                W228[k+6, i, 21+s] = W2[k, i, s]
    
    # Build 2x4x8 from two 2x2x8 blocks
    U248 = np.zeros((2, 4, 56))
    V248 = np.zeros((4, 8, 56))
    W248 = np.zeros((8, 2, 56))
    
    for s in range(28):
        for i in range(2):
            for j in range(2):
                U248[i, j, s] = U228[i, j, s]
        for j in range(2):
            for k in range(8):
                V248[j, k, s] = V228[j, k, s]
        for k in range(8):
            for i in range(2):
                W248[k, i, s] = W228[k, i, s]
        
        for i in range(2):
            for j in range(2):
                U248[i, j+2, 28+s] = U228[i, j, s]
        for j in range(2):
            for k in range(8):
                V248[j+2, k, 28+s] = V228[j, k, s]
        for k in range(8):
            for i in range(2):
                W248[k, i, 28+s] = W228[k, i, s]
    
    # Build 4x4x8 from two 2x4x8 blocks
    U_final = np.zeros((16, 112))
    V_final = np.zeros((32, 112))
    W_final = np.zeros((32, 112))
    
    for s in range(56):
        for i in range(2):
            for j in range(4):
                U_final[i*4+j, s] = U248[i, j, s]
        for j in range(4):
            for k in range(8):
                V_final[j*8+k, s] = V248[j, k, s]
        for k in range(8):
            for i in range(2):
                W_final[k*2+i, s] = W248[k, i, s]
        
        for i in range(2):
            for j in range(4):
                U_final[(i+2)*4+j, 56+s] = U248[i, j, s]
        for j in range(4):
            for k in range(8):
                V_final[j*8+k, 56+s] = V248[j, k, s]
        for k in range(8):
            for i in range(2):
                W_final[k*2+i, 56+s] = W248[k, i, s]
    
    # This gives rank 112, which is worse than 98!
    # The issue: 2x2x8 has rank 28 (four Strassen), and we're tiling inefficiently.
    
    # Let me go back to the rank-98 solution and try to verify it's correct,
    # then look for a better decomposition.
    
    # Actually, let me try a different tiling: 4x4x8 as 4 copies of 1x4x8?
    # No, that's not helpful.
    
    # The rank-98 solution is likely near-optimal for this family.
    # Let me just return it with a time guard.
    
    if time.time() > deadline:
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
    
    return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
# EVOLVE-BLOCK-END
