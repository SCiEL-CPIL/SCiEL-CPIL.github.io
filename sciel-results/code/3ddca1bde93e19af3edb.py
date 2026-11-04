import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # U is (i,j) flattened, V is (j,k) flattened, W is (k,i) flattened
    # Each entry in rank r contributes to tensor[i,j,k] as U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        """Add a Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * 
           B[j_base:j_base+2, k_base:k_base+2] starting at rank index base_rank."""
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val
    
    # A0 * Bt0: i_base=0, j_base=0, k_base=0
    add_block(0, 0, 0, 0)
    
    # A0 * Bt1: i_base=0, j_base=0, k_base=2
    add_block(7, 0, 0, 2)
    
    # A0 * Bt2: i_base=0, j_base=0, k_base=4
    add_block(14, 0, 0, 4)
    
    # A0 * Bt3: i_base=0, j_base=0, k_base=6
    add_block(21, 0, 0, 6)
    
    # A1 * Bb0: i_base=0, j_base=2, k_base=0
    add_block(28, 0, 2, 0)
    
    # A1 * Bb1: i_base=0, j_base=2, k_base=2
    add_block(35, 0, 2, 2)
    
    # A1 * Bb2: i_base=0, j_base=2, k_base=4
    add_block(42, 0, 2, 4)
    
    # A1 * Bb3: i_base=0, j_base=2, k_base=6
    add_block(49, 0, 2, 6)
    
    # Now attempt rank reduction by finding linear dependencies among the rank-1 tensors
    # Convert to numpy arrays for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find if any rank-1 component is linearly dependent on others
    
    # Build the tensor as a 3D array
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Check if we can remove any rank component
    # For each rank index r, check if the rank-1 tensor T_r is in the span of others
    # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # We'll try to find linear dependencies by checking if removing a component
    # and re-solving still gives the same tensor
    
    # Strategy: For each pair of blocks with the same (i_base, k_base), 
    # check if we can combine them more efficiently
    
    # Actually, let's try a simpler approach: check if any two rank components
    # are proportional (which would indicate redundancy)
    
    # For each rank r, compute the "signature" of the rank-1 tensor
    # and check for linear dependencies
    
    # More practical approach: try to find if we can express one block's
    # Strassen decomposition using fewer rank terms by sharing with another block
    
    # The key insight: blocks A0*Bt0 and A0*Bt1 both use A0, so their
    # U factors are identical. We might be able to share rank terms.
    
    # Let's try: for blocks with the same i_base, can we combine their
    # Strassen decompositions?
    
    # For A0*Bt0 (rank 0-6) and A0*Bt1 (rank 7-13):
    # Both use the same U pattern (for i=0,1; j=0,1)
    # But V and W differ
    
    # Actually, the standard approach is to use the fact that:
    # A0 * [Bt0 | Bt1 | Bt2 | Bt3] can be computed with rank 7*4 = 28
    # But maybe we can do better by treating it as a 2x4 * 4x8 problem
    
    # Let's try a different strategy: use the rank-56 as baseline and
    # attempt to find if any rank term is truly necessary
    
    # For each rank r, check if T_r = sum_{r' != r} c_{r',r} * T_{r'}
    # This is equivalent to checking if the vector (U[:,r], V[:,r], W[:,r])
    # is in the span of other such vectors when viewed appropriately
    
    # Simpler check: see if any rank component has all zeros in one factor
    # (which would make it zero) or if two components are identical
    
    # Check for zero components
    for r in range(rank):
        if np.allclose(U_np[:, r], 0) or np.allclose(V_np[:, r], 0) or np.allclose(W_np[:, r], 0):
            # This rank term contributes nothing, can be removed
            # But in Strassen, no term should be zero, so this shouldn't happen
            pass
    
    # Check for proportional components
    # For each pair (r1, r2), check if U[:,r1] is proportional to U[:,r2],
    # V[:,r1] to V[:,r2], and W[:,r1] to W[:,r2]
    
    # This is expensive, so let's be selective
    # Focus on blocks that share structure
    
    # The 8 blocks are:
    # Block 0: A0*Bt0 (rank 0-6), i_base=0, j_base=0, k_base=0
    # Block 1: A0*Bt1 (rank 7-13), i_base=0, j_base=0, k_base=2
    # Block 2: A0*Bt2 (rank 14-20), i_base=0, j_base=0, k_base=4
    # Block 3: A0*Bt3 (rank 21-27), i_base=0, j_base=0, k_base=6
    # Block 4: A1*Bb0 (rank 28-34), i_base=0, j_base=2, k_base=0
    # Block 5: A1*Bb1 (rank 35-41), i_base=0, j_base=2, k_base=2
    # Block 6: A1*Bb2 (rank 42-48), i_base=0, j_base=2, k_base=4
    # Block 7: A1*Bb3 (rank 49-55), i_base=0, j_base=2, k_base=6
    
    # Blocks 0-3 all use i_base=0, j_base=0, so their U factors are identical
    # Blocks 4-7 all use i_base=0, j_base=2, so their U factors are identical
    
    # For blocks 0-3, the U factor for rank r (within the block) is the same
    # as for rank r+7, r+14, r+21
    
    # Check: for block 0 (r=0) and block 1 (r=7), is U[:,0] proportional to U[:,7]?
    # They should be identical since both use i_base=0, j_base=0
    
    # Let's verify and look for opportunities
    
    # Actually, the real opportunity is: can we compute A0 * B_top (2x2 * 2x8)
    # with rank less than 4*7 = 28?
    
    # The tensor for 2x2 * 2x8 has dimension 2*2*8 = 32
    # The naive rank is 4*8 = 32, Strassen gives 4*7 = 28
    
    # Is there a known better bound? For 2x2 * 2x2n, the rank is at most 7n
    # but might be less.
    
    # Let's try to find if we can do 2x2 * 2x8 with rank < 28
    
    # One approach: use the fact that we can share Strassen terms across
    # the 4 sub-blocks
    
    # For now, let's just return the rank-56 solution since finding a
    # better decomposition requires more sophisticated search
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find linear dependencies among rank components
    # For each rank r, check if it's in the span of the others
    
    # This is a hard problem, so let's try a heuristic:
    # For each pair of blocks with the same (i_base, j_base), check if
    # we can combine them
    
    # Blocks 0,1,2,3 have same (i_base=0, j_base=0)
    # Their U factors are identical for corresponding Strassen indices
    
    # So for Strassen index s (0-6), we have:
    # U[:, s] = U[:, 7+s] = U[:, 14+s] = U[:, 21+s]
    
    # This means the U factor is shared. Can we exploit this?
    
    # The tensor contribution from Strassen index s across all 4 blocks:
    # sum_{b=0}^{3} U[:,s] * V[:,7b+s] * W[:,7b+s]
    
    # This is a sum of 4 rank-1 tensors with the same U factor.
    # We can factor this as U[:,s] * (V[:,s] + V[:,7+s] + V[:,14+s] + V[:,21+s]) * W[:,s]
    # No, that's not right because V and W differ for each block.
    
    # Actually, we can't simply factor because V and W are different.
    
    # Let's try a different approach: use linear algebra to find if
    # any rank component is redundant
    
    # For each r, compute the residual if we remove rank r
    # and check if it can be explained by the other ranks
    
    # This is equivalent to checking if the tensor T_r is in the
    # span of {T_{r'} : r' != r}
    
    # Let's do this for a subset of r values to save time
    
    # Convert tensor to a vector
    T_vec = T.flatten()
    
    # For each r, compute T_r_vec
    # T_r[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # We'll check a few r values
    for r in range(0, rank, 7):  # Check one from each block
        if time.time() > deadline - 10:
            break
        
        # Compute T_r
        T_r = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_r[i,j,k] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
        
        T_r_vec = T_r.flatten()
        
        # Check if T_r is in the span of other T_r'
        # This is expensive, so let's just check if T_r is zero
        if np.allclose(T_r_vec, 0):
            # Can remove this rank
            pass
        
        # For now, we don't have a fast way to check linear dependence
        # So we'll stick with rank 56
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
