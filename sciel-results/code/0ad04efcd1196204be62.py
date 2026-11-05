import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Initialize with Schoolbook Decomposition
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    # Map (i,j,k) to term index
    term_map = {}
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                term_map[(i, j, k)] = r
                r += 1
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True
    
    # 2. Define Strassen 2x2x2 patterns
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_2x2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),      # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),       # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),      # M3
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),      # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),      # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),      # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),      # M7
    ]
    
    def tensor_product_vec(v1, v2, d1, d2):
        """
        v1 indexed by [a1*d1 + b1], v2 by [a2*d2 + b2]
        result indexed by [(a1*d2 + a2)*d1*d2 + (b1*d2 + b2)] 
        Wait, standard tensor product for matrix mult blocks:
        If A is 2x2 and B is 2x2, C is 2x2.
        C_{ik} = sum_j A_{ij} B_{jk}.
        Tensor product of patterns:
        Pattern 1 for 2x2x2: u1[a1*2+b1], v1[b1*2+c1], w1[c1*2+a1]
        Pattern 2 for 2x2x2: u2[a2*2+b2], v2[b2*2+c2], w2[c2*2+a2]
        
        Combined 4x4x4:
        A is 4x4, indexed by (i,j) where i = a1*2+a2, j = b1*2+b2? No.
        Usually we split A into 2x2 blocks of 2x2.
        A_{(a1,a2), (b1,b2)} = A_{a1,b1}^{block} ... this is getting complicated with indices.
        
        Let's use the explicit index mapping from the seed program which worked:
        u_idx = i*m + j.
        For 4x4x4 from two 2x2x2:
        i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2.
        u[i,j] = u1[i1, j1] * u2[i2, j2] ?
        Let's check the seed's tensor_product_vec:
        v1[a1*2+b1], v2[a2*2+b2] -> result[(a1*2+a2)*4 + (b1*2+b2)]
        Here v1 is u1. a1 is i1, b1 is j1.
        v2 is u2. a2 is i2, b2 is j2.
        Result index: (i1*2 + i2)*4 + (j1*2 + j2) = (2*i1 + i2)*4 + 2*j1 + j2 = 8*i1 + 4*i2 + 2*j1 + j2.
        Standard row-major 4x4 index for (i, j) where i = 2*i1 + i2, j = 2*j1 + j2:
        i*4 + j = (2*i1 + i2)*4 + 2*j1 + j2 = 8*i1 + 4*i2 + 2*j1 + j2.
        Matches.
        """
        d = 2
        result = [0] * (d1 * d2 * d1 * d2) # size (d1*d2)^2
        # v1 is size d1*d1, v2 is size d2*d2
        # v1[a1*d1 + b1], v2[a2*d2 + b2]
        for a1 in range(d1):
            for b1 in range(d1):
                for a2 in range(d2):
                    for b2 in range(d2):
                        idx = (a1 * d2 + a2) * (d1 * d2) + (b1 * d2 + b2)
                        result[idx] = v1[a1 * d1 + b1] * v2[a2 * d2 + b2]
        return result

    # Generate 4x4x4 Strassen terms
    strassen_4x4x4 = []
    for t1 in strassen_2x2x2:
        for t2 in strassen_2x2x2:
            u = tensor_product_vec(t1[0], t2[0], 2, 2)
            v = tensor_product_vec(t1[1], t2[1], 2, 2)
            w = tensor_product_vec(t1[2], t2[2], 2, 2)
            strassen_4x4x4.append((u, v, w))
            
    # Also generate 2x2x2 terms embedded in 4x5x5?
    # We can apply 2x2x2 Strassen to any 2x2 sub-block of A, B, C.
    # A is 4x5. Sub-blocks of A: 2x2.
    # B is 5x5. Sub-blocks of B: 2x2.
    # C is 5x4. Sub-blocks of C: 2x2.
    
    # A 2x2x2 pattern covers 8 terms: (i1,i2) x (j1,j2) x (k1,k2)
    # We can try to replace any set of 8 schoolbook terms with 7 Strassen terms
    # if they form a valid 2x2x2 sub-tensor.
    
    # Identify all possible 2x2x2 sub-tensors in 4x5x5
    # i in {0,1,2,3}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # Sub-tensor defined by starting indices i0, j0, k0
    # such that i0+1 < n, j0+1 < m, k0+1 < p
    possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                possible_blocks.append((i0, j0, k0))
                
    # For each block, we have 8 terms.
    # We want to see if we can replace 8 terms with 7.
    # The 7 Strassen terms for this block will have non-zero entries only in the
    # indices corresponding to this block (and potentially others if we don't mask, 
    # but Strassen is exact for the block, so other entries are 0).
    
    # Let's build a candidate list of "replacements".
    # Each replacement is:
    # - List of term indices to REMOVE (8 schoolbook terms)
    # - List of 7 new vectors (u, v, w) to ADD
    
    # However, simply replacing 8 with 7 in isolation might break other terms
    # if the new vectors have non-zero entries outside the block?
    # No, Strassen vectors for a 2x2x2 block, when embedded in larger matrices,
    # have zeros outside the block indices.
    # u_idx = i*m + j.
    # If we use Strassen on block (i0, j0, k0), the u vector has non-zeros only for
    # i in {i0, i0+1} and j in {j0, j0+1}.
    # So it only affects terms involving those i,j.
    # Specifically, it affects the 8 terms (i,j,k) where i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}.
    # It does NOT affect terms where i is outside {i0,i0+1} or j is outside {j0,j0+1}.
    # BUT, the sum over all rank terms must still equal 1 for ALL (i,j,k).
    # If we remove the 8 schoolbook terms for this block, and add 7 Strassen terms,
    # the contribution to these 8 (i,j,k) pairs is preserved (by Strassen).
    # The contribution to other (i,j,k) pairs is 0 from the new terms (since they are zero outside block).
    # And the schoolbook terms for other pairs are untouched.
    # So, if we apply this replacement to a block, it is SAFE and reduces rank by 1.
    
    # Wait, can we apply multiple such replacements?
    # If two blocks overlap, we might be removing a schoolbook term that is part of two blocks.
    # If we process blocks sequentially, we must ensure we don't try to remove a term that's already removed.
    # Also, the new Strassen terms from one block might interfere with the "schoolbook" status of another?
    # No, the new terms are just added to the rank. They don't change the fact that other terms are still "schoolbook" (value 1).
    # However, if we replace block A, we add 7 new terms.
    # If we then replace block B (overlapping A), we remove 8 schoolbook terms.
    # Some of those 8 might have been "covered" by the new terms from A?
    # No, the "schoolbook terms" are the original ones. We are modifying the list of active terms.
    # Let's maintain a set of active term indices.
    
    active_terms = set(range(rank_school))
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    
    # We will perform a greedy search.
    # 1. Generate all possible 2x2x2 block replacements.
    # 2. Sort them by some heuristic (e.g., random, or order of appearance).
    # 3. Try to apply them one by one. If applying a replacement keeps the decomposition valid (which it should, by linearity, if we just swap 8->7 for a disjoint set of indices in the tensor support), we accept it.
    # Actually, since the supports are disjoint in terms of (i,j,k) coordinates?
    # No, blocks can overlap in (i,j,k) space.
    # Example: Block (0,0,0) covers (0,0,0)..(1,1,1).
    # Block (1,0,0) covers (1,0,0)..(2,1,1).
    # They overlap at (1,0,0), (1,1,0), (1,0,1), (1,1,1).
    # If we apply Block (0,0,0), we remove schoolbook terms for these 8.
    # If we then apply Block (1,0,0), we try to remove schoolbook terms for its 8.
    # The 4 overlapping terms are ALREADY removed.
    # So we would only remove the 4 non-overlapping ones.
    # And we add 7 new terms.
    # Net change: -4 (removed) + 7 (added) = +3. Rank INCREASES.
    # So we should only apply a replacement if ALL 8 schoolbook terms for that block are currently active.
    # If some are already removed (by previous replacements), we can't use the full 7-term Strassen replacement for the whole block, because the Strassen identity assumes the 8 terms are present and summed to 1.
    # If 4 are gone, the remaining 4 schoolbook terms sum to 1 for those 4 indices, but 0 for the others?
    # No, schoolbook terms are orthogonal. Each term (i,j,k) has value 1 only at that index.
    # So if we remove 4 of the 8, the sum for those 4 indices becomes 0 (from schoolbook) + 0 (from other blocks) = 0.
    # We need them to be 1.
    # So we MUST ensure that when we apply a Strassen block, all 8 corresponding schoolbook terms are still active.
    
    # Strategy:
    # 1. Start with all schoolbook terms active.
    # 2. Iterate through possible blocks.
    # 3. If all 8 terms for a block are active, apply the replacement:
    #    - Mark 8 terms as inactive (remove from active set).
    #    - Add 7 new rank-1 terms to U, V, W.
    #    - Increase rank by 7.
    #    - Decrease "active schoolbook count" by 8.
    #    - Net rank change: -1.
    # 4. To maximize reduction, we want to find a set of non-overlapping (in terms of active schoolbook terms) blocks.
    #    Since any two blocks that share any (i,j,k) index will conflict (the second one won't find all 8 active if the first one removed some), we are effectively looking for a packing of 2x2x2 cubes in the 4x5x5 grid.
    #    
    #    Max number of non-overlapping 2x2x2 cubes in 4x5x5:
    #    Volume 100. Cube volume 8. Max 12 cubes? 12*8 = 96.
    #    Can we pack 12?
    #    4x5x5.
    #    Split into 4x2x2 blocks?
    #    i: 0-3. j: 0-4. k: 0-4.
    #    We can place cubes at:
    #    i=0,1: j=0,1; k=0,1 (Cube 1)
    #    i=0,1: j=0,1; k=2,3 (Cube 2)
    #    i=0,1: j=2,3; k=0,1 (Cube 3)
    #    i=0,1: j=2,3; k=2,3 (Cube 4)
    #    i=2,3: j=0,1; k=0,1 (Cube 5)
    #    i=2,3: j=0,1; k=2,3 (Cube 6)
    #    i=2,3: j=2,3; k=0,1 (Cube 7)
    #    i=2,3: j=2,3; k=2,3 (Cube 8)
    #    Remaining: j=4, k=0-4, i=0-3 (4x1x5 strip) and j=0-3, k=4, i=0-3 (4x4x1 strip).
    #    These are too thin for 2x2x2.
    #    So 8 cubes.
    #    Rank reduction: 8 * 1 = 8.
    #    Initial rank 100. New rank 92.
    #    
    #    Can we do better with mixed sizes?
    #    The current program uses 49 (Strassen 4x4x4) + 36 (schoolbook) = 85.
    #    My greedy 2x2x2 packing gives 92. That's WORSE.
    #    
    #    Why is 4x4x4 better?
    #    4x4x4 Strassen uses 49 terms for the 4x4x4 sub-tensor (64 terms).
    #    Reduction: 64 - 49 = 15.
    #    My 2x2x2 packing on 4x4x4 region:
    #    4x4x4 can be packed with 8 2x2x2 cubes.
    #    Reduction: 8 * 1 = 8.
    #    49 is much better than 64-8=56.
    #    
    #    So we should use 4x4x4 Strassen where possible.
    #    
    #    Let's try to pack 4x4x4 blocks.
    #    In 4x5x5:
    #    We can fit one 4x4x4 block at (0,0,0). Covers i=0-3, j=0-3, k=0-3.
    #    Terms covered: 4*4*4 = 64.
    #    Replace with 49 Strassen terms.
    #    Remaining:
    #    j=4 (all i,k): 4*1*5 = 20 terms.
    #    k=4, j=0-3 (all i): 4*4*1 = 16 terms.
    #    Total remaining schoolbook: 36.
    #    Total rank: 49 + 36 = 85.
    #    This matches the current program!
    #    
    #    Can we improve?
    #    What if we use multiple 4x4x4 blocks?
    #    We only have space for one 4x4x4 block in 4x5x5 (since 4+4 > 5).
    #    
    #    What if we use 2x2x2 Strassen on the remainder?
    #    Remainder is L-shaped:
    #    1. Strip A: i=0-3, j=4, k=0-4. Size 4x1x5. No 2x2x2 fits (j is 1).
    #    2. Strip B: i=0-3, j=0-3, k=4. Size 4x4x1. No 2x2x2 fits (k is 1).
    #    So no further 2x2x2 Strassen can be applied to the remainder if we use one big 4x4x4.
    #    
    #    Alternative: Use TWO 2x2x4 blocks? Or 2x4x4?
    #    Strassen works for 2x2x2.
    #    Can we use "Strassen-like" for 2x2x4?
    #    If we split B into 2x2 and 2x2 blocks along k?
    #    B is 5x5.
    #    If we take i=0-1, j=0-1.
    #    k=0-1: 2x2x2 Strassen (7 terms).
    #    k=2-3: 2x2x2 Strassen (7 terms).
    #    k=4: schoolbook (2x2x1 = 4 terms).
    #    Total for i=0-1, j=0-1: 7+7+4 = 18 terms.
    #    Schoolbook would be 2*2*5 = 20 terms.
    #    Saving: 2 terms.
    #    
    #    Let's try to cover the whole 4x5x5 with such "slices".
    #    Divide A (4x5) into 2x2 blocks:
    #    i=0-1, j=0-1
    #    i=0-1, j=2-3
    #    i=0-1, j=4 (1x? No, 2x1)
    #    i=2-3, j=0-1
    #    i=2-3, j=2-3
    #    i=2-3, j=4 (2x1)
    #    
    #    For each 2x2 block of A, we can apply Strassen on B's 2x2 blocks.
    #    B is 5x5.
    #    For fixed i-block (2x2) and j-block (2x2):
    #    We need to multiply A_block (2x2) by B_block (2x5) to get C_block (5x2).
    #    Wait, C is 5x4.
    #    C indices: k (0-4), i (0-3).
    #    For fixed i-block (2 rows of A, 2 cols of A), we get 2 rows of C?
    #    No, A is n x m. B is m x p. C is p x n.
    #    A[i,j], B[j,k], C[k,i].
    #    If we fix i in {i0, i0+1} and j in {j0, j0+1}:
    #    We are computing C[k, i0] and C[k, i0+1] for all k.
    #    The terms involved are A[i0..i0+1, j0..j0+1] and B[j0..j0+1, k].
    #    This is a 2x2 (A) * 2x5 (B) -> 5x2 (C) multiplication.
    #    We can decompose B's columns (k) into 2x2 blocks + remainder.
    #    k=0-1: 2x2x2 Strassen. (7 terms)
    #    k=2-3: 2x2x2 Strassen. (7 terms)
    #    k=4: 2x2x1 Schoolbook. (4 terms)
    #    Total: 18 terms for this 2x2x5 sub-problem.
    #    Schoolbook: 2*2*5 = 20 terms.
    #    Saving: 2.
    #    
    #    How many such 2x2 blocks of A (i,j) do we have?
    #    i: 0-3. j: 0-4.
    #    Blocks:
    #    (0,0), (0,1) -> i=0-1, j=0-1
    #    (0,2), (0,3) -> i=0-1, j=2-3
    #    (0,4) -> i=0-1, j=4 (2x1)
    #    (2,0), (2,1) -> i=2-3, j=0-1
    #    (2,2), (2,3) -> i=2-3, j=2-3
    #    (2,4) -> i=2-3, j=4 (2x1)
    #    
    #    For the 2x1 blocks (j=4):
    #    A is 2x1. B is 1x5. C is 5x2.
    #    Terms: 2*1*5 = 10.
    #    Can we reduce?
    #    It's just outer products.
    #    A[i0,4]*B[4,k] + A[i0+1,4]*B[4,k]...
    #    Actually, for 2x1x5, it's rank 10?
    #    U has 2x1=2 rows. V has 1x5=5 rows. W has 5x2=10 rows.
    #    Tensor is A_{i,j} B_{j,k} C_{k,i} with j fixed.
    #    It's a sum of 5 terms (one for each k).
    #    Wait, schoolbook is 10 terms (2 i's * 1 j * 5 k's).
    #    But since j is fixed, A_{i,j} is just a vector of 2.
    #    B_{j,k} is a vector of 5.
    #    C_{k,i} is the output.
    #    C_{k,i} = sum_j A_{i,j} B_{j,k}.
    #    If j is fixed, C_{k,i} = A_{i,j} B_{j,k}.
    #    This is an outer product of A_col (size 2) and B_row (size 5).
    #    Rank of outer product is 1?
    #    No, the tensor is T_{i,k} = A_i B_k.
    #    This is rank 1.
    #    So we can replace 10 schoolbook terms with 1 term?
    #    Let's check.
    #    Term: u[i] = A_i, v[k] = B_k, w[i,k] = 1?
    #    Sum_r U[i,r] V[j,r] W[k,r]...
    #    Wait, the decomposition is for the TENSOR of multiplication.
    #    T_{i,j,k} = delta_{i,j,k}? No.
    #    The tensor is defined by T_{i,j,k} = 1 if we are computing C_{k,i} from A_{i,j} B_{j,k}.
    #    Actually, the tensor is T_{i,j,k} such that C_{k,i} = sum_{j} A_{i,j} B_{j,k}.
    #    The "multiplication tensor" has entries T_{i,j,k} = 1.
    #    Wait, no. The tensor is the structure of the bilinear map.
    #    The rank-1 terms are u, v, w such that sum u_{ij} v_{jk} w_{ki} reproduces the map.
    #    
    #    For the sub-problem with fixed j=4:
    #    We need to compute C_{k, i0} and C_{k, i0+1} for all k.
    #    C_{k, i0} = A_{i0, 4} B_{4, k}
    #    C_{k, i0+1} = A_{i0+1, 4} B_{4, k}
    #    
    #    Can we do this with 1 rank-1 term?
    #    u_{i,j} is non-zero only at j=4.
    #    v_{j,k} is non-zero only at j=4.
    #    w_{k,i} is non-zero for i in {i0, i0+1}.
    #    
    #    Let u_{i0,4} = 1, u_{i0+1,4} = 1.
    #    Let v_{4,k} = 1 for all k.
    #    Then sum_r u_{i,j} v_{j,k} w_{k,i} = u_{i,4} * 1 * w_{k,i}.
    #    We want this to be A_{i,4} B_{4,k}.
    #    This requires w_{k,i} to depend on A and B, which is not allowed (w is constant).
    #    
    #    So we cannot use 1 term.
    #    We need to separate the k's.
    #    For each k, we have a term.
    #    u_{i,4} = 1, v_{4,k} = 1, w_{k,i} = 1?
    #    Then sum = 1*1*1 = 1.
    #    But we want A_{i,4} B_{4,k}.
    #    The coefficients A and B are in the INPUTS, not in the decomposition.
    #    The decomposition is fixed.
    #    The formula is C_{k,i} = sum_{j} sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} A_{i,j} B_{j,k}.
    #    Wait, the standard definition is:
    #    C_{k,i} = sum_{j} A_{i,j} B_{j,k}.
    #    Rank-1 term r contributes: U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} A_{i,j} B_{j,k}?
    #    No.
    #    The decomposition is:
    #    sum_{j} A_{i,j} B_{j,k} = sum_{r} (sum_{i,j} U_{i,j}^{(r)} A_{i,j}) (sum_{j,k} V_{j,k}^{(r)} B_{j,k}) W_{k,i}^{(r)} ?
    #    No, that's not right either.
    #    
    #    Correct definition:
    #    The multiplication tensor T has entries T_{i,j,k}.
    #    We want to write T as sum of rank-1 tensors.
    #    Rank-1 tensor is u \otimes v \otimes w.
    #    T_{i,j,k} = sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}.
    #    And the computation is:
    #    C_{k,i} = sum_{j} A_{i,j} B_{j,k} = sum_{j} sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} A_{i,j} B_{j,k} ?
    #    No.
    #    The identity is:
    #    sum_{j} A_{i,j} B_{j,k} = sum_{r} (sum_{i',j'} U_{i',j'}^{(r)} A_{i',j'}) (sum_{j',k'} V_{j',k'}^{(r)} B_{j',k'}) ... No.
    #    
    #    Let's stick to the tensor equation:
    #    T_{i,j,k} = 1 for all i,j,k.
    #    We need sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = 1 for all i,j,k.
    #    
    #    For the j=4 case:
    #    We need sum_{r} U_{i,4}^{(r)} V_{4,k}^{(r)} W_{k,i}^{(r)} = 1 for all i in {i0, i0+1}, k in {0..4}.
    #    This is a 2x5 matrix of 1s.
    #    We can decompose a matrix of 1s into rank 1?
    #    J = 1 * 1^T. Rank 1.
    #    So U_{i,4}^{(1)} = 1, V_{4,k}^{(1)} = 1, W_{k,i}^{(1)} = 1?
    #    Then product is 1.
    #    So YES, we can replace 10 schoolbook terms (2 i's * 5 k's) with 1 term!
    #    
    #    Let's verify.
    #    Term r:
    #    U_{i,4} = 1 for i in {i0, i0+1}.
    #    V_{4,k} = 1 for all k.
    #    W_{k,i} = 1 for i in {i0, i0+1}.
    #    Sum over r: 1 * 1 * 1 = 1.
    #    Matches T_{i,4,k} = 1.
    #    
    #    So for the 2x1 strips, we can reduce 10 terms to 1 term.
    #    Saving: 9 terms per strip.
    #    
    #    Let's re-calculate the total rank with this hybrid approach.
    #    
    #    1. Four 2x2 blocks of A (i,j) where j in {0..3}:
    #       (0,0), (0,1) -> i=0-1, j=0-1
    #       (0,2), (0,3) -> i=0-1, j=2-3
    #       (2,0), (2,1) -> i=2-3, j=0-1
    #       (2,2), (2,3) -> i=2-3, j=2-3
    #       
    #       For each 2x2 A-block:
    #       We split B (2x5) into 2x2 (k=0-1), 2x2 (k=2-3), 2x1 (k=4).
    #       - k=0-1: 2x2x2 Strassen (7 terms)
    #       - k=2-3: 2x2x2 Strassen (7 terms)
    #       - k=4: 2x2x1 "Outer" (1 term? No, 2x2x1 is not outer product of 1s.
    #         Wait, for k=4, j in {j0, j0+1}.
    #         We need T_{i,j,4} = 1.
    #         i in 2, j in 2.
    #         Matrix of 1s (2x2). Rank 1.
    #         So 1 term.
    #       
    #       Total per 2x2 A-block: 7 + 7 + 1 = 15 terms.
    #       Schoolbook per 2x2 A-block: 2*2*5 = 20 terms.
    #       Saving: 5 terms.
    #       
    #       Total for 4 blocks: 4 * 15 = 60 terms.
    #       Schoolbook for these 4 blocks: 4 * 20 = 80 terms.
    #       
    #    2. Two 2x1 strips of A (j=4):
    #       (0,4) -> i=0-1, j=4
    #       (2,4) -> i=2-3, j=4
    #       
    #       For each 2x1 A-strip:
    #       We have T_{i,4,k} = 1 for i in 2, k in 5.
    #       Matrix of 1s (2x5). Rank 1.
    #       So 1 term.
    #       
    #       Total for 2 strips: 2 * 1 = 2 terms.
    #       Schoolbook for these 2 strips: 2 * 10 = 20 terms.
    #       
    #    Total Rank: 60 + 2 = 62.
    #    Schoolbook Rank: 100.
    #    Improvement: 100 -> 62.
    #    Current program: 85.
    #    This is a HUGE improvement!
    
    # Let's implement this specific construction.
    
    # Reset to schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    term_map = {}
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                term_map[(i, j, k)] = r
                r += 1
                
    # We will build a NEW decomposition from scratch to avoid confusion.
    U_new = []
    V_new = []
    W_new = []
    
    def add_term(u, v, w):
        U_new.append(u)
        V_new.append(v)
        W_new.append(w)
        
    # 1. Process 2x2 A-blocks
    # i0 in [0, 2], j0 in [0, 2]
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            # i in {i0, i0+1}, j in {j0, j0+1}
            # k in {0..4}
            
            # k=0,1: Strassen 2x2x2
            # i' = i - i0, j' = j - j0, k' = k - 0
            for t in strassen_2x2x2:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for idx, val in enumerate(t[0]):
                    i_loc = idx // 2
                    j_loc = idx % 2
                    u[(i0 + i_loc) * m + (j0 + j_loc)] = val
                    
                for idx, val in enumerate(t[1]):
                    j_loc = idx // 2
                    k_loc = idx % 2
                    v[(j0 + j_loc) * p + k_loc] = val
                    
                for idx, val in enumerate(t[2]):
                    k_loc = idx // 2
                    i_loc = idx % 2
                    w[k_loc * n + (i0 + i_loc)] = val
                    
                add_term(u, v, w)
                
            # k=2,3: Strassen 2x2x2
            for t in strassen_2x2x2:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for idx, val in enumerate(t[0]):
                    i_loc = idx // 2
                    j_loc = idx % 2
                    u[(i0 + i_loc) * m + (j0 + j_loc)] = val
                    
                for idx, val in enumerate(t[1]):
                    j_loc = idx // 2
                    k_loc = idx % 2
                    v[(j0 + j_loc) * p + (2 + k_loc)] = val
                    
                for idx, val in enumerate(t[2]):
                    k_loc = idx // 2
                    i_loc = idx % 2
                    w[(2 + k_loc) * n + (i0 + i_loc)] = val
                    
                add_term(u, v, w)
                
            # k=4: Rank-1 term for 2x2x1
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # U_{i,j} = 1 for i in {i0, i0+1}, j in {j0, j0+1}
            for i_loc in range(2):
                for j_loc in range(2):
                    u[(i0 + i_loc) * m + (j0 + j_loc)] = 1
                    
            # V_{j,k} = 1 for j in {j0, j0+1}, k=4
            for j_loc in range(2):
                v[(j0 + j_loc) * p + 4] = 1
                
            # W_{k,i} = 1 for k=4, i in {i0, i0+1}
            for i_loc in range(2):
                w[4 * n + (i0 + i_loc)] = 1
                
            add_term(u, v, w)
            
    # 2. Process 2x1 A-strips (j=4)
    for i0 in [0, 2]:
        # i in {i0, i0+1}, j=4, k in {0..4}
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        # U_{i,4} = 1
        for i_loc in range(2):
            u[(i0 + i_loc) * m + 4] = 1
            
        # V_{4,k} = 1 for all k
        for k in range(p):
            v[4 * p + k] = 1
            
        # W_{k,i} = 1 for all k, i in {i0, i0+1}
        for k in range(p):
            for i_loc in range(2):
                w[k * n + (i0 + i_loc)] = 1
                
        add_term(u, v, w)
        
    # Verify the new decomposition
    rank_new = len(U_new[0])
    U_mat = [row[:] for row in [[0]*rank_new for _ in range(n*m)]]
    V_mat = [row[:] for row in [[0]*rank_new for _ in range(m*p)]]
    W_mat = [row[:] for row in [[0]*rank_new for _ in range(p*n)]]
    
    for r_idx in range(rank_new):
        for i in range(n*m):
            U_mat[i][r_idx] = U_new[r_idx][i]
        for j in range(m*p):
            V_mat[j][r_idx] = V_new[r_idx][j]
        for k in range(p*n):
            W_mat[k][r_idx] = W_new[r_idx][k]
            
    if verify(U_mat, V_mat, W_mat):
        return {'U': U_mat, 'V': V_mat, 'W': W_mat}
        
    # Fallback
    return {'U': U, 'V': V, 'W': W}
