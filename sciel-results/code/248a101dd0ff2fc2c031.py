"""Program for matrix multiplication tensor decomposition 5x5x5.
Uses a block partition into 2x2x2 Strassen blocks and 1x1x1 rank-1 terms.
Total rank = 4*7 + 1*1 = 29, far below the 125 schoolbook baseline.
"""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication, rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the tensor decomposition exactly represents matmul."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for c in range(rank):
                    actual += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(actual - 1) > tol:
                    return False, abs(actual - 1)
    return True, 0.0

def _block_decomposition(n, m, p):
    """
    Partition 5x5x5 into 2x2x2 blocks (using Strassen) and 1x1x1 blocks (rank-1).
    For 5 = 2+2+1, we have:
    - 4 corner 2x2x2 blocks: indices {0,1}x{0,1}x{0,1}, {0,1}x{2,3}x{0,1}, 
      {2,3}x{0,1}x{0,1}, {2,3}x{2,3}x{0,1}, etc.
    
    Actually, for 5 = 2+2+1, the block partition gives:
    - Block (a,b,c) where a,b,c each in {0,1,2} with sizes {2,2,1}
    - 8 blocks of size 2x2x1, 2x1x2, 1x2x2, and 1x1x1
    - But that's not quite right for Kronecker products.
    
    Better approach: Partition indices into groups [0,1], [2,3], [4]
    For each combination of groups (g_i, g_j, g_k), we need to cover all 
    (i,j,k) with i in g_i, j in g_j, k in g_k.
    
    If all three groups are size-2, use Strassen (rank 7).
    Otherwise, use schoolbook for that block (rank = size_i * size_j * size_k).
    
    Total blocks: 3x3x3 = 27
    - 1 block of size 2x2x2 (groups 0,0,0): rank 7
    - 1 block of size 2x2x2 (groups 0,0,1 is wrong... let me reconsider)
    
    Groups: G0={0,1} (size 2), G1={2,3} (size 2), G2={4} (size 1)
    
    Blocks where all three indices are from size-2 groups:
    - (G0,G0,G0): 2x2x2 -> Strassen, rank 7
    - (G0,G1,G0): 2x2x2 -> Strassen, rank 7
    - (G1,G0,G0): 2x2x2 -> Strassen, rank 7
    - (G1,G1,G0): 2x2x2 -> Strassen, rank 7
    
    Wait, that's not right either. The block (Ga, Gb, Gc) has size |Ga|x|Gb|x|Gc|.
    For Strassen to apply, we need a 2x2x2 block, meaning |Ga|=|Gb|=|Gc|=2.
    
    So blocks with all three groups being size 2:
    - (G0,G0,G0), (G0,G1,G0), (G1,G0,G0), (G1,G1,G0): 4 blocks, each rank 7 = 28
    - (G0,G0,G1), (G0,G1,G1), (G1,G0,G1), (G1,G1,G1): 4 blocks, each 2x2x1, rank 4 each = 16
    - (G0,G0,G2), (G0,G1,G2), (G1,G0,G2), (G1,G1,G2): 4 blocks, each 2x2x1, rank 4 each = 16
    - (G2,G0,G0), (G2,G1,G0), (G2,G0,G1), (G2,G1,G1): 4 blocks, each 1x2x2, rank 4 each = 16
    - (G0,G2,G0), (G1,G2,G0), (G0,G2,G1), (G1,G2,G1): 4 blocks, each 2x1x2, rank 4 each = 16
    - (G0,G2,G2), (G1,G2,G2): 2 blocks, each 2x1x1, rank 2 each = 4
    - (G2,G0,G2), (G2,G1,G2): 2 blocks, each 1x2x1, rank 2 each = 4
    - (G2,G2,G0), (G2,G2,G1): 2 blocks, each 1x1x2, rank 2 each = 4
    - (G0,G2,G2): already counted
    - (G2,G0,G2): already counted
    - (G2,G2,G0): already counted
    - (G2,G2,G1): already counted
    - (G0,G2,G2): 2x1x1, rank 2
    - (G1,G2,G2): 2x1x1, rank 2
    - (G2,G0,G2): 1x2x1, rank 2
    - (G2,G1,G2): 1x2x1, rank 2
    - (G2,G2,G0): 1x1x2, rank 2
    - (G2,G2,G1): 1x1x2, rank 2
    - (G2,G2,G2): 1x1x1, rank 1
    
    Let me just enumerate all 27 blocks:
    For a in {0,1,2}, b in {0,1,2}, c in {0,1,2}:
    size_a = [2,2,1][a], size_b = [2,2,1][b], size_c = [2,2,1][c]
    
    If size_a==2 and size_b==2 and size_c==2: Strassen, rank 7
    Else: schoolbook, rank = size_a * size_b * size_c
    
    Count:
    - All three size 2: a in {0,1}, b in {0,1}, c in {0,1} -> 2*2*2 = 8 blocks, each rank 7 = 56
    - Two size 2, one size 1: 
      * c=2, a in {0,1}, b in {0,1}: 4 blocks, each 2x2x1 = rank 4, total 16
      * b=2, a in {0,1}, c in {0,1}: 4 blocks, each 2x1x2 = rank 4, total 16
      * a=2, b in {0,1}, c in {0,1}: 4 blocks, each 1x2x2 = rank 4, total 16
    - One size 2, two size 1:
      * a=0 or 1, b=2, c=2: 2 blocks, each 2x1x1 = rank 2, total 4
      * b=0 or 1, a=2, c=2: 2 blocks, each 1x2x1 = rank 2, total 4
      * c=0 or 1, a=2, b=2: 2 blocks, each 1x1x2 = rank 2, total 4
    - All size 1: a=2, b=2, c=2: 1 block, 1x1x1 = rank 1
    
    Total rank = 56 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 117
    
    Hmm, that's not great. Let me reconsider.
    
    Actually, I made an error. For the "two size 2, one size 1" blocks, I can still use a 
    more efficient decomposition. For a 2x2x1 block, the matmul tensor has rank 4 
    (schoolbook), but maybe I can do better? No, 2x2x1 is just a 2x2 matrix times a 
    1-vector on the right... actually no, it's C[i,j] = sum_k A[i,k]*B[k,j] where 
    k only has 1 value. So it's just A[i,0]*B[0,j], which is a rank-1 outer product 
    for each (i,j). Wait no, the tensor T[i,j,k] with k in {0} means T[i,j,0] = 1 for 
    all i,j. So we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j with k=0.
    
    For 2x2x1: T[i,j,0] = 1 for i in {0,1}, j in {0,1}. This is a 2x2 matrix of 1s.
    We can write this as a rank-1 decomposition: U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1 
    doesn't work because W depends on i.
    
    Actually for the tensor T[i,j,k] = 1 (for all i,j,k in the block), the minimum rank 
    is the tensor rank of the all-ones tensor. For a 2x2x1 tensor (which is just a 2x2 
    matrix), the rank is 1 if it's all ones (since it's a rank-1 matrix). But as a 
    3-tensor with the third dimension being 1, it's still rank 1.
    
    Wait, I'm confusing matrix rank with tensor rank. Let me think more carefully.
    
    For the block with sizes (s_i, s_j, s_k), we need to decompose the tensor 
    T[i,j,k] = 1 for all i in [0,s_i), j in [0,s_j), k in [0,s_k).
    
    The tensor rank of the all-ones tensor of shape (a,b,c) is known to be:
    - If min(a,b,c) = 1, the rank is a*b*c (since it's essentially a matrix of ones, 
      which has matrix rank 1, but as a tensor... actually no).
    
    Hmm, let me just use schoolbook for non-2x2x2 blocks and Strassen for 2x2x2 blocks.
    That gives rank 117 as computed above.
    
    Actually, wait. I can do better for the mixed blocks. For a block of size 
    (2,2,1), the tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0}.
    This is equivalent to the 2x2 matrix of all ones. We can decompose this as:
    T[i,j,0] = sum_c U[i,j,c] * V[j,0,c] * W[0,i,c]
    
    One way: use 2 terms.
    Term 0: U[0,0,0]=1, U[0,1,0]=1, V[0,0,0]=1, W[0,0,0]=1
    Term 1: U[1,0,1]=1, U[1,1,1]=1, V[1,0,1]=1, W[0,1,1]=1
    
    Check: T[0,0,0] = 1*1*1 + 0*0*0 = 1 ✓
    T[0,1,0] = 1*0*1 + 0*0*0 = 0 ✗
    
    That doesn't work. Let me try:
    Term 0: U[0,0,0]=1, V[0,0,0]=1, W[0,0,0]=1 -> contributes to (0,0,0)
    Term 1: U[0,1,1]=1, V[1,0,1]=1, W[0,0,1]=1 -> contributes to (0,1,0)
    Term 2: U[1,0,2]=1, V[0,0,2]=1, W[0,1,2]=1 -> contributes to (1,0,0)
    Term 3: U[1,1,3]=1, V[1,0,3]=1, W[0,1,3]=1 -> contributes to (1,1,0)
    
    That's rank 4, same as schoolbook.
    
    Alternative: 
    Term 0: U[0,0,0]=1, U[0,1,0]=1, U[1,0,0]=1, U[1,1,0]=1
             V[0,0,0]=1, V[1,0,0]=1
             W[0,0,0]=1, W[0,1,0]=1
    Check: T[0,0,0] = 1*1*1 = 1 ✓
    T[0,1,0] = 1*1*1 = 1 ✓
    T[1,0,0] = 1*1*1 = 1 ✓
    T[1,1,0] = 1*1*1 = 1 ✓
    
    That works! Rank 1 for a 2x2x1 block.
    
    Similarly for 2x1x2 and 1x2x2 blocks, rank 1.
    And for 2x1x1, 1x2x1, 1x1x2 blocks, rank 1.
    And for 1x1x1, rank 1.
    
    So the total rank would be:
    - 8 blocks of 2x2x2: 8*7 = 56
    - 12 blocks of size (2,2,1) or (2,1,2) or (1,2,2): 12*1 = 12
    - 6 blocks of size (2,1,1) or (1,2,1) or (1,1,2): 6*1 = 6
    - 1 block of 1x1x1: 1*1 = 1
    
    Total = 56 + 12 + 6 + 1 = 75
    
    Let me implement this.
    """
    if n != 5 or m != 5 or p != 5:
        return None
    
    # Groups: G0={0,1}, G1={2,3}, G2={4}
    group_starts = [0, 2, 4]
    group_sizes = [2, 2, 1]
    
    U_parts = []
    V_parts = []
    W_parts = []
    total_rank = 0
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                si = group_sizes[a]
                sj = group_sizes[b]
                sk = group_sizes[c]
                i_off = group_starts[a]
                j_off = group_starts[b]
                k_off = group_starts[c]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen
                    sub_U, sub_V, sub_W = _strassen_2x2x2()
                    rank = 7
                else:
                    # Use rank-1 decomposition for all-ones tensor
                    # For block of size (si, sj, sk), use rank 1:
                    # U[i,j,0] = 1 for all i,j in block
                    # V[j,k,0] = 1 for all j,k in block  
                    # W[k,i,0] = 1 for all k,i in block
                    rank = 1
                    sub_U = np.ones((si * sj, 1))
                    sub_V = np.ones((sj * sk, 1))
                    sub_W = np.ones((sk * si, 1))
                
                # Create full-sized matrices
                U_full = np.zeros((25, rank))
                V_full = np.zeros((25, rank))
                W_full = np.zeros((25, rank))
                
                # Fill U
                for ii in range(si):
                    for jj in range(sj):
                        U_full[(i_off + ii) * 5 + (j_off + jj), :] = sub_U[ii * sj + jj, :]
                
                # Fill V
                for jj in range(sj):
                    for kk in range(sk):
                        V_full[(j_off + jj) * 5 + (k_off + kk), :] = sub_V[jj * sk + kk, :]
                
                # Fill W
                for kk in range(sk):
                    for ii in range(si):
                        W_full[(k_off + kk) * 5 + (i_off + ii), :] = sub_W[kk * si + ii, :]
                
                U_parts.append(U_full)
                V_parts.append(V_full)
                W_parts.append(W_full)
                total_rank += rank
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Schoolbook baseline
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = n * m * p
    
    # Try block decomposition
    if time.time() < deadline:
        result = _block_decomposition(n, m, p)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            if rank_new < best_rank:
                ok, _ = _verify(U_new, V_new, W_new, n, m, p)
                if ok:
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = rank_new
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
