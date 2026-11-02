import numpy as np
import time
from itertools import product as iproduct

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 2, 4
    
    # Try rank 13 first, then fall back to rank 14
    for target_rank in [13, 14]:
        result = _try_rank(target_rank, deadline)
        if result is not None:
            return result
    
    return _schoolbook()

def _try_rank(rank, deadline):
    """Try to construct a rank-`rank` decomposition.
    
    Strategy: Use the two-Strassen-block structure but allow sharing.
    For rank 13, we need to find a decomposition that shares 1 product between blocks.
    
    Actually, let's use a systematic approach: the 2x2*2x4 tensor can be seen as
    two 2x2*2x2 tensors. We use Strassen for each, but try to merge them.
    
    A known rank-13 construction for 2x2*2x4 exists. Let's try to find it
    by exploiting the structure where A is shared.
    
    Key idea: In the rank-14 solution, the A-factors (U columns) for block 1 and
    block 2 are identical patterns. If we can find a W that uses shared products,
    we can reduce rank.
    
    Let me try a direct construction based on known results.
    """
    n, m, p = 2, 2, 4
    U = np.zeros((4, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    if rank == 14:
        # Standard two-Strassen-block construction
        _build_rank14(U, V, W)
    elif rank == 13:
        # Try to find a rank-13 construction
        # The 2x2*2x4 matrix multiplication tensor has bilinear complexity 13.
        # Construction: Use 7 products for the "left" 2x2*2x2 (k=0,1) and
        # 6 products for the "right" 2x2*2x2 (k=2,3), sharing 1 product.
        # 
        # Actually, let me try a different approach. The known rank-13 solution
        # for 2x2*2x4 uses the following structure:
        # 
        # We can write the 2x4 B matrix as [B_left | B_right] where each is 2x2.
        # The A matrix is 2x2.
        # 
        # One known construction (from literature on small matrix multiplication):
        # Use Strassen-like formulas but with shared terms.
        
        # Let me try a parameterized search.
        # The U matrix has 4 rows, V has 8 rows, W has 8 rows, each with `rank` columns.
        # Each column r defines a rank-1 term: U[:,r] * V[:,r] * W[:,r]
        # 
        # For the tensor T[i,j,k] = delta(i,j,k) where the multiplication is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor entry is T[i,j,k] = 1 if this term is in the sum, else 0.
        # But actually the tensor is: T_{i,j,k} = 1 for all i,j,k (it's the
        # matrix multiplication tensor), and we need sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
        # 
        # Wait, the convention is:
        # U shape (n*m, rank) = (4, rank)
        # V shape (m*p, rank) = (8, rank)  
        # W shape (p*n, rank) = (8, rank)
        # 
        # The decomposition: sum_r U[2*i+j, r] * V[4*j+k, r] * W[2*k+i, r] = 1
        # for all i in {0,1}, j in {0,1}, k in {0,1,2,3}
        # 
        # This is the matrix multiplication tensor for (2x2) * (2x4).
        
        # Let me try to construct rank 13 directly.
        # Based on the structure, I'll try:
        # 7 products for left block (k=0,1) using Strassen
        # 6 products for right block (k=2,3) using a modified Strassen
        # that shares 1 product with the left block.
        
        # Actually, the simplest approach: try all possible "merge" patterns.
        # But that's complex. Let me just try the known rank-13 construction.
        
        # From the literature, a rank-13 construction for 2x2*2x4:
        # The idea is to use the fact that we can write:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
        # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
        
        # I'll try a direct construction. Let me use the approach where
        # we define 13 rank-1 terms.
        
        success = _build_rank13(U, V, W)
        if not success:
            return None
    
    # Verify
    if not _verify(U, V, W):
        return None
    
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify(U_r, V_r, W_r):
        U, V, W = U_r, V_r, W_r
    
    if time.time() > deadline - 5:
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank14(U, V, W):
    """Standard two-Strassen-block construction, rank 14."""
    n, m, p = 2, 2, 4
    rank = 14
    
    # Block 1 (k=0,1)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[5,0] = 1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    U[0,2] = 1
    V[1,2] = 1; V[5,2] = -1
    U[3,3] = 1
    V[4,3] = 1; V[0,3] = -1
    U[0,4] = 1; U[1,4] = 1
    V[5,4] = 1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    U[1,6] = 1; U[3,6] = -1
    V[4,6] = 1; V[5,6] = 1
    
    # Block 2 (k=2,3)
    U[0,7] = 1; U[3,7] = 1
    V[2,7] = 1; V[7,7] = 1
    U[2,8] = 1; U[3,8] = 1
    V[2,8] = 1
    U[0,9] = 1
    V[3,9] = 1; V[7,9] = -1
    U[3,10] = 1
    V[6,10] = 1; V[2,10] = -1
    U[0,11] = 1; U[1,11] = 1
    V[7,11] = 1
    U[2,12] = 1; U[0,12] = -1
    V[2,12] = 1; V[3,12] = 1
    U[1,13] = 1; U[3,13] = -1
    V[6,13] = 1; V[7,13] = 1
    
    # W construction
    c00 = [1, 0, 0, 1, -1, 0, 1]
    c10 = [0, 1, 0, 1, 0, 0, 0]
    c01 = [0, 0, 1, 0, 1, 0, 0]
    c11 = [1, -1, 1, 0, 0, 1, 0]
    
    for m in range(7):
        W[0, m] = c00[m]
        W[1, m] = c10[m]
        W[2, m] = c01[m]
        W[3, m] = c11[m]
        W[4, 7+m] = c00[m]
        W[5, 7+m] = c10[m]
        W[6, 7+m] = c01[m]
        W[7, 7+m] = c11[m]

def _build_rank13(U, V, W):
    """Try to construct a rank-13 decomposition.
    
    Strategy: The 2x2*2x4 problem can be decomposed using 13 rank-1 terms.
    We use a construction that shares structure between the two 2x2*2x2 blocks.
    
    Key insight: We can use 7 Strassen products for the first block and
    find 6 additional products for the second block that together with
    the first 7 give a valid decomposition.
    
    Actually, a cleaner approach: use the fact that the tensor is a "tensor
    product" structure. Let me try a direct construction.
    """
    n, m, p = 2, 2, 4
    rank = 13
    
    # I'll try a construction based on the following idea:
    # Use 4 "direct" products and 9 "Strassen-like" products.
    # 
    # Actually, let me try a systematic approach using the known
    # rank-13 formula for 2x2*2x4.
    # 
    # The construction uses the following 13 rank-1 terms:
    # 
    # Term 0: A00 * B00 -> contributes to C00
    # Term 1: A11 * B11 -> contributes to C11
    # Term 2: (A00+A11) * (B00+B11) -> contributes to C00, C11
    # ... etc.
    # 
    # Let me just try a few known constructions.
    
    # Construction attempt 1: Based on sharing Strassen products
    # Left block (k=0,1) uses Strassen products 0-6
    # Right block (k=2,3) uses products 7-12 (6 products)
    # But we need to verify this works.
    
    # Let me try a different approach: use the full Strassen for left
    # and a modified version for right that reuses one product.
    
    # Actually, I recall that the bilinear complexity of 2x2*2x4 is indeed 13.
    # Let me try to construct it.
    
    # The 2x2*2x4 tensor can be written as the sum of two 2x2*2x2 tensors
    # (for k=0,1 and k=2,3). Each 2x2*2x2 has complexity 7, so naive sum is 14.
    # To get 13, we need to share one rank-1 term.
    
    # A rank-1 term is defined by (u, v, w) where u in R^4, v in R^8, w in R^8.
    # For the term to contribute to both blocks, w must have non-zero entries
    # in both the left (rows 0-3) and right (rows 4-7) parts.
    
    # Let me try: use 7 Strassen products for left block, and for the right
    # block, use 6 new products plus modify one of the left products to
    # also contribute to the right block.
    
    # This is getting complex. Let me just try to verify if a rank-13
    # solution exists by checking if the current rank-14 can be compressed.
    
    # Alternative: try all possible ways to merge one column from left
    # and one from right.
    
    # For now, let's try a specific construction:
    # 
    # Products for left block (k=0,1):
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    #
    # Products for right block (k=2,3):
    # P7 = (A00+A11)(B02+B13)
    # P8 = (A10+A11)B02
    # P9 = A00(B03-B13)
    # P10 = A11(B12-B02)
    # P11 = (A00+A01)B13
    # P12 = (A10-A00)(B02+B03)
    #
    # And we need P6 to also help with the right block, or find a
    # different set.
    
    # Let me try: can we express the right block using 6 products?
    # The 2x2*2x2 tensor has rank 7, so it cannot be done with 6.
    # But maybe we can use a non-Strassen approach that shares with left.
    
    # I think the key is that we don't need to use Strassen separately.
    # We can use a global decomposition.
    
    # Let me try a parameterized search with small integers.
    # This might be too slow, so let me just try a few hand-crafted constructions.
    
    # Construction: Try to use the following 13 terms:
    # 
    # I'll use a construction from the theory of matrix multiplication.
    # The 2x2*2x4 tensor is equivalent to computing 8 bilinear forms.
    
    # Let me just return False for now and fall back to rank 14.
    # The rank-14 solution is valid and scores 1.1429.
    
    return False

def _verify(U, V, W):
    """Verify the decomposition is exact."""
    n, m, p = 2, 2, 4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[2*i+j, :] * V[4*j+k, :] * W[2*k+i, :])
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _schoolbook():
    n, m, p = 2, 2, 4
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
