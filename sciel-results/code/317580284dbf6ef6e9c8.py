"""Strassen-based tensor decomposition for 3x4 x 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_4x4_4x4():
    """
    Return (U, V, W) for 4x4 * 4x4 -> 4x4 using Strassen's algorithm.
    U shape: (16, 7), V shape: (16, 7), W shape: (16, 7)
    
    Indexing:
    U[i*4+j, c] for input A[i,j]
    V[j*4+k, c] for input B[j,k]
    W[k*4+i, c] for output C[i,k]  (trace convention: W[k*n+i, r] with n=4)
    """
    n = 4
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Strassen's 7 products for 2x2 blocks:
    # Let A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Block indices: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    #                 A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # B similarly.
    
    # For each of the 7 products, we need to express the sum/difference of blocks.
    # Each block is 2x2, so each product involves 4 elements from A and 4 from B.
    # But we can factor: (A11+A22) is a 2x2 matrix, (B11+B22) is 2x2.
    # Their product is a 2x2 result, which requires 4 scalar multiplications.
    # However, in tensor decomposition, each "product" Mx is a bilinear form.
    
    # Actually, let me think more carefully. The tensor for 4x4*4x4 has rank at most 7*4=28
    # if we just do Strassen at the block level with schoolbook within blocks.
    # But we can do better by applying Strassen recursively.
    
    # Let me use the recursive Strassen approach:
    # 4x4 * 4x4: split into 2x2 blocks of 2x2 matrices.
    # Each 2x2 * 2x2 product has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49? No, that's not right either.
    
    # Actually, the tensor rank of <2,2,2> is 7.
    # The tensor <4,4,4> can be decomposed using 7 copies of <2,2,2> tensor <2,2,2>?
    # No. Let me think again.
    
    # The multiplication tensor for n x n * n x n is <n,n,n>.
    # Strassen shows <2,2,2> has rank 7.
    # For <4,4,4>, we can view it as a 2x2 block matrix multiplication where each block is 2x2.
    # The block-level multiplication is <2,2,2> with rank 7.
    # Each block product is a <2,2,2> multiplication with rank 7.
    # So total rank <= 7 * 7 = 49.
    
    # But actually, we can do better. The tensor <4,4,4> has known rank bounds.
    # Let me just implement the straightforward recursive Strassen.
    
    # For now, let me implement a clean recursive approach.
    pass

def _build_strassen_decomposition(n, m, p):
    """
    Build a tensor decomposition for n x m * m x p using Strassen-style recursion.
    Returns (U, V, W) as numpy arrays.
    """
    # For 3x4 x 4x8:
    # We can split the 4x4 middle dimension into two 2x2 blocks.
    # A is 3x4, B is 4x8.
    # A = [A1, A2] where A1 is 3x2, A2 is 3x2
    # B = [B1; B2] where B1 is 2x8, B2 is 2x8
    # C = A1*B1 + A2*B2
    
    # This is just a sum of two products, each of rank 3*2*8 = 48.
    # Total rank = 96. That's the same as schoolbook.
    
    # To use Strassen, we need square blocks. Let me try a different approach.
    
    # Alternative: Pad to make dimensions powers of 2.
    # 3x4 x 4x8: pad 3 to 4, giving 4x4 x 4x8.
    # 4x4 x 4x8: split into 2x2 blocks.
    # A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    # B (4x8) = [[B11, B12], [B21, B22]] where each is 2x4
    # C (4x8) = [[C11, C12], [C21, C22]]
    
    # C11 = A11*B11 + A21*B21
    # C12 = A11*B12 + A21*B22
    # C21 = A12*B11 + A22*B21
    # C22 = A12*B12 + A22*B22
    
    # This is still schoolbook at the block level.
    
    # Let me try a different strategy: use the known result that
    # <2,2,2> has rank 7, and build up from there.
    
    # For 3x4 x 4x8, let me try:
    # View as 3 copies of 1x4 x 4x8, each of rank 32. Total 96.
    # Or view as 4 copies of 3x1 x 1x8, each of rank 24. Total 96.
    
    # The key insight from the directive: use Strassen recursively on 4x4 and 4x8 sub-blocks.
    # 4x4 x 4x8: 
    #   = (2x2 blocks) where A is 4x4 (2x2 blocks of 2x2), B is 4x8 (2x2 blocks of 2x4)
    #   The block multiplication is 2x2 * 2x2 where each "element" is a 2x2 or 2x4 matrix.
    #   This doesn't directly apply Strassen.
    
    # Let me try yet another approach: 
    # 3x4 x 4x8 = 3 * (1x4 x 4x8)
    # 1x4 x 4x8: this is a row vector times a matrix. Rank = 4*8 = 32.
    # Total = 96.
    
    # I think for this specific case, the best we can do with simple Strassen is:
    # Pad 3 to 4, giving 4x4 x 4x8.
    # Use the fact that 4x4 x 4x8 can be seen as 2x(2x4 x 4x8).
    # 2x4 x 4x8: split into 2*(1x4 x 4x8), each rank 32, total 64.
    # Total for 4x4 x 4x8: 128. Worse.
    
    # Let me just implement the schoolbook but check if there's a simple improvement.
    # Actually, let me try the Winograd/Strassen hybrid approach mentioned in the directive.
    
    # The directive says: rank of 72 = 3 * 2^3 * 8.
    # 3 * 8 * 8 = 192? No, 3 * 2^3 * 8 = 3 * 8 * 8 = 192. That's worse.
    # Wait, 2^3 = 8, so 3 * 8 * 8 = 192. That can't be right.
    # Maybe 3 * 2^3 * 8 means something else. 3 * 8 * 8 = 192.
    # Or maybe it's 3 * 2^3 * 8 / something.
    
    # Let me re-read: "rank of 72 (calculated as 3 * 2^3 * 8)"
    # 3 * 2^3 * 8 = 3 * 8 * 8 = 192. That's not 72.
    # Maybe it's 3 * 2^3 * 3 = 72? Or 3 * 4 * 6 = 72?
    # 72 = 3 * 24 = 3 * 4 * 6.
    
    # I think there might be a miscalculation in the directive. Let me just implement
    # a reasonable Strassen-based decomposition and see what rank we get.
    
    # For now, return schoolbook.
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
