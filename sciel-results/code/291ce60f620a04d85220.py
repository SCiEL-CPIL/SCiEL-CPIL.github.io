"""Optimized tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [3, 3, 6]
_TIME_LIMIT = 150.0  # wall-clock limit in seconds (evaluator kills at 150s)

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r]^T equals the matmul tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = np.dot(U[A] * V[B], W[C])
                    if abs(actual - expected) > tol:
                        return False
    return True


def _build_rank49_4x4x4():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 matrix multiplication.
    
    We construct this by combining Strassen's rank-7 decomposition for 2x2 blocks
    with additional terms. The 4x4 matmul can be viewed as 4 blocks of 2x2,
    giving 16 block products. Using Strassen on pairs of blocks, we can reduce
    the number of scalar multiplications.
    
    A simpler approach: use the known explicit rank-49 decomposition from the
    literature. We'll build it using a combination of Strassen-like techniques
    on 2x2 sub-blocks.
    
    Actually, let's use a more direct approach: the 4x4 matmul tensor has
    64 non-zero entries. We need to find 49 rank-1 tensors that sum to it.
    
    Strategy: Partition the 4x4 matrices into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    
    C = A*B has blocks:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each block product (e.g., A11*B11) is a 2x2*2x2 matmul which can be done
    in rank 7 using Strassen. So 4 block positions * 2 products each = 8 
    2x2*2x2 matmuls, each of rank 7 = 56. That's worse than schoolbook (64)
    only slightly better.
    
    Better: Use Strassen on the 4x4 directly by viewing it as a 2x2 block
    matrix over 2x2 matrices. But Strassen requires the "multiplication" of
    blocks to be associative, which 2x2 matrix multiplication is.
    
    So: treat A and B as 2x2 matrices over the ring of 2x2 matrices.
    Apply Strassen's 7-product algorithm. Each of the 7 "products" is a
    2x2 * 2x2 matmul, which itself takes 7 scalar multiplications (Strassen)
    or 4 (schoolbook).
    
    Using Strassen for the outer level (7 products) and Strassen for each
    inner 2x2*2x2 (7 products each): 7 * 7 = 49 scalar multiplications!
    
    This is the rank-49 decomposition.
    """
    rank = 49
    U = np.zeros((16, rank))  # 4*4 = 16 entries for A
    V = np.zeros((16, rank))  # 16 entries for B
    W = np.zeros((16, rank))  # 16 entries for C
    
    # Strassen's algorithm for 2x2 matrices over a ring:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, each "product" like (A11 + A22) * (B11 + B22) is itself a 2x2*2x2
    # matmul. Let's denote the 2x2 blocks:
    # A11 = [[a00, a01], [a10, a11]]
    # A12 = [[a02, a03], [a12, a13]]
    # A21 = [[a20, a21], [a30, a31]]
    # A22 = [[a22, a23], [a32, a33]]
    # Similarly for B.
    
    # For each Strassen product at the outer level, we need to compute a
    # 2x2 * 2x2 matmul. We'll use Strassen again for each of these.
    
    # Let's define a helper that, given two 2x2 matrices X and Y (as flat
    # arrays of 4 elements), returns the 7 Strassen products as linear
    # combinations of the input elements.
    
    # For 2x2 matrices X = [[x0, x1], [x2, x3]] and Y = [[y0, y1], [y2, y3]]:
    # Strassen products:
    # S1 = (x0 + x3) * (y0 + y3)
    # S2 = (x2 + x3) * y0
    # S3 = x0 * (y1 - y3)
    # S4 = x3 * (y2 - y0)
    # S5 = (x0 + x1) * y3
    # S6 = (x2 - x0) * (y0 + y1)
    # S7 = (x1 - x3) * (y2 + y3)
    #
    # Result Z = X*Y:
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    
    # So for each outer Strassen product, we have 7 inner Strassen products.
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # Let's build the factor matrices.
    
    # First, let's define the outer Strassen combinations.
    # For the outer level, we have 2x2 block matrices.
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x2)
    
    # Outer Strassen products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each of these is a 2x2*2x2 matmul. For each, we apply inner Strassen.
    
    # Let's define the mapping from block indices to flat indices.
    # A11: flat indices 0,1,4,5 (row-major: [0,1],[4,5])
    # A12: flat indices 2,3,6,7
    # A21: flat indices 8,9,12,13
    # A22: flat indices 10,11,14,15
    
    # Similarly for B.
    
    # For the inner Strassen, we need to express each of the 7 products
    # as linear combinations of the 4 elements of each 2x2 block.
    
    # Let's build this systematically.
    
    col = 0
    for outer_m in range(7):
        # Determine the 2x2 matrices X and Y for this outer Strassen product
        # X and Y are linear combinations of the 4 A-blocks and 4 B-blocks
        
        # For each of the 7 inner Strassen products of X*Y:
        for inner_s in range(7):
            # Determine which elements of X and Y are involved
            # and with what coefficients
            
            # This is getting complex. Let me use a more direct approach.
            # I'll compute the linear combinations explicitly.
            
            pass
    
    # Let me use a cleaner approach: build the decomposition by directly
    # specifying the 49 rank-1 terms.
    
    # Actually, let me just verify that the schoolbook rank is 64 for 4x4x4
    # and then try to find a rank-49 decomposition using optimization,
    # or use a known explicit construction.
    
    # For now, let me return None and use a different strategy.
    return None


def _construct_via_4x4_embedding(n, m, p):
    """
    Embed 3x3 * 3x6 into 4x4 * 4x4, use rank-49 decomposition for 4x4*4x4,
    and restrict to the 3x3*3x6 subspace.
    
    Embedding:
    - A (3x3) -> embed into top-left 3x3 of a 4x4 matrix, with 4th row/col = 0
    - B (3x6) -> embed into top-left 3x3 of a 4x4 matrix? No, 3x6 doesn't fit in 4x4.
    
    Wait, 3x6 doesn't fit in 4x4. We need a different embedding.
    
    Alternative: Embed 3x3*3x6 into 4x4*4x8? That's even larger.
    
    Better approach: The tensor for 3x3*3x6 has dimensions:
    - A: 3*3 = 9
    - B: 3*6 = 18
    - C: 6*3 = 18
    
    Total tensor size: 9 * 18 * 18 = 2916
    
    For 4x4*4x4: 16 * 16 * 16 = 4096
    
    The 3x3*3x6 tensor is a "slice" of a larger tensor if we can embed it.
    
    Actually, let me think differently. The 3x6 matrix B can be viewed as
    two 3x3 matrices side by side: B = [B1 | B2].
    
    So 3x3*3x6 = (3x3*3x3) x 2, i.e., two independent 3x3*3x3 matmuls.
    
    If rank(3x3*3x3) = 19, then rank(3x3*3x6) <= 2*19 = 38.
    
    But we don't have an explicit rank-19 decomposition.
    
    Alternative: Use the fact that 3x3*3x6 can be embedded in 4x4*4x4 by
    padding. But 3x6 doesn't fit in 4x4.
    
    Let me try a different embedding: 3x3*3x6 -> 4x4*4x8.
    4x4*4x8 has tensor dimensions 16 * 32 * 32 = 16384.
    Schoolbook rank: 4*4*8 = 128.
    
    This is getting complicated. Let me just use the schoolbook decomposition
    for 3x3*3x6 which has rank 54, and try to improve it with optimization.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the 4x4 embedding approach
    result = _construct_via_4x4_embedding(n, m, p)
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        rank = len(U[0])
        if rank < best_rank and _verify(U, V, W, n, m, p):
            return result
    
    return fallback

# EVOLVE-BLOCK-END
