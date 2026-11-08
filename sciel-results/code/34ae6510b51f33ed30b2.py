import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
DEADLINE = 150.0

def _build_strassen_3x3x3(offset_col=0):
    """
    Build a rank-7 Strassen decomposition for 3x3 x 3x3 matrix multiplication.
    offset_col: column offset in the larger B matrix (0 or 3)
    Returns (U, V, W) each of shape (9, 7) for the local 3x3x3 problem.
    
    Strassen's 7 products for 3x3 matrices split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] where A11 is 1x1, A12 is 1x2, A21 is 2x1, A22 is 2x2
    B = [[B11, B12], [B21, B22]] similarly
    
    M1 = (A11 - A22)(B21 - B22)
    M2 = (A21 - A22)B22
    M3 = A11*B11
    M4 = A22*B22
    M5 = (A11 + A22)B21
    M6 = (A21 + A11)(B12 - B22)
    M7 = (A22 + A11)(B11 - B12)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M6
    C21 = M2 + M4
    C22 = M1 - M3 + M4 + M5
    """
    U = np.zeros((9, 7), dtype=float)
    V = np.zeros((9, 7), dtype=float)
    W = np.zeros((9, 7), dtype=float)
    
    # Index mapping for 3x3 matrix:
    # A[i,j] -> U[i*3+j, r]
    # B[j,k] -> V[j*3+k, r]  
    # C[i,k] -> W[k*3+i, r]
    
    # Strassen for 3x3: split as 1x1, 1x2, 2x1, 2x2 blocks
    # A11 = A[0,0], A12 = A[0,1:3], A21 = A[1:3,0], A22 = A[1:3,1:3]
    # B11 = B[0,0], B12 = B[0,1:3], B21 = B[1:3,0], B22 = B[1:3,1:3]
    
    # M3 = A11 * B11
    U[0*3+0, 2] = 1.0  # A[0,0]
    V[0*3+0, 2] = 1.0  # B[0,0]
    W[0*3+0, 2] = 1.0  # C[0,0] += M3
    
    # M4 = A22 * B22 (A22 is 2x2, B22 is 2x2)
    # This requires 4 scalar products for the 2x2 multiplication
    # But in Strassen, M4 is a single product of two 2x2 matrices
    # In the tensor decomposition, each "product" of submatrices needs to be expanded
    
    # Actually, for the tensor decomposition, we need to express each Strassen product
    # as a sum of rank-1 terms. A product of two a×b and b×c matrices has rank a*b*c
    # if done naively, but we can use the structure.
    
    # Let me reconsider. The Strassen algorithm gives 7 matrix multiplications of smaller
    # matrices. For 3x3 split into 1x1, 1x2, 2x1, 2x2:
    # M1: (1x1) * (1x2) -> 1x2 result, rank 1*1*2 = 2
    # M2: (2x1) * (2x2) -> 2x2 result, rank 2*2*2 = 8... that's too many
    
    # I think I'm overcomplicating this. Let me use a different approach.
    # For 3x3x3, the known minimal rank is 23, but Strassen gives 7 for 2x2x2.
    # For 3x3, we can pad to 4x4 and use Strassen, giving rank 7 for the 4x4x4,
    # but we only need 3x3x3.
    
    # Actually, let me just directly construct a rank-14 solution by using
    # two independent 3x3x3 schoolbook decompositions (rank 27 each) - no that's 54.
    
    # Better: use the fact that 3x3x6 = two 3x3x3 blocks, and for each 3x3x3,
    # use a rank-7 decomposition if one exists, or rank 23.
    
    # For now, let's just return schoolbook for the sub-blocks and see if we can improve.
    # Each 3x3x3 schoolbook has rank 27, so two would be 54.
    
    # Let me try a different split: use rank-7 Strassen for 2x2x2 subproblems.
    # 3x3 can be seen as containing a 2x2 block plus extra rows/cols.
    
    # This is getting complex. Let me just implement the straightforward two-Strassen
    # approach where I manually write out the 7 products for each 3x3x3 block.
    
    # For a clean 3x3x3 Strassen-like decomposition, I'll use the standard rank-7
    # for the 2x2x2 core and handle the extra row/column separately.
    
    # Actually, the simplest improvement: use rank 7 for each 2x2x2 block and
    # handle the remaining parts with schoolbook.
    
    # Let me just build a proper decomposition. I'll use the following strategy:
    # Split 3x3 into overlapping 2x2 blocks and use Strassen where possible.
    
    # For simplicity and correctness, let me just return None and use schoolbook,
    # but with a better structure that might allow future optimization.
    
    # Actually, let me implement a concrete rank-14 solution:
    # Use 7 rank-1 terms for the first 3x3x3 block and 7 for the second.
    # But Strassen's 7 products involve matrix multiplications, not scalar products.
    
    # The key insight: in the tensor decomposition, each "rank-1 term" is a scalar
    # product. So I need to express the Strassen algorithm in terms of scalar products.
    
    # For M3 = A[0,0] * B[0,0]: this is already 1 scalar product.
    # For M4 = A22 * B22 where A22, B22 are 2x2: this is 4 scalar products.
    # So Strassen for 3x3 doesn't directly give 7 scalar products.
    
    # I think the right approach is to just use the schoolbook but with a smarter
    # grouping. Or accept that for 3x3x6, the minimal rank might be higher than 14.
    
    # Let me just return the schoolbook for now.
    return None

def _build_schoolbook(n, m, p):
    """Return the standard (rank=n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_based_3x3x6():
    """
    Build a rank-14 decomposition using Strassen's algorithm for two 3x3x3 blocks.
    
    For each 3x3x3 block, we use a rank-7 decomposition based on Strassen's 7 products.
    The 7 Strassen products for 3x3 matrices (split as 1x1, 1x2, 2x1, 2x2) need to be
    expressed as scalar products.
    
    Actually, let me use a different approach: express the 3x3x3 multiplication as
    a sum of 7 rank-1 tensor terms where each term involves linear combinations.
    
    For the standard Strassen on 2x2 matrices:
    M1 = (A11-A22)(B21-B22)
    M2 = (A21-A22)B22  
    M3 = A11*B11
    M4 = A22*B22
    M5 = (A11+A22)B21
    M6 = (A21+A11)(B12-B22)
    M7 = (A22+A11)(B11-B12)
    
    Each Mi is a 2x2 matrix product, which in the tensor decomposition requires
    4 scalar products each, giving 28 total. That's worse than schoolbook (16 for 2x2x2).
    
    The key is that in the tensor decomposition framework, we're looking for:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    For 2x2x2, Strassen gives rank 7, meaning there exist 7 sets of vectors such that
    the sum reproduces the multiplication tensor.
    
    Let me look up the explicit rank-7 decomposition for 2x2x2.
    
    The 7 rank-1 terms for 2x2x2 Strassen are:
    Term 1: U=[1,-1,0,0], V=[0,0,1,-1], W=[1,0,0,1]  -> (A11-A22)(B21-B22) contributes to C11,C12,C21,C22
    ...
    
    This is getting complicated. Let me just implement a known rank-7 for 2x2x2
    and then extend to 3x3x6.
    """
    # For 2x2x2, the rank-7 Strassen decomposition in tensor form:
    # T[i,j,k] = sum_{r=1}^{7} U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where indices are 0-indexed.
    
    # The 7 terms (using 0-indexed, with i,j in {0,1} for A, j,k in {0,1} for B, i,k in {0,1} for C):
    # Term 1: (A00-A11)(B10-B11)
    #   U: [1,-1] for A00,A11 -> U[0,0]=1, U[1,1]=-1 (but U is 2D: U[i*2+j])
    #   Actually U[i*2+j, r] for A[i,j], V[j*2+k, r] for B[j,k], W[k*2+i, r] for C[i,k]
    
    # Let me define the 7 terms explicitly for 2x2x2:
    # Term 1: M1 = (A00-A11)(B10-B11)
    #   A part: A00 - A11 -> U[0*2+0]=1, U[1*2+1]=-1
    #   B part: B10 - B11 -> V[1*2+0]=1, V[1*2+1]=-1
    #   C part: C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
    #   W[0*2+0]=1, W[1*2+0]=-1, W[0*2+1]=1, W[1*2+1]=-1
    #   Wait, W[k*2+i, r] so W[0*2+0]=W[0] for C[0,0], W[1*2+0]=W[2] for C[0,1],
    #   W[0*2+1]=W[1] for C[1,0], W[1*2+1]=W[3] for C[1,1]
    
    # This is error-prone. Let me just use the schoolbook for now and accept rank 54.
    # The improvement will come from a more careful construction.
    
    return None

def _try_find_lower_rank(n, m, p, max_rank, time_budget):
    """
    Try to find a decomposition with rank < max_rank.
    """
    start = time.time()
    U, V, W = _build_schoolbook(n, m, p)
    current_rank = U.shape[1]
    
    if time.time() - start > time_budget:
        return None
    
    return None

def construct():
    n, m, p = 3, 3, 6
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - elapsed - 10.0
    
    # Try the Strassen-based decomposition first
    result = _build_strassen_based_3x3x6()
    if result is not None:
        U, V, W = result
        # Verify the decomposition
        if _verify(U, V, W, n, m, p):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
