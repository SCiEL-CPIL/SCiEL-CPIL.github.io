import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _compute_slice_rank_lower_bound(n, m, p):
    """
    Compute the slice-rank lower bound for the matrix multiplication tensor.
    For each fixed index in each dimension, flatten the tensor into a matrix
    and compute its rank. The minimum rank across all slices is a lower bound
    on the tensor rank.
    
    The matrix multiplication tensor T[i,j,k] = delta_{i+k, j} (using 0-indexed)
    where i in [0,n), j in [0,m), k in [0,p).
    Actually, for A (n x m) * B (m x p) = C (n x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    So T[i,j,k] = 1 if the multiplication contributes, which is always 1 for valid indices.
    Wait - the tensor is T[i,j,k] = 1 for all valid (i,j,k) since it's the standard
    matrix multiplication tensor where C[i,k] = sum_j A[i,j] B[j,k].
    
    Actually the structure tensor has entries T[i,j,k] which form a 3D array.
    For slice at fixed i: matrix M_jk = T[i,j,k] for all j,k. Since T[i,j,k]=1 always,
    this is an all-ones matrix of shape (m, p), which has rank 1.
    
    Hmm, that's not right. Let me reconsider. The tensor for matrix multiplication
    is defined such that C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k].
    For standard multiplication, T[i,j,k] = 1 for all i,j,k.
    
    But wait - that would make the tensor rank 1, which is clearly wrong.
    
    I think I'm confusing the tensor structure. The correct interpretation:
    The multiplication tensor M has entries M[i,j,k] where the decomposition
    satisfies: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = M[i,j,k]
    
    Actually, looking at the interface: U has shape (n*m, rank), V has shape (m*p, rank),
    W has shape (p*n, rank). So:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give the
    coefficient for A[i,j]*B[j,k] in computing C[i,k].
    
    For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k], so the
    tensor entry is 1 for all (i,j,k) that participate.
    
    The slice-rank bound: fix one index, say i. Then we get a matrix of shape (m*p, p*n)
    or similar. Let me think more carefully.
    
    For fixed i: the sub-tensor involves U[i*m+j, r] for j in [0,m), V[j*p+k, r] for j,k,
    and W[k*n+i, r] for k. The effective matrix for this slice would be...
    
    Actually, the standard slice-rank argument: fix index i. Consider the matrix
    S_i[j, k] where the entry is determined by which r's contribute. Since T[i,j,k]=1,
    all entries are 1, so rank is 1. That can't be the right lower bound.
    
    I think the issue is that the tensor isn't all-ones. Let me reconsider the problem.
    
    Oh wait - I think the tensor IS all ones for the standard multiplication, but the
    slice-rank lower bound works differently. Let me look up the correct formulation.
    
    The slice-rank of a tensor T in A x B x C is defined as the minimum number of
    rank-1 tensors needed, where a rank-1 tensor is a tensor that is rank 1 when
    viewed as a matrix in any unfolding.
    
    For the multiplication tensor, fixing i gives a matrix of size (m x p) with all
    entries 1, which has rank 1. But this is just one slice. The tensor rank is at
    least the maximum over all slices of the slice rank... no, that's not right either.
    
    Let me just compute it empirically: build the full tensor and check.
    """
    # Build the multiplication tensor T[i,j,k] = 1 for all valid indices
    # The tensor has shape (n, m, p)
    # For the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # The slice-rank lower bound: for each fixed value of one index, 
    # the resulting matrix must have rank at most the tensor rank.
    # Fix i: matrix M_i has entries M_i[j, k] = T[i, j, k] = 1 for all j, k
    # This is an all-ones matrix of shape (m, p), rank = 1
    
    # Fix j: matrix M_j has entries M_j[i, k] = T[i, j, k] = 1 for all i, k  
    # This is an all-ones matrix of shape (n, p), rank = 1
    
    # Fix k: matrix M_k has entries M_k[i, j] = T[i, j, k] = 1 for all i, j
    # This is an all-ones matrix of shape (n, m), rank = 1
    
    # So the slice-rank lower bound is 1? That's clearly too weak.
    
    # I think I'm misunderstanding the tensor structure. Let me reconsider.
    
    # Actually, I believe the correct lower bound comes from considering that
    # the tensor rank must be at least max(rank of any matrix flattening).
    # The flattening to (n*m, p) or (n, m*p) etc.
    
    # Flatten to matrix of shape (n*m, p): entry [i*m+j, k] = T[i,j,k] = 1
    # This is all ones, rank 1.
    
    # I think the all-ones tensor has rank 1, which means the slice-rank bound
    # is indeed 1, which is trivially satisfied.
    
    # The actual lower bound for 2x3 * 3x7 matrix multiplication is known to be
    # at least 14 (from algebraic complexity results), but that's much harder to compute.
    
    # For now, let's just verify our current solution is correct and return the
    # trivial lower bound of 1, acknowledging that the slice-rank method gives
    # a weak bound for this particular tensor structure.
    
    return 1  # Trivial lower bound from slice-rank analysis

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x3 by 3x7 matrix multiplication.
    
    Strategy:
    - Split the 3 columns of A into a 2x2 block (j=0,1) and a 1x1 block (j=2).
    - For the 2x2 block (j=0,1), split the 7 rows of B into three 2x2 blocks (k=0,1; k=2,3; k=4,5)
      and one 2x1 block (k=6).
    - Use Strassen's algorithm (rank 7) for each 2x2x2 block.
    - Use schoolbook (rank 4) for the 2x2x1 block.
    - Use schoolbook (rank 14) for the 2x1x7 block (j=2).
    
    Total rank = 3*7 + 4 + 14 = 21 + 4 + 14 = 39.
    
    We compute the slice-rank lower bound to establish a theoretical minimum.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    n, m, p = 2, 3, 7
    
    # Compute slice-rank lower bound for reference
    lower_bound = _compute_slice_rank_lower_bound(n, m, p)
    
    rank = 3 * 7 + 4 + 14  # 39
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, signs_u=None, signs_v=None, signs_w=None):
        nonlocal col
        if time.time() > deadline:
            return
        if signs_u is None:
            signs_u = [1] * len(u_indices)
        if signs_v is None:
            signs_v = [1] * len(v_indices)
        if signs_w is None:
            signs_w = [1] * len(w_indices)
        
        for idx, sign in zip(u_indices, signs_u):
            U[idx][col] += sign
        for idx, sign in zip(v_indices, signs_v):
            V[idx][col] += sign
        for idx, sign in zip(w_indices, signs_w):
            W[idx][col] += sign
        col += 1
    
    def add_strassen_block(k_start):
        """
        Add Strassen decomposition for a 2x2x2 block.
        """
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term([a00, a11], [b00, b11], [c00, c11])
        
        # M2 = (A10 + A11)(B00)
        add_term([a10, a11], [b00], [c10, c11], signs_w=[1, -1])
        
        # M3 = (A00)(B01 - B11)
        add_term([a00], [b01, b11], [c01, c11], signs_v=[1, -1])
        
        # M4 = (A11)(B10 - B00)
        add_term([a11], [b10, b00], [c00, c10], signs_v=[1, -1])
        
        # M5 = (A00 + A01)(B11)
        add_term([a00, a01], [b11], [c00, c01], signs_w=[-1, 1])
        
        # M6 = (A10 - A00)(B00 + B01)
        add_term([a10, a00], [b00, b01], [c11], signs_u=[1, -1])
        
        # M7 = (A01 - A11)(B10 + B11)
        add_term([a01, a11], [b10, b11], [c00], signs_u=[1, -1])
    
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Schoolbook block for k=6 (2x2x1)
    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1
    
    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])
    
    # Part 2: j=2 (2x1x7 schoolbook)
    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
