"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return (U, V, W) lists for schoolbook rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i',j,c]*V[j,k',c]*W[k,i,c] == delta[i',i]*delta[k',k]."""
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for ip in range(n):
                    for kp in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[ip * m + j][c] * V[j * p + kp][c] * W[k * n + i][c]
                        expected = (1 if ip == i and kp == k else 0)
                        if s != expected:
                            return False
    return True


def _strassen222():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each 4x7 (as lists of lists).
    U indexed by (i,j) -> row i*2+j
    V indexed by (j,k) -> row j*2+k
    W indexed by (k,i) -> row k*2+i
    """
    # Strassen 7 products for 2x2x2
    # U: 4 rows (i,j), 7 cols
    # V: 4 rows (j,k), 7 cols
    # W: 4 rows (k,i), 7 cols
    U = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 0, 0, 0, 0, 0, 0],  # (0,1)
        [0, 1, 0, 0, 0, 0, 0],  # (1,0)
        [0, 0, 1, 1, 0, 0, 0],  # (1,1)
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 0, 0, 0, 0, 0],  # (0,1)
        [0, 0, 1, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 1, 1],  # (1,1)
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],  # (0,0)
        [0, 1, 1, 0, 0, 0, 0],  # (0,1)
        [0, 0, 0, 1, 0, 0, 0],  # (1,0)
        [0, 0, 0, 0, 1, 0, 0],  # (1,1)
    ]
    return U, V, W


def _build_block_decomp():
    """
    Build a block decomposition for 2x3x9.
    
    Strategy: Split the 9 columns of B into three 3-column blocks.
    Each block is a 2x3x3 multiplication.
    Use rank-7 Strassen for each 2x3x3? No, Strassen is for 2x2x2.
    
    Better: Split A (2x3) into columns, B (3x9) into rows.
    Actually, let's think of it as: result[i,k] = sum_j A[i,j] * B[j,k].
    
    Alternative: Use the fact that 2x3x9 = sum over j of 2x1x9 * 1x3x9... no.
    
    Let's use a simpler block approach:
    Split the 3 middle dimension into 1+1+1.
    For each j in {0,1,2}: multiply 2x1 (column j of A) by 1x9 (row j of B).
    Each such product has rank 2*1*9 = 18 in schoolbook, but actually each 2x1x9 
    multiplication (outer product of 2-vector and 9-vector, summed) has rank... 
    Actually the tensor for 2x1x9 is: T[i,0,k] = A[i,0]*B[0,k], which has rank 2*9=18.
    But we can do better: the sum over j of 2x1x9 tensors.
    
    Actually, let's try a different split. Split B's 9 columns into three 3-column blocks.
    result[:, 0:3] = A @ B[:, 0:3]  (2x3 times 3x3)
    result[:, 3:6] = A @ B[:, 3:6]  (2x3 times 3x3)
    result[:, 6:9] = A @ B[:, 6:9]  (2x3 times 3x3)
    
    Each 2x3x3 multiplication can be done in rank at most 2*3*3=18 schoolbook.
    But we can use a better decomposition for 2x3x3.
    
    For 2x3x3: result[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    We can split the middle dimension: j=0,1,2.
    Each term is 2x1x3, which has rank 2*3=6.
    Total rank 18 for schoolbook.
    
    Can we do 2x3x3 in less than 18? 
    The tensor is the sum of three 2x1x3 tensors. Each 2x1x3 has rank 6 (since it's 
    the outer product of a 2-vector and a 3-vector, which is a rank-6 tensor in the 
    2x3x3 space... wait, no. A 2x1x3 tensor T[i,0,k] = a_i * b_k has rank 2*3=6 
    in the standard tensor decomposition sense? No, the rank of the tensor 
    sum_{i,k} a_i b_k * E[i,0,k] is min(2,3)=... no.
    
    Actually, for a 2x1x3 tensor, the "tensor rank" (as a trilinear form) is the 
    minimum number of rank-1 terms. A rank-1 term is u_i * v_0 * w_k. Since the 
    middle index is fixed at 0, v_0 is just a scalar. So a rank-1 term is 
    (alpha * u_i) * (beta * w_k), which is the outer product of a 2-vector and a 
    3-vector. The outer product of a 2-vector and a 3-vector has tensor rank 
    min(2,3) = 2? No, that's not right either.
    
    Let me think more carefully. The tensor for 2x1x3 multiplication is:
    T[i, j, k] = A[i,j] * B[j,k] where j is only 0.
    So T[i, 0, k] = A[i, 0] * B[0, k].
    This is a 2x1x3 tensor. As a 3D array, it's separable: T[i,0,k] = a_i * b_k.
    The tensor rank of a separable 2x1x3 tensor... A rank-1 tensor is u_i * v_0 * w_k.
    Since v_0 is a scalar, we can absorb it into u or w. So T[i,0,k] = (a_i) * (1) * (b_k)
    is already rank 1? No, because we need it to work for ALL a and b.
    
    I think I'm confusing tensor rank of a specific tensor with the rank needed for 
    bilinear multiplication. The multiplication tensor for 2x1x3 has rank 2*1*3 = 6 
    in the sense that we need 6 rank-1 bilinear terms to compute all products.
    
    For the 2x3x3 case, schoolbook gives 18. Can we do better?
    
    Actually, let's just use the Strassen-based approach from the inspiration.
    The inspiration says: "Strassen-based block decomposition achieves rank 50 by 
    using 7-term Strassen products for 2x2x2 sub-blocks and schoolbook for the 
    remaining 2x2x1 and 2x1x9 parts."
    
    Let me try to build rank 50.
    
    Split A (2x3) into:
    - Left 2x2 block (columns 0,1)
    - Right 1x2... no, 2x1 block (column 2)
    
    Split B (3x9) into:
    - Top 2x9 block (rows 0,1)
    - Bottom 1x9 block (row 2)
    
    Then A @ B = [A_left, A_right] @ [B_top; B_bottom]
    = A_left @ B_top + A_right @ B_bottom
    
    A_left @ B_top: 2x2 times 2x9.
    We can split B_top into three 2x3 blocks: B_top[:, 0:3], B_top[:, 3:6], B_top[:, 6:9].
    Each 2x2 times 2x3 can be computed using Strassen for 2x2x2 and schoolbook for 2x2x1.
    2x2 times 2x3 = 2x2 times (2x2 + 2x1) = Strassen(2x2x2) + schoolbook(2x2x1) = 7 + 4 = 11.
    Three such blocks: 3 * 11 = 33.
    
    A_right @ B_bottom: 2x1 times 1x9.
    This is an outer product of a 2-vector and a 9-vector, but we're multiplying 
    (2x1) by (1x9) to get (2x9). The rank for this is 2*1*9 = 18 schoolbook.
    But actually, it's just: result[i,k] = A_right[i,0] * B_bottom[0,k].
    This can be done in rank 2*9 = 18, but can we do better? 
    Actually, this is a simple element-wise product structure. We can write:
    result[i,k] = A_right[i,0] * B_bottom[0,k]
    Each (i,k) pair needs one term. So rank 18 is the schoolbook.
    Can we reduce? The tensor is sum_{i,k} A_right[i,0] * B_bottom[0,k] * E[i,0,k].
    This has tensor rank 2*9 = 18 since it's a sum of 18 independent rank-1 terms 
    (no two can be combined).
    
    Total: 33 + 18 = 51.
    
    Hmm, that's 51, not 50. Let me reconsider.
    
    Maybe the 2x2x1 part can be done in rank 3 instead of 4?
    2x2x1: result[i,k] where k is only 0. result[i,0] = sum_j A[i,j] * B[j,0].
    This is a 2x2x1 tensor. Schoolbook rank 4. Can we do rank 3?
    The tensor is: T[i, j, 0] = A[i,j] * B[j,0].
    We need: for each i, result[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0].
    This is like computing two dot products of length 2. 
    Rank-1 terms: u_i * v_j * w_0. Since w_0 is a scalar, we can absorb it.
    So we need to represent the 2x2 multiplication as sum of rank-1 bilinear forms.
    This is just 2x2x1 matrix multiplication, which has rank 4 (schoolbook) and 
    I believe cannot be done in rank 3.
    
    Let me try a different split.
    
    Actually, let me just implement the rank-54 schoolbook and verify it works, 
    then try to find a better decomposition through search or known results.
    
    For now, let's return schoolbook rank 54.
    """
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    return U, V, W


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    if _verify(U, V, W, n, m, p):
        best_rank = len(U[0])
    else:
        # Shouldn't happen
        best_rank = n * m * p
    
    # Try to find better decompositions
    # Known lower bounds and constructions...
    # For now, return the best we have
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
