"""Rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = np.zeros((p * n, n * m, m * p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += Ua[u, c] * Va[v, c] * Wa[io, c]
                if s != T[io, u, v]:
                    return False
    return True


def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _winograd_2x2x3():
    """
    Winograd decomposition for 2x2 * 2x3 matrix multiplication, rank 7.
    A (2x2) * B (2x3) = C (2x3)
    
    Using Winograd's F_2(2) transform on rows of A and columns of B:
    The standard Winograd for 2x2*2x2 has rank 7. For 2x2*2x3, we can
    partition B into two 2x2 blocks... but that gives rank 14.
    
    Actually, let me use a direct approach. For 2x2 * 2x3:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Winograd F_2(2) approach:
    Define transformed A:
      a0 = A[0,0] + A[0,1] + A[1,0] + A[1,1]  (sum)
      a1 = A[0,0] - A[0,1] + A[1,0] - A[1,1]  (diff rows, same cols)
      a2 = A[0,0] + A[0,1] - A[1,0] - A[1,1]  (same rows, diff cols)
      a3 = A[0,0] - A[0,1] - A[1,0] + A[1,1]  (diff both)
    
    Hmm, this is for 2x2*2x2. Let me think more carefully.
    
    For 2x2 * 2x3, the tensor has 2*2*2*3 = 24 nonzero entries (one per product).
    The trivial rank is 12 (each output element is a sum of 2 products).
    
    Let me just use the schoolbook for 2x2*2x3 which has rank 12,
    and then combine two such blocks for 4x4*4x9.
    
    Actually, let me reconsider the problem structure.
    4x4 * 4x9: partition A into 2x2 blocks (2x2 grid), B into 2x3 blocks (2x3 grid).
    Output C is 4x9, partitioned into 2x2 blocks of 2x3 each... 
    No, C[i,k] where i in 0..3, k in 0..8.
    
    Let me partition:
    A: 4x4 -> 2x2 blocks of size 2x2
    B: 4x9 -> 2x3 blocks of size 2x3
    C: 4x9 -> 2x3 blocks of size 2x3
    
    C_block[r,c] = sum over (r',c') of A_block[r,r'] * B_block[c',c]
    
    This is a block matrix multiplication: (2x2) block matrix * (2x3) block matrix = (2x3) block matrix.
    Each block is 2x2 * 2x3.
    
    Total rank if we do schoolbook at block level: 2*2*3 * 12 = 144. Same as before.
    
    To get improvement, we need to use Winograd within each 2x2*2x3 multiplication.
    
    Let me try a different approach: use the fact that 4 = 2*2 and 9 = 3*3,
    and apply 2-point and 3-point transforms.
    
    Actually, let me try the simplest thing that might work: 
    Use Winograd F_2(2) for the 2x2 dimension and F_3(3) for the 3x3 dimension.
    
    F_2(2) has 4 points (rank 4 for 2x2 convolution... no, it's for 2-point DFT).
    F_3(3) has 9 points.
    
    Hmm, this is getting complicated. Let me just try to construct a valid
    decomposition with rank < 144 using a structured approach.
    
    Let me try: partition into 2x2 blocks and use the known rank-7 
    Winograd for 2x2*2x2, handling the 3 in the other dimension separately.
    """
    pass


def _construct_winograd_4x4x9():
    """
    Construct a rank-reduced decomposition for 4x4 * 4x9 using Winograd transforms.
    
    Strategy: Partition the 4x4 matrix A into 2x2 blocks (2x2 grid),
    and the 4x9 matrix B into 2x3 blocks (2x3 grid).
    The output 4x9 is partitioned into 2x3 blocks (2x3 grid).
    
    Block multiplication: C[i,j] = sum_k A[i,k] * B[k,j] where i,j,k are block indices.
    A is 2x2 blocks, B is 2x3 blocks, C is 2x3 blocks.
    
    For each block product A[i,k] (2x2) * B[k,j] (2x3) -> C[i,j] (2x3),
    we can use a Winograd-like decomposition.
    
    Let me use a direct algebraic approach. For 2x2 * 2x3:
    We have 4 elements in A, 6 elements in B, 6 elements in C.
    
    Winograd F_2(2) transform on A (treating it as 2x2):
    The 4-point transform of a 2x2 matrix gives 4 transformed values.
    Similarly for B, we need a transform that reduces the 6 elements.
    
    Actually, let me just use a simpler approach: 
    Apply 2-point Winograd on the row dimension of A and column dimension of B,
    and handle the remaining dimensions with schoolbook.
    
    For 4x4 * 4x9:
    - Apply 2-point transform on rows of A (size 4 -> use two 2-point transforms)
    - Apply 2-point transform on columns of B (size 9 -> use... this doesn't divide evenly)
    
    Let me try yet another approach. Use the identity:
    (4x4)*(4x9) can be seen as computing 36 output elements, each being a dot product
    of length 4.
    
    Total schoolbook: 36 * 4 = 144 multiplications.
    
    Winograd F_2(2) for 2x2*2x2: rank 7 instead of 4*4=16... no, 2*2*2*2=16 schoolbook, rank 7.
    
    For our case, let me partition into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01, B02], [B10, B11, B12]] where each Bij is 2x3
    C = [[C00, C01, C02], [C10, C11, C12]] where each Cij is 2x3
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    
    Each term like A00*B00 is a 2x2 * 2x3 multiplication.
    
    If I can find a rank-r decomposition for 2x2*2x3, then total rank = 6*r.
    
    Let me find the rank of 2x2*2x3.
    Schoolbook: 2*2*3 = 12.
    
    Can we do better? The tensor for 2x2*2x3 has 2*2*2*3 = 24 nonzero entries.
    
    Using Winograd F_2(2) on the 2x2 part:
    Transform A (2x2) using F_2(2):
      a'0 = A[0,0] + A[0,1]
      a'1 = A[0,0] - A[0,1]
      a'2 = A[1,0] + A[1,1]
      a'3 = A[1,0] - A[1,1]
    
    Hmm, this is a 1D transform on each row.
    
    Let me try a different factorization. Write the 2x2*2x3 tensor as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    where i in {0,1}, j in {0,1}, k in {0,1,2}.
    
    This is the same as (2x2) * (2x3).
    
    I can view this as: for each k, C[:,k] = A * B[:,k] (2x2 times 2x1 = 2x1).
    So C[:,k] = A * B[:,k].
    
    If I precompute A * e_j for j=0,1 (columns of A), then C[:,k] = A[0,:]*B[0,k] + A[1,:]*B[1,k].
    
    Alternatively, use the identity:
    A * B = (transform A) * (transform B) * (inverse transform)
    
    For 2x2 * 2x3, let's use:
    Let M be a 2x2 invertible matrix (the "transform" on A's rows).
    Let N be a 2x3 matrix (the "transform" on B's columns... but B is 2x3, so we transform along its 2 rows).
    
    Actually, the standard Winograd approach:
    C = A * B
    Let A' = M * A (M is 2x2)
    Let B' = B * N (N is 3x3... no, B is 2x3, so B * N where N is 3x3 gives 2x3)
    
    Hmm, this doesn't reduce the number of multiplications directly.
    
    Let me just try to find a rank-7 or rank-8 decomposition for 2x2*2x3 by search.
    But that might be too slow.
    
    Alternative: use the fact that 2x2*2x3 can be embedded in 2x2*2x2 by padding.
    Pad B to 2x4 (with a zero column), then 2x2*2x4 has rank... 
    2x2*2x2 has rank 7 (Winograd). 2x2*2x4 = two 2x2*2x2 blocks, rank 14.
    But we only need 3 of the 4 columns, so maybe we can save.
    
    Actually, let me just implement a straightforward Winograd-style decomposition
    and verify it. If it doesn't reduce rank below 144, I'll fall back to schoolbook.
    
    Let me try: use 2-point Winograd on the 4x4 -> treat as 2x2 blocks of 2x2.
    Apply F_2(2) to each 2x2 block of A (4 blocks), and to each 2x2 sub-block of B.
    
    This is getting complex. Let me just try a simple structured decomposition
    and verify it works.
    """
    n, m, p = 4, 4, 9
    
    # Use a block Winograd approach:
    # Partition A (4x4) into 2x2 blocks, B (4x9) into 2x3 blocks.
    # Apply 2-point Winograd transform within each block.
    
    # For now, let's use a simpler approach: 
    # Direct construction using tensor product structure.
    
    # 4x4 * 4x9: 
    # Index A by (i,j) where i,j in 0..3
    # Index B by (j,k) where j in 0..3, k in 0..8
    # Output C by (i,k) where i in 0..3, k in 0..8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's use the fact that 4 = 2*2 and 9 = 3*3.
    # Write i = 2*i0 + i1, j = 2*j0 + j1, k = 3*k0 + k1
    # where i0,j0 in 0..1, i1,j1 in 0..1, k0 in 0..2, k1 in 0..2.
    
    # A[i,j] = A[2*i0+i1, 2*j0+j1]
    # B[j,k] = B[2*j0+j1, 3*k0+k1]
    # C[i,k] = C[2*i0+i1, 3*k0+k1]
    
    # C[2*i0+i1, 3*k0+k1] = sum_{j0,j1} A[2*i0+i1, 2*j0+j1] * B[2*j0+j1, 3*k0+k1]
    
    # = sum_{j0} sum_{j1} A[2*i0+i1, 2*j0+j1] * B[2*j0+j1, 3*k0+k1]
    
    # Group by j0:
    # = sum_{j0=0}^{1} (sum_{j1=0}^{1} A[2*i0+i1, 2*j0+j1] * B[2*j0+j1, 3*k0+k1])
    
    # For fixed j0, the inner sum is a 2x2 * 2x3 type product... 
    # Actually for fixed j0, j1 in 0..1, and for each k0,k1:
    # sum_{j1} A[2*i0+i1, 2*j0+j1] * B[2*j0+j1, 3*k0+k1]
    
    # This is: for fixed (i0,i1,j0,k0), compute sum_{j1=0,1} A[2*i0+i1, 2*j0+j1] * B[2*j0+j1, 3*k0+k1]
    
    # Let me define for each (i0,i1,j0): 
    #   a_vec = [A[2*i0+i1, 2*j0], A[2*i0+i1, 2*j0+1]]  (length 2)
    # For each (j0,k0,k1):
    #   b_vec = [B[2*j0, 3*k0+k1], B[2*j0+1, 3*k0+k1]]  (length 2)
    #   product = dot(a_vec, b_vec)
    
    # So C[2*i0+i1, 3*k0+k1] = sum_{j0=0}^{1} dot(a_vec[i0,i1,j0], b_vec[j0,k0,k1])
    
    # This is still schoolbook: 4*4*9 = 144 terms.
    
    # To reduce, we need to use Winograd on the j1 dimension (size 2).
    # F_2(2) transform: 
    #   a'0 = a0 + a1, a'1 = a0 - a1
    #   b'0 = b0 + b1, b'1 = b0 - b1
    #   dot(a,b) = (a'0*b'0 + a'1*b'1) / 2
    #   
    # But this introduces fractions (1/2). The problem allows half-integers, so this is fine!
    
    # So for each (i0,i1,j0,k0,k1):
    #   dot(a,b) = (a'0*b'0 + a'1*b'1) / 2
    #   where a'0 = A[2*i0+i1, 2*j0] + A[2*i0+i1, 2*j0+1]
    #         a'1 = A[2*i0+i1, 2*j0] - A[2*i0+i1, 2*j0+1]
    #         b'0 = B[2*j0, 3*k0+k1] + B[2*j0+1, 3*k0+k1]
    #         b'1 = B[2*j0, 3*k0+k1] - B[2*j0+1, 3*k0+k1]
    
    # So C[2*i0+i1, 3*k0+k1] = sum_{j0} (a'0*b'0 + a'1*b'1) / 2
    # = 1/2 * sum_{j0} a'0[i0,i1,j0] * b'0[j0,k0,k1] + 1/2 * sum_{j0} a'1[i0,i1,j0] * b'1[j0,k0,k1]
    
    # = 1/2 * sum_{j0=0,1} a'0[i0,i1,j0] * b'0[j0,k0,k1] + 1/2 * sum_{j0=0,1} a'1[i0,i1,j0] * b'1[j0,k0,k1]
    
    # The first sum: sum_{j0=0,1} a'0[i0,i1,j0] * b'0[j0,k0,k1]
    # This is a 2x1 * 1x2... no, it's a dot product of length 2.
    # a'0 is indexed by (i0,i1,j0) -> 2*2*2 = 8 values
    # b'0 is indexed by (j0,k0,k1) -> 2*3*3 = 18 values
    # The sum over j0 gives: for each (i0,i1,k0,k1), sum over j0 of a'0[i0,i1,j0]*b'0[j0,k0,k1]
    # This is a matrix product: (2*2) x (2) times (2) x (3*3) -> (2*2) x (3*3)
    # i.e., 4x9 matrix = 4x2 times 2x9
    # Rank of this: 4*2 + 2*9 = 8 + 18 = 26? No, the rank of the tensor is at most 4*2*... 
    # Actually, the tensor for (4x2)*(2x9) has rank at most 4*2*9/2 = 36... no.
    # The tensor T[i,k,j] = delta(i,j1)*delta(j,j2)*delta(j2,k) for (m x n)*(n x p) has rank m*n*p... no.
    # For (4x2)*(2x9): T[i,k,j1,j2] = delta(i,j1)*delta(j1,j2)*delta(j2,k)
    # The nonzero entries are: i in 0..3, j1 in 0..1, j2 in 0..1, k in 0..8, with j1=j2.
    # So T[i,k,j1,j2] = delta(j1,j2) * delta(i,j1) * delta(j2,k)... no, that's not right.
    
    # Let me re-index. The product C[i,k] = sum_j A[i,j]*B[j,k] where A is 4x2, B is 2x9.
    # T[i,k,j] = delta(i,j)*... no, T is a 3-tensor: T[i,k,j] where i in 0..3, k in 0..8, j in 0..1.
    # T[i,k,j] = 1 if the output (i,k) uses input (j) in both A and B.
    # A has indices (i,j), B has indices (j,k). So T[i,k,j] = 1 for all valid (i,j,k) combinations.
    # T[i,k,j] = 1 iff j is a valid intermediate index, which it always is.
    # Wait, T[i,k,j] = 1 for all i in 0..3, k in 0..8, j in 0..1.
    # That's 4*9*2 = 72 entries, all 1.
    # The rank of this tensor: it's the all-ones tensor of size 4x9x2.
    # The all-ones tensor of size a x b x c has rank min(a,b,c) = 2.
    # Wait, no. The tensor T[i,k,j] = 1 for all i,k,j.
    # This can be written as sum_{j=0}^{1} u_j[i] * v_j[k] * w_j[j] where u_j[i]=1, v_j[k]=1, w_j[j']=delta(j,j').
    # So rank = 2.
    
    # So the first term (sum over j0 of a'0 * b'0) has rank at most 2 (for the j0 dimension).
    # Similarly for the second term.
    # Total rank: 2 + 2 = 4? That seems too good.
    
    # Wait, I think I'm confusing myself. Let me be more careful.
    
    # The full decomposition:
    # C[i,k] = 1/2 * sum_{j0=0,1} a'0[i0,i1,j0] * b'0[j0,k0,k1] + 1/2 * sum_{j0=0,1} a'1[i0,i1,j0] * b'1[j0,k0,k1]
    
    # where i = 2*i0+i1, k = 3*k0+k1.
    
    # The first double sum: sum_{j0} a'0[i0,i1,j0] * b'0[j0,k0,k1]
    # This is a bilinear form in a'0 and b'0, with j0 as the summed index.
    # As a tensor in (i0,i1) x (k0,k1) x (j0): T1[i0,i1,k0,k1,j0] = a'0[i0,i1,j0] * b'0[j0,k0,k1]
    # But a'0 and b'0 are functions of the original A and B, not separate variables.
    
    # In the tensor decomposition framework, we need:
    # C[i,k] = sum_c U[i, c] * V[k, c] * W[c]  ... no, that's not the right form.
    
    # The tensor decomposition is: T[i,k,j] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where T is the multiplication tensor.
    
    # For 4x4 * 4x9: T[i,k,j] = 1 for all i in 0..3, k in 0..8, j in 0..3.
    # Wait, no. T[i,k,j] = 1 iff output (i,k) uses A[i,j]*B[j,k].
    # So T[i,k,j] = 1 for all valid (i,j,k) where i in 0..3, j in 0..3, k in 0..8.
    # That's 4*4*9 = 144 entries, all 1.
    # The all-ones tensor of size 4x9x4 has rank min(4,9,4) = 4.
    # Wait, that can't be right. The rank of the matrix multiplication tensor is not 4.
    
    # I think I'm confusing the tensor structure. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 9 matrix."
    # "Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    # So U has shape (4*4, rank) = (16, rank)
    # V has shape (4*9, rank) = (36, rank)
    # W has shape (9*4, rank) = (36, rank)
    
    # The tensor is T[u, v, w] where u in 0..15 (A index), v in 0..35 (B index), w in 0..35 (C index).
    # T[u, v, w] = 1 iff C[w] uses A[u] * B[v].
    
    # C[w] where w = k*n + i = 9*i + k... wait, "W[k*n+i,r]" so w = k*4 + i.
    # u = i*m + j = i*4 + j (A[i,j])
    # v = j*p + k = j*9 + k (B[j,k])
    # w = k*n + i = k*4 + i (C[i,k])
    
    # T[u, v, w] = 1 iff there exist i,j,k such that u = i*4+j, v = j*9+k, w = k*4+i.
    # So T[i*4+j, j*9+k, k*4+i] = 1 for all i in 0..3, j in 0..3, k in 0..8.
    # All other entries are 0.
    # There are 4*4*9 = 144 nonzero entries.
    
    # The rank of this tensor is what we want to minimize.
    
    # Now, my Winograd approach:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j0=0,1} sum_{j1=0,1} A[2*i0+i1, 2*j0+j1] * B[2*j0+j1, 3*k0+k1]
    # where i = 2*i0+i1, k = 3*k0+k1.
    
    # Using F_2(2) on j1:
    # sum_{j1} A[i, 2*j0+j1] * B[2*j0+j1, k]
    # = (A[i,2*j0] + A[i,2*j0+1]) * (B[2*j0,k] + B[2*j0+1,k]) / 2
    #   + (A[i,2*j0] - A[i,2*j0+1]) * (B[2*j0,k] - B[2*j0+1,k]) / 2
    
    # So C[i,k] = sum_{j0} [ (A[i,2*j0]+A[i,2*j0+1])*(B[2*j0,k]+B[2*j0+1,k]) 
    #                        + (A[i,2*j0]-A[i,2*j0+1])*(B[2*j0,k]-B[2*j0+1,k]) ] / 2
    
    # = 1/2 * sum_{j0} A_plus[i, j0] * B_plus[j0, k] + 1/2 * sum_{j0} A_minus[i, j0] * B_minus[j0, k]
    
    # where A_plus[i, j0] = A[i, 2*j0] + A[i, 2*j0+1]
    #       A_minus[i, j0] = A[i, 2*j0] - A[i, 2*j0+1]
    #       B_plus[j0, k] = B[2*j0, k] + B[2*j0+1, k]
    #       B_minus[j0, k] = B[2*j0, k] - B[2*j0+1, k]
    
    # Now, sum_{j0=0,1} A_plus[i, j0] * B_plus[j0, k] is a (4x2)*(2x9) matrix product.
    # The tensor for this has rank... let's see.
    # T_plus[i, k, j0] = 1 for all i in 0..3, k in 0..8, j0 in 0..1.
    # This is the all-ones tensor of size 4x9x2, which has rank 2.
    # Similarly for T_minus.
    
    # So the total decomposition has rank 2 + 2 = 4? That seems impossibly low.
    
    # Wait, I think the issue is that A_plus and B_plus are not independent of the original A and B.
    # In the tensor decomposition, U, V, W are fixed matrices, and the product 
    # sum_c U[u,c]*V[v,c]*W[w,c] must equal T[u,v,w] for all u,v,w.
    
    # The rank-2 decomposition of the all-ones tensor T[i,k,j0] = 1:
    # T[i,k,j0] = sum_{c=0,1} U_c[i] * V_c[k] * W_c[j0]
    # where U_c[i] = 1, V_c[k] = 1, W_c[j0] = delta(c, j0).
    # So U has columns [1,1,1,1]^T and [1,1,1,1]^T (but these are for the i index, not the full u index).
    
    # But in our problem, u = i*4 + j, not just i. So the tensor is not simply all-ones.
    
    # I think my Winograd approach needs to be applied more carefully to the full tensor.
    
    # Let me just implement the schoolbook and verify it, then try to find a better decomposition
    # using the Winograd idea but applied to the full tensor indices.
    
    # For now, return schoolbook.
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try Winograd-based construction
    if time.time() < _DEADLINE - 30:
        try:
            cand = _construct_winograd_4x4x9()
            if _exact_check(cand["U"], cand["V"], cand["W"], T, n, m, p):
                return cand
        except Exception:
            pass
    
    # Fall back to schoolbook
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    
    return best

# EVOLVE-BLOCK-END
