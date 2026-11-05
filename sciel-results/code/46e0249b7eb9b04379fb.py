"""Matrix multiplication 4x4x6 - structural decomposition using Strassen blocks."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 4x4 matrix by a 4x6 matrix.
    
    Strategy: Decompose the 4x4x6 multiplication into three independent 4x4x2
    subproblems. Each 4x4x2 can be computed using 4 independent 2x2x2 products
    (each with rank 7 via Strassen), giving total rank 3 * 4 * 7 = 84.
    
    The key insight: A 4x4 matrix A times a 4x6 matrix B gives a 4x6 result C.
    We can split B into three 4x2 blocks: B = [B1 | B2 | B3].
    Then C = [A*B1 | A*B2 | A*B3], and each A*Bj is 4x2.
    
    For a 4x4 * 4x2 product: split A into 2x2 blocks and B_j into 2x2 blocks.
    A = [[A11, A12], [A21, A22]], B_j = [[B1j, B2j]] (2x2 blocks, but B_j is 4x2
    so it's [[B1j], [B2j]] where each B1j, B2j is 2x2).
    
    Wait, let me reconsider. A is 4x4, B_j is 4x2.
    A = [[A11, A12], [A21, A22]] where each is 2x2.
    B_j = [[C1j], [C2j]] where C1j and C2j are 2x2.
    
    Then A * B_j = [[A11*C1j + A12*C2j], [A21*C1j + A22*C2j]]
    
    This is 4 2x2*2x2 products. Each can be done with Strassen (rank 7).
    So 4x4x2 has rank 4*7 = 28.
    Three such subproblems: 3 * 28 = 84.
    
    The output convention is W[k*n+i, r] where the result is C[i,k] = sum_r U[i*m+j... 
    Wait, let me re-read the convention.
    
    The problem says: "The output factor follows the official trace convention W[k*n+i,r]."
    So W has shape (p*n, rank) = (6*4, rank) = (24, rank).
    W index: k*n + i where k is the column index of C (0..5) and i is the row index (0..3).
    So C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right either. Let me think about the tensor decomposition.
    
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] = 1 if this is the multiplication tensor.
    A rank-1 decomposition: T = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    But the convention here is W[k*n+i, r], so W is indexed by (k, i) -> k*n+i.
    This means W[k*n+i, r] corresponds to the coefficient for C[i,k].
    
    So: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Hmm, but U has shape (n*m, rank) and V has shape (m*p, rank). The index for U is i*m+j
    and for V is j*p+k. But j is summed over... this doesn't look like a standard decomposition.
    
    Actually, I think the decomposition is:
    C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    No wait, that would make it depend on j in both U and V which is unusual.
    
    Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    I think the correct interpretation is:
    - U[i*m+j, r] is the coefficient for A[i,j] in the r-th term
    - V[j*p+k, r] is the coefficient for B[j,k] in the r-th term  
    - W[k*n+i, r] is the coefficient for C[i,k] in the r-th term
    
    And the decomposition satisfies:
    sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i, r] = delta_{j,j'} * delta_{k,k'} * (something)
    
    Actually no. The standard tensor decomposition for matrix multiplication is:
    T_{ijk} = sum_r U_{i,r} * V_{j,r} * W_{k,r}
    where T_{ijk} = 1 represents the multiplication tensor.
    
    But here the shapes are (n*m, rank), (m*p, rank), (p*n, rank). This suggests:
    - First factor: indexed by (i, j) with i in [0,n), j in [0,m) -> index i*m+j
    - Second factor: indexed by (j, k) with j in [0,m), k in [0,p) -> index j*p+k  
    - Third factor: indexed by (k, i) with k in [0,p), i in [0,n) -> index k*n+i
    
    And the contraction is: for each r, we have a "term" that contributes to C.
    The actual matrix multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form, the multiplication tensor M has entries M[i,j,k] = 1 for all valid i,j,k,
    and the multiplication is: C[i,k] = sum_{j} M[i,j,k] * A[i,j] * B[j,k]
    
    Wait, that's not right either. M[i,j,k] should be 1 only when the indices match properly.
    
    Actually, I think the tensor is defined as: the bilinear map (A, B) -> C where
    C[i,k] = sum_j A[i,j] B[j,k].
    
    This is a trilinear map from (R^{n*m} x R^{m*p}) -> R^{n*p}.
    
    The rank-1 decomposition: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_{j'} B[j',k] V[j'*p+k, r]) * W[k*n+i, r]
    
    Hmm, that doesn't look right either because the sums over j and j' are independent.
    
    Let me think about this differently. In the schoolbook decomposition:
    For each (i, j, k), we have one term:
    U[i*m+j, col] = 1
    V[j*p+k, col] = 1
    W[k*n+i, col] = 1
    
    And C[i,k] = sum_{i',j',k'} sum_col U[i'*m+j', col] * V[j'*p+k', col] * W[k'*n+i', col] * A[i',j'] * B[j',k']
    
    Wait, that's not how it works. Let me look at the schoolbook more carefully.
    
    In the schoolbook, for each triple (i,j,k), we set:
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    The reconstruction would be:
    C_reconstructed[i,k] = sum_{i',j',k'} sum_col U[i'*m+j', col] * V[j'*p+k', col] * W[k'*n+i', col] * A[i',j'] * B[j',k']
    
    For a given col corresponding to (i0,j0,k0):
    U[i'*m+j', col] = 1 only if i'=i0 and j'=j0
    V[j'*p+k', col] = 1 only if j'=j0 and k'=k0
    W[k'*n+i', col] = 1 only if k'=k0 and i'=i0
    
    So the contribution is A[i0,j0] * B[j0,k0] * 1 * 1 * 1, and it goes to C[i0,k0].
    
    So C[i,k] = sum_{i0,j0,k0: i0=i, k0=k} A[i0,j0] * B[j0,k0] = sum_{j0} A[i,j0] * B[j0,k]
    
    Yes! That's correct. So the decomposition works.
    
    Now for Strassen on 2x2:
    Standard Strassen for 2x2 * 2x2 = 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B21)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B21)
    M7 = (A12-A22)(B12+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mr is a product of linear combinations of A entries and B entries.
    
    For the 4x4 * 4x2 case:
    A is 4x4, B is 4x2.
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into 2x2 blocks: B = [[B1], [B2]] where B1, B2 are 2x2.
    
    C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    This requires 4 independent 2x2*2x2 multiplications.
    Each can use Strassen with 7 products.
    Total: 4 * 7 = 28 products for 4x4*4x2.
    
    For 4x4*4x6, we split B into three 4x2 blocks: B = [B1, B2, B3]
    C = [A*B1, A*B2, A*B3]
    Total: 3 * 28 = 84 products.
    
    Now I need to construct the U, V, W matrices for this decomposition.
    
    Let me define the Strassen decomposition for a single 2x2*2x2 product.
    For A2x2 = [[a00, a01], [a10, a11]] and B2x2 = [[b00, b01], [b10, b11]]:
    
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b10-b11)
    M4 = a11(b01-b10)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mr corresponds to one rank-1 term in the decomposition.
    
    For M1: U coefficients for a00, a11 are 1,1; V coefficients for b00, b11 are 1,1; W coefficients for C00, C11 are 1,1.
    For M2: U for a10, a11 are 1,1; V for b00 is 1; W for C10 is 1.
    etc.
    
    Now I need to map these to the global indices.
    
    For the 4x4*4x2 subproblem (say for B block j, j=0,1,2):
    A is 4x4 with indices (i, j_a) where i, j_a in [0,4).
    B_j is 4x2 with indices (j_b, k) where j_b in [0,4), k in [0,2).
    C_j is 4x2 with indices (i, k) where i in [0,4), k in [0,2).
    
    Global indices:
    U: i*m + j_a where m=4, so U index = i*4 + j_a
    V: j_b*p + k where p=6, so V index = j_b*6 + k. But k is local to the block (0 or 1), global k = j*2 + k_local where j is the block index (0,1,2).
    W: k_global*n + i where n=4, so W index = k_global*4 + i.
    
    For block j (j=0,1,2), the local B columns are k_local=0,1, which map to global k = j*2 + k_local.
    
    Now, for the 2x2*2x2 subproblems within 4x4*4x2:
    A blocks: A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    B block: B1 = B[0:2, :], B2 = B[2:4, :] (each 2x2)
    
    The 4 products are:
    P1 = A11 * B1 -> contributes to C[0:2, :]
    P2 = A12 * B2 -> contributes to C[0:2, :]
    P3 = A21 * B1 -> contributes to C[2:4, :]
    P4 = A22 * B2 -> contributes to C[2:4, :]
    
    C[0:2, :] = P1 + P2
    C[2:4, :] = P3 + P4
    
    For each P, we use Strassen with 7 terms.
    
    Let me code this up.
    
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    num_blocks = 3  # three 4x2 blocks
    rank = 84  # 3 blocks * 4 subproducts * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 84)
    V = [[0] * rank for _ in range(m * p)]  # (24, 84)
    W = [[0] * rank for _ in range(p * n)]  # (24, 84)
    
    # Strassen coefficients for 2x2*2x2
    # Each term r (0-6) has:
    #   U_coeffs: dict from local A index (0-3) to coefficient
    #   V_coeffs: dict from local B index (0-3) to coefficient
    #   W_coeffs: dict from local C index (0-3) to coefficient
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # M1 = (a00+a11)(b00+b11)
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        # M2 = (a10+a11)b00
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1}},
        # M3 = a00(b10-b11)
        {'U': {0: 1}, 'V': {2: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        # M4 = a11(b01-b10)
        {'U': {3: 1}, 'V': {1: 1, 2: -1}, 'W': {0: 1, 2: 1}},
        # M5 = (a00+a01)b11
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        # M6 = (a10-a00)(b00+b10)
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 2: 1}, 'W': {3: 1}},
        # M7 = (a01-a11)(b01+b11)
        {'U': {1: 1, 3: -1}, 'V': {1: 1, 3: 1}, 'W': {0: 1}},
    ]
    
    # For each block j (0,1,2) of B (each 4x2):
    for block in range(num_blocks):
        # For each of the 4 subproducts (A block * B block):
        # Subproduct 0: A11 * B1 -> C rows 0-1
        # Subproduct 1: A12 * B2 -> C rows 0-1
        # Subproduct 2: A21 * B1 -> C rows 2-3
        # Subproduct 3: A22 * B2 -> C rows 2-3
        
        subproducts = [
            # (A_row_block, A_col_block, B_row_block, C_row_block)
            (0, 0, 0, 0),  # A11 * B1
            (0, 1, 1, 0),  # A12 * B2
            (1, 0, 0, 2),  # A21 * B1
            (1, 1, 1, 2),  # A22 * B2
        ]
        
        for sp_idx, (a_rb, a_cb, b_rb, c_rb) in enumerate(subproducts):
            # For each Strassen term:
            for s_idx, term in enumerate(strassen_terms):
                col = block * 28 + sp_idx * 7 + s_idx
                
                # Add U coefficients
                for local_a_idx, coeff in term['U'].items():
                    # local_a_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) within the 2x2 block
                    lr, lc = divmod(local_a_idx, 2)
                    global_i = a_rb * 2 + lr
                    global_j = a_cb * 2 + lc
                    u_idx = global_i * m + global_j
                    U[u_idx][col] += coeff
                
                # Add V coefficients
                for local_b_idx, coeff in term['V'].items():
                    lr, lc = divmod(local_b_idx, 2)
                    global_j = b_rb * 2 + lr
                    global_k = block * 2 + lc
                    v_idx = global_j * p + global_k
                    V[v_idx][col] += coeff
                
                # Add W coefficients
                for local_c_idx, coeff in term['W'].items():
                    lr, lc = divmod(local_c_idx, 2)
                    global_i = c_rb + lr
                    global_k = block * 2 + lc
                    w_idx = global_k * n + global_i
                    W[w_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
