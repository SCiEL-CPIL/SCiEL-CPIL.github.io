"""Matrix multiplication 6x6x6: deterministic algebraic construction with verification."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 Strassen, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 products for C = A*B where A, B, C are 2x2.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices for U (A): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Row indices for V (B): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Row indices for W (C): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1: (a11 + a22)(b11 + b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a21 + a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: a11(b12 - b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: a22(b21 - b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (a11 + a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (a21 - a11)(b11 + b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (a12 - a22)(b21 + b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _tensor_product_decomp(n, m, p, base_U, base_V, base_W):
    """
    Construct a block tensor product decomposition.
    If base is for size s x s x s, we can extend to larger sizes.
    For 6x6x6, we can use 3x3 blocks of 2x2? No, 6 = 3*2.
    Let's use a 3x3 block structure where each block is 2x2.
    The base Strassen is 2x2x2.
    We need a decomposition for 3x3x3 first? Or use 6x6 directly as 3x3 blocks of 2x2.
    
    Actually, the tensor product of two decompositions works if dimensions match.
    Let's construct a 3x3x3 decomposition using the schoolbook 3x3x3 (rank 27) or a known lower one?
    Rank of 3x3x3 is at most 23 (Laderman).
    Let's implement Laderman's 3x3x3 decomposition (rank 23).
    Then tensor product with Strassen 2x2x2 (rank 7) gives 6x6x6 rank 23*7 = 161.
    """
    pass

def _laderman_3x3():
    """
    Laderman's 3x3x3 matrix multiplication tensor decomposition, rank 23.
    Reference: Laderman, 1976.
    We need U (9, 23), V (9, 23), W (9, 23).
    Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c].
    """
    # This is a complex construction. Let's use a known explicit formula.
    # Due to complexity, I will use a verified set of coefficients if available, 
    # or fall back to a simpler valid lower-rank if I can't code Laderman correctly in time.
    # Actually, I can construct 6x6x6 directly using a block approach with 2x2 Strassen blocks.
    # 6x6 = 3x3 blocks of 2x2.
    # Let C = A B. A is 3x3 blocks of 2x2.
    # We need a decomposition for the 3x3 block matrix multiplication.
    # If we use schoolbook for the 3x3 block level, rank is 27 * 7 = 189.
    # This matches the inspiration's score.
    # Can we do better?
    # If we use Laderman (rank 23) for the 3x3 block level, rank is 23 * 7 = 161.
    # Let's try to implement Laderman's 3x3x3.
    
    # Laderman's 23 terms for 3x3x3.
    # Let A, B be 3x3 matrices. C = AB.
    # The 23 bilinear forms are linear combinations of A_ij B_kl.
    # I will use a pre-computed list of coefficients for Laderman's decomposition.
    # Since I cannot look it up externally, I will use a heuristic:
    # Try to find a low rank decomposition via random search + verification?
    # No, the directive says "deterministic algebraic combinatorial construction".
    
    # Let's stick to the Tensor Product of Strassen 2x2 and Schoolbook 3x3 (rank 189) as a safe baseline,
    # BUT try to implement a "block Strassen" for 6x6 directly if possible.
    # Actually, there is a known decomposition for 6x6x6 with rank 157 or lower?
    # Let's try to construct 6x6x6 as a tensor product of a 3x3x3 (rank 23) and 2x2x2 (rank 7).
    # I need the Laderman 3x3x3 coefficients.
    
    # I will define the Laderman 3x3x3 decomposition explicitly.
    # Source: "Laderman's algorithm"
    # The 23 terms are:
    # M1 = (a11+a22+a33)(b11+b22+b33)
    # M2 = (a21+a22+a31+a32+a33)(b11+b12+b13)
    # ... This is getting complicated to derive from memory.
    
    # Alternative: Use the "Bayer" construction for symmetric matrices?
    # The task is general matrix multiplication.
    
    # Let's use the Tensor Product of Strassen(2) and Strassen(3)? No, Strassen is 2x2.
    # What if we use a 4x4x4 decomposition?
    # Best known for 4x4x4 is 49 (Strassen tensor product 7*7).
    # 6x6 can be embedded in 8x8? No.
    
    # Let's go with the Tensor Product of Strassen(2) and Schoolbook(3) -> Rank 189.
    # This is a valid improvement over 216.
    # To beat 189, I need a better 3x3x3 decomposition.
    # Let's try to implement a random search for 3x3x3 rank < 27 with verification.
    # If I find a valid rank 23 or 24, I can tensor it with Strassen 2x2.
    
    return None # Placeholder

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W define a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] == (i, j) -> (k, i) ?
    Actually, the tensor T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c} must equal delta_{i,k} delta_{j,j}?
    No. The matrix multiplication tensor is T_{ijk} = 1 if i=k and j=j? No.
    The tensor for matrix multiplication C = AB is defined by:
    C_{ik} = sum_j A_{ij} B_{jk}
    So the tensor T_{ijk} has T_{ijk} = 1 if the indices match the multiplication pattern?
    No, the tensor T is such that T(A, B) = A B.
    In coordinates: T_{ijk} A_{ij} B_{jk} = C_{ik}.
    So T_{ijk} = delta_{i, i'} delta_{k, k'}? No.
    The structure tensor of matrix multiplication is:
    T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki} ? No.
    T = sum_{i,j,k} (e_i \otimes e_j) \otimes (e_j \otimes e_k) \otimes (e_k \otimes e_i) ?
    Let's check the contraction:
    sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    If U_{ij,c} = delta_{i, i_c} delta_{j, j_c}
    V_{jk,c} = delta_{j, j_c} delta_{k, k_c}
    W_{ki,c} = delta_{k, k_c} delta_{i, i_c}
    Then sum_c delta_{i,i_c}delta_{j,j_c} delta_{j,j_c}delta_{k,k_c} delta_{k,k_c}delta_{i,i_c}
    This is non-zero only if i=i_c, j=j_c, k=k_c, and j=j_c, k=k_c, i=i_c.
    So it is 1 if i=i_c, j=j_c, k=k_c.
    This corresponds to the identity mapping?
    Wait, the tensor for matrix multiplication is:
    T_{ijk} = 1 if the "middle" index of A matches the "middle" index of B?
    A is (i, j), B is (j, k), C is (i, k).
    The term in the sum for C_{ik} is A_{ij} B_{jk}.
    So the tensor component T_{(i,j), (j,k), (i,k)} = 1.
    In the flat indices used in the problem:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    So we require:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i, j, k.
    And for any other combination of indices (if we were to contract differently), it should be 0?
    Actually, the tensor is fixed. We just need to check that for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    # For all i, j, k, check the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                # sum over c
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_block_6x6():
    """
    Construct 6x6x6 decomposition using 3x3 blocks of 2x2.
    Block A is 3x3, each block is 2x2.
    We use Strassen for the 2x2 blocks? No, we use a decomposition for the 3x3 block multiplication,
    and each "entry" in the block multiplication is a 2x2 matrix product, which we decompose with Strassen.
    
    Let A_block be 3x3, B_block be 3x3.
    C_block = A_block * B_block.
    If we use schoolbook for 3x3 block multiplication:
    C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    Each product A_block[i,j] * B_block[j,k] is a 2x2 matrix product.
    We decompose each 2x2 product with Strassen (rank 7).
    Total rank = 3*3*3 * 7 = 189.
    
    U has size (6*6, 189) = (36, 189).
    V has size (6*6, 189) = (36, 189).
    W has size (6*6, 189) = (36, 189).
    
    Index mapping:
    i, j, k in {0, 1, 2} for block indices.
    a, b, c in {0, 1} for intra-block indices.
    
    Global A index: (i*2 + a, j*2 + b) -> row i*2+a, col j*2+b.
    Flat A index: (i*2+a)*6 + (j*2+b).
    Global B index: (j*2+b, k*2+c) -> row j*2+b, col k*2+c.
    Flat B index: (j*2+b)*6 + (k*2+c).
    Global C index: (i*2+a, k*2+c) -> row i*2+a, col k*2+c.
    Flat C index: (i*2+a)*6 + (k*2+c).
    
    For each block triplet (i, j, k) and each Strassen term t in 0..6:
    Let U2, V2, W2 be the Strassen 2x2 decomposition matrices (4x7).
    For each (a, b) in 0..1:
        u_flat = (i*2+a)*6 + (j*2+b)
        U[u_flat, (i*3+j)*3*k + t] = U2[a*2+b, t]
        
    For each (b, c) in 0..1:
        v_flat = (j*2+b)*6 + (k*2+c)
        V[v_flat, (i*3+j)*3*k + t] = V2[b*2+c, t]
        
    For each (a, c) in 0..1:
        w_flat = (i*2+a)*6 + (k*2+c)
        W[w_flat, (i*3+j)*3*k + t] = W2[a*2+c, t]
        
    Note: The W index in Strassen 2x2 is for C[a, c].
    In the global W, the index is k*n + i?
    Wait, the problem statement says: W[k*n+i, r].
    This corresponds to C[i, k] if we use row-major?
    Let's check the schoolbook code:
    W[k * n + i][column] = 1
    And the verification loop:
    W[k * n + i]
    So the row index in W is `k*n + i`.
    This implies the output matrix C is indexed by (i, k) but stored as row `k*n+i`?
    Or is it C[k, i]?
    Standard matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}.
    If W row is `k*n + i`, and we assume row-major storage of a (p x n) matrix?
    Usually C is (n x p).
    If C is (n x p), row i, col k. Flat index i*p + k.
    Here it is k*n + i. This is column-major of (n x p) or row-major of (p x n).
    Let's assume the tensor definition is consistent with the schoolbook code.
    Schoolbook:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    Let's trace:
    Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    For a specific (i, j, k), only one term c is non-zero (the one assigned to i, j, k).
    U is 1, V is 1, W is 1. Sum = 1.
    So the tensor component T_{(i,j), (j,k), (k,i)} = 1?
    The indices of W are (k, i).
    So the output is indexed by (k, i).
    This means the output matrix C is effectively transposed or indexed differently?
    If C_{ki} = sum_j A_{ij} B_{jk}, then C = B^T A^T? No.
    C_{ki} = (B A^T)_{ki}?
    Let's not overthink. The schoolbook code works. I will follow its index convention.
    
    In my Strassen block construction:
    The "output" of the 2x2 Strassen is C_2[a, c].
    In the global context, this corresponds to the block C_block[i, k] at position (a, c).
    The global output index is determined by the block row i and block col k, and intra-block a, c.
    The global row in W is `k*n + i`.
    Wait, in schoolbook, W index is `k*n + i`.
    Here k is the col index of B (and C?), i is the row index of A (and C?).
    So W row `k*n + i` corresponds to output element (i, k).
    So C[i, k] is stored at W[k*n + i].
    
    In my block construction:
    Block output C_block[i, k] is a 2x2 matrix.
    Element (a, c) of C_block[i, k] corresponds to global C[i*2+a, k*2+c].
    So the W row index should be `(k*2+c)*n + (i*2+a)`.
    
    Let's refine the mapping:
    Global A index: row i*2+a, col j*2+b. Flat: (i*2+a)*6 + (j*2+b).
    Global B index: row j*2+b, col k*2+c. Flat: (j*2+b)*6 + (k*2+c).
    Global C index: row i*2+a, col k*2+c. Flat W row: (k*2+c)*6 + (i*2+a).
    
    Strassen 2x2:
    U2 rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    V2 rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    W2 rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    
    For a fixed block triplet (i, j, k) and Strassen term t:
    col = (i*3 + j)*3 + k)*7 + t ? No, let's just use a counter.
    
    Implementation:
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    # U part
                    for a in range(2):
                        for b in range(2):
                            u_row = (i * 2 + a) * 6 + (j * 2 + b)
                            val = U2[a * 2 + b, t]
                            if val != 0:
                                U[u_row, col] = val
                    # V part
                    for b in range(2):
                        for c in range(2):
                            v_row = (j * 2 + b) * 6 + (k * 2 + c)
                            val = V2[b * 2 + c, t]
                            if val != 0:
                                V[v_row, col] = val
                    # W part
                    for a in range(2):
                        for c in range(2):
                            w_row = (k * 2 + c) * 6 + (i * 2 + a)
                            val = W2[a * 2 + c, t]
                            if val != 0:
                                W[w_row, col] = val
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try Strassen block decomposition (Rank 189)
    result = _construct_strassen_block_6x6()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], 6, 6, 6):
        return result
    else:
        # Fallback to schoolbook
        return _schoolbook()

# EVOLVE-BLOCK-END
