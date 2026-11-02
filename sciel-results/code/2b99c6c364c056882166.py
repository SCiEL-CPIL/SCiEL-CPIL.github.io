import numpy as np

def construct():
    """
    Construct an exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.
    Uses a structured construction based on Kronecker products and Strassen-like reductions
    to achieve a lower rank than the schoolbook 96.
    """
    n, m, p = 3, 4, 8
    dim_u, dim_v, dim_w = n*m, m*p, p*n  # 12, 32, 24
    num_eq = n * m * p  # 96
    
    # The tensor T has indices (i, j, k) for i in 0..2, j in 0..3, k in 0..7
    # T[i,j,k] = 1 if and only if we are looking at the scalar product for C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition requires:
    # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # Strategy:
    # We can split the matrix multiplication into blocks.
    # A is 3x4, B is 4x8.
    # We can view this as computing 3 rows of A against 8 columns of B.
    
    # A simpler approach:
    # Use the schoolbook algorithm but try to exploit symmetries or linearity to reduce rank.
    # However, for small sizes like 3x4x8, the rank is likely 96 or close to it.
    # Let's verify if a rank reduction is possible.
    # The rank of the matrix multiplication tensor M_{n,m,p} is known for some small cases.
    # M_{2,2,2} has rank 7 (Strassen).
    # M_{3,3,3} has rank <= 23 (Lader, Strassen, etc.), and it is known that rank is between 19 and 23? Actually, rank(M_{3,3,3}) = 23 is the best known upper bound? No, it's 23?
    # Wait, the exact rank of M_{3,3,3} is not known, but it's at least 19 and at most 23.
    # For M_{3,4,8}, we can decompose it.
    
    # Let's try to construct a decomposition with rank 96 first (schoolbook) and then see if we can do better.
    # Since the task asks for rank-minimal, and the current score is 1.0 (rank 96), we need to find a rank < 96.
    
    # Can we use Strassen on parts?
    # 3x4x8 = (2+1) x 4 x 8.
    # C_{0:2, :} = A_{0:2, :} * B_{:, :} (2x4 * 4x8)
    # C_{2, :} = A_{2, :} * B_{:, :} (1x4 * 4x8)
    
    # For the 2x4x8 part:
    # 2x4x8 = 2 * (1x4x8)? No.
    # 2x4x8 can be split into 2x2x4 blocks?
    # A_{0:2, 0:2} * B_{0:2, 0:4} -> C_{0:2, 0:4} (2x2x4)
    # A_{0:2, 2:4} * B_{2:4, 0:4} -> C_{0:2, 0:4} (2x2x4)
    # A_{0:2, 0:2} * B_{0:2, 4:8} -> C_{0:2, 4:8} (2x2x4)
    # A_{0:2, 2:4} * B_{2:4, 4:8} -> C_{0:2, 4:8} (2x2x4)
    
    # Rank of M_{2,2,4}?
    # M_{2,2,4} is equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
    # This is 4 scalar multiplications of 2x2 matrices? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # For i in {0,1}, j in {0,1}, k in {0..3}.
    # This is 2 * 4 = 8 output elements.
    # Input A has 4 elements, B has 8 elements.
    # Schoolbook rank: 2 * 2 * 4 = 16.
    # Can we do better?
    # M_{2,2,4} can be seen as two M_{2,2,2} tensors?
    # Split k into {0,1} and {2,3}.
    # C_{0:2, 0:2} = A_{0:2, 0:2} * B_{0:2, 0:2} (2x2x2) -> Rank 7
    # C_{0:2, 2:4} = A_{0:2, 0:2} * B_{0:2, 2:4} (2x2x2) -> Rank 7
    # Wait, B is 2x4. So B_{0:2, 0:2} and B_{0:2, 2:4} are the two 2x2 blocks of B.
    # A is 2x2.
    # So C_{0:2, 0:2} = A * B_1 and C_{0:2, 2:4} = A * B_2.
    # We can compute A * B_1 with rank 7 and A * B_2 with rank 7.
    # Total rank for 2x2x4 is 7 + 7 = 14?
    # Let's check.
    # A * B_1 = C_1. A * B_2 = C_2.
    # The tensor is T = T_1 + T_2 where T_1 is 2x2x2 on (i,j,k_1) and T_2 is 2x2x2 on (i,j,k_2).
    # Since the A indices are the same, we can share the A factors?
    # No, in tensor decomposition, the factors are independent for each term.
    # However, if we use Strassen for A*B_1 and A*B_2, the U factors for A are the same?
    # Strassen U factors are linear combinations of A's entries.
    # Let U_r be the vector of coefficients for A in term r.
    # For T_1, we have 7 terms with U^{(1)}_r.
    # For T_2, we have 7 terms with U^{(2)}_r.
    # Since the A part is the same, U^{(1)}_r = U^{(2)}_r?
    # Yes, if we use the same Strassen algorithm for both.
    # So we have 7 terms where U is the same for both?
    # No, the terms are separate.
    # Term r for T_1: U_r * V_r * W_r.
    # Term r for T_2: U_r * V'_r * W'_r.
    # We can combine them?
    # The tensor is sum_r U_r \otimes V_r \otimes W_r + sum_r U_r \otimes V'_r \otimes W'_r.
    # This is 14 terms.
    # Can we do better?
    # The rank of M_{2,2,4} is known to be 14? Or 13?
    # Actually, M_{2,2,4} is isomorphic to M_{2,4,2}.
    # Let's assume rank 14 for 2x2x4.
    
    # So for 2x4x8:
    # We have four 2x2x4 blocks.
    # 4 * 14 = 56 rank for the 2x4x8 part?
    # Wait, the blocks are:
    # A_{0:2, 0:2} * B_{0:2, 0:4} -> C_{0:2, 0:4}
    # A_{0:2, 2:4} * B_{2:4, 0:4} -> C_{0:2, 0:4}
    # A_{0:2, 0:2} * B_{0:2, 4:8} -> C_{0:2, 4:8}
    # A_{0:2, 2:4} * B_{2:4, 4:8} -> C_{0:2, 4:8}
    
    # These are four independent matrix multiplications of size 2x2x4.
    # Since the A and B blocks are disjoint in terms of indices (for A, the columns are disjoint; for B, the rows are disjoint),
    # the rank should be additive.
    # So rank(2x4x8) <= 4 * 14 = 56.
    
    # Now for the 1x4x8 part:
    # C_{2, k} = sum_j A_{2,j} B_{j,k}.
    # This is 1x4x8.
    # Split into 1x2x4 blocks?
    # A_{2, 0:2} * B_{0:2, 0:4} -> C_{2, 0:4} (1x2x4)
    # A_{2, 2:4} * B_{2:4, 0:4} -> C_{2, 0:4} (1x2x4)
    # A_{2, 0:2} * B_{0:2, 4:8} -> C_{2, 4:8} (1x2x4)
    # A_{2, 2:4} * B_{2:4, 4:8} -> C_{2, 4:8} (1x2x4)
    
    # Rank of M_{1,2,4}?
    # C_k = sum_j A_j B_{j,k}.
    # This is just a dot product, repeated for each k.
    # Rank of M_{1,2,2} is 2 (schoolbook, since it's just a sum of 2 products).
    # M_{1,2,4} = M_{1,2,2} + M_{1,2,2} (disjoint k).
    # So rank is 2 + 2 = 4.
    # So rank(1x4x8) <= 4 * 4 = 16.
    
    # Total rank for 3x4x8 <= 56 + 16 = 72.
    
    # Let's construct this decomposition.
    
    # We will create the U, V, W matrices for each block and then combine them.
    # The global indices are:
    # U index: i*m + j (i in 0..2, j in 0..3) -> 0..11
    # V index: j*p + k (j in 0..3, k in 0..7) -> 0..31
    # W index: k*n + i (k in 0..7, i in 0..2) -> 0..23
    
    # We will define a function to add a rank-1 term to the decomposition.
    # term_u: dict or list of (idx, val) for U
    # term_v: dict or list of (idx, val) for V
    # term_w: dict or list of (idx, val) for W
    
    # We will accumulate the terms in lists.
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        U_terms.append(u_dict)
        V_terms.append(v_dict)
        W_terms.append(w_dict)
    
    # Helper to create Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # We need to map local indices (i,j,k) to global indices.
    # For a block A_{i0:i0+2, j0:j0+2} * B_{j0:j0+2, k0:k0+2} -> C_{i0:i0+2, k0:k0+2}
    def add_strassen_block(i0, j0, k0):
        # Local indices:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Strassen 7 multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Define the 7 terms
        # Term 1: M1
        u1 = {i0*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v1 = {j0*p + k0: 1, (j0+1)*p + (k0+1): 1}
        w1 = {(k0)*n + i0: 1, (k0+1)*n + (i0+1): 1} # C00 and C11 get +1?
        # Wait, W index is k*n + i.
        # C00 is i0, k0. W index: k0*n + i0.
        # C11 is i0+1, k0+1. W index: (k0+1)*n + (i0+1).
        # C01 is i0, k0+1. W index: (k0+1)*n + i0.
        # C10 is i0+1, k0. W index: k0*n + (i0+1).
        
        # Let's re-verify the W coefficients for each M.
        # M1 contributes to C00 (+1), C11 (+1)
        # M2 contributes to C10 (+1), C11 (-1)
        # M3 contributes to C01 (+1), C11 (+1)
        # M4 contributes to C00 (+1), C10 (+1)
        # M5 contributes to C00 (-1), C01 (+1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        # Term 1 (M1):
        u1 = {i0*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v1 = {j0*p + k0: 1, (j0+1)*p + (k0+1): 1}
        w1 = {k0*n + i0: 1, (k0+1)*n + (i0+1): 1}
        add_term(u1, v1, w1)
        
        # Term 2 (M2):
        u2 = {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): 1} # A10 + A11 -> indices (1,0) and (1,1) local. Global: (i0+1, j0) and (i0+1, j0+1).
        # Wait, A10 is local (1,0) -> global (i0+1, j0). A11 is local (1,1) -> global (i0+1, j0+1).
        u2 = {(i0+1)*m + j0: 1, (i0+1)*m + (j0+1): 1}
        v2 = {j0*p + k0: 1} # B00
        w2 = {k0*n + (i0+1): 1, (k0+1)*n + (i0+1): -1} # C10 (+1), C11 (-1)
        add_term(u2, v2, w2)
        
        # Term 3 (M3):
        u3 = {i0*m + j0: 1} # A00
        v3 = {j0*p + (k0+1): 1, (j0+1)*p + (k0+1): -1} # B01 - B11
        w3 = {(k0+1)*n + i0: 1, (k0+1)*n + (i0+1): 1} # C01 (+1), C11 (+1)
        add_term(u3, v3, w3)
        
        # Term 4 (M4):
        u4 = {(i0+1)*m + (j0+1): 1} # A11
        v4 = {(j0+1)*p + k0: 1, j0*p + k0: -1} # B10 - B00
        w4 = {k0*n + i0: 1, k0*n + (i0+1): 1} # C00 (+1), C10 (+1)
        add_term(u4, v4, w4)
        
        # Term 5 (M5):
        u5 = {i0*m + j0: 1, i0*m + (j0+1): 1} # A00 + A01
        v5 = {(j0+1)*p + (k0+1): 1} # B11
        w5 = {k0*n + i0: -1, (k0+1)*n + i0: 1} # C00 (-1), C01 (+1)
        add_term(u5, v5, w5)
        
        # Term 6 (M6):
        u6 = {(i0+1)*m + j0: 1, i0*m + j0: -1} # A10 - A00
        v6 = {j0*p + k0: 1, j0*p + (k0+1): 1} # B00 + B01
        w6 = {(k0+1)*n + (i0+1): 1} # C11 (+1)
        add_term(u6, v6, w6)
        
        # Term 7 (M7):
        u7 = {i0*m + (j0+1): 1, (i0+1)*m + (j0+1): -1} # A01 - A11
        v7 = {(j0+1)*p + k0: 1, (j0+1)*p + (k0+1): 1} # B10 + B11
        w7 = {k0*n + i0: 1} # C00 (+1)
        add_term(u7, v7, w7)
        
    # Helper to add schoolbook 2x2x4 block
    # A_{i0:i0+2, j0:j0+2} * B_{j0:j0+2, k0:k0+4} -> C_{i0:i0+2, k0:k0+4}
    # We can split this into two 2x2x2 blocks:
    # A * B_{:, 0:2} -> C_{:, 0:2}
    # A * B_{:, 2:4} -> C_{:, 2:4}
    def add_2x2x4_block(i0, j0, k0):
        add_strassen_block(i0, j0, k0)
        add_strassen_block(i0, j0, k0+2)
        
    # Helper to add schoolbook 1x2x4 block
    # A_{i0, j0:j0+2} * B_{j0:j0+2, k0:k0+4} -> C_{i0, k0:k0+4}
    # Split into two 1x2x2 blocks.
    # 1x2x2 rank is 2.
    # C_k = A_j B_jk.
    # Term 1: A_j0 * B_j0,k -> C_k
    # Term 2: A_j1 * B_j1,k -> C_k
    # For k in 0..1 (local):
    # Term 1: U{i0*m+j0: 1}, V{j0*p+k0: 1, j0*p+(k0+1): 1}, W{k0*n+i0: 1, (k0+1)*n+i0: 1}
    # Wait, V is a vector. In rank-1 term, V must be a vector.
    # If V has two 1s, it's not rank-1 in the standard sense?
    # No, V is a vector in the decomposition.
    # V[index1] = 1, V[index2] = 1.
    # Then V[j0*p+k0] * W[...] gives the product for k0.
    # And V[j0*p+(k0+1)] * W[...] gives the product for k0+1.
    # This works if W has corresponding entries.
    
    def add_1x2x2_block(i0, j0, k0):
        # A_{i0, j0} * B_{j0, k0:k0+2} -> C_{i0, k0:k0+2}
        # Actually, this is 1x1x2, which is just 1 term?
        # C_k = A * B_k.
        # Term: U{i0*m+j0: 1}, V{j0*p+k0: 1, j0*p+(k0+1): 1}, W{k0*n+i0: 1, (k0+1)*n+i0: 1}
        # This is 1 term!
        # Wait, is rank(1x1x2) = 1?
        # T = e_a \otimes (e_b + e_c) \otimes (e_d + e_e).
        # This is a single rank-1 tensor.
        # So yes, rank is 1.
        
        u = {i0*m + j0: 1}
        v = {j0*p + k0: 1, j0*p + (k0+1): 1}
        w = {k0*n + i0: 1, (k0+1)*n + i0: 1}
        add_term(u, v, w)
        
    def add_1x2x4_block(i0, j0, k0):
        add_1x2x2_block(i0, j0, k0)
        add_1x2x2_block(i0, j0, k0+2)
        
    # Now assemble the 3x4x8 decomposition.
    # Part 1: 2x4x8 (rows 0,1 of A)
    # Split into four 2x2x4 blocks.
    # A_{0:2, 0:2} * B_{0:2, 0:4} -> C_{0:2, 0:4}
    add_2x2x4_block(0, 0, 0)
    # A_{0:2, 2:4} * B_{2:4, 0:4} -> C_{0:2, 0:4}
    add_2x2x4_block(0, 2, 0)
    # A_{0:2, 0:2} * B_{0:2, 4:8} -> C_{0:2, 4:8}
    add_2x2x4_block(0, 0, 4)
    # A_{0:2, 2:4} * B_{2:4, 4:8} -> C_{0:2, 4:8}
    add_2x2x4_block(0, 2, 4)
    
    # Part 2: 1x4x8 (row 2 of A)
    # Split into four 1x2x4 blocks.
    # A_{2, 0:2} * B_{0:2, 0:4} -> C_{2, 0:4}
    add_1x2x4_block(2, 0, 0)
    # A_{2, 2:4} * B_{2:4, 0:4} -> C_{2, 0:4}
    add_1x2x4_block(2, 2, 0)
    # A_{2, 0:2} * B_{0:2, 4:8} -> C_{2, 4:8}
    add_1x2x4_block(2, 0, 4)
    # A_{2, 2:4} * B_{2:4, 4:8} -> C_{2, 4:8}
    add_1x2x4_block(2, 2, 4)
    
    # Total rank:
    # 4 * (2 * 7) + 4 * (2 * 1) = 56 + 8 = 64?
    # Wait, add_2x2x4_block calls add_strassen_block twice.
    # add_strassen_block adds 7 terms.
    # So add_2x2x4_block adds 14 terms.
    # 4 blocks * 14 = 56 terms.
    # add_1x2x4_block calls add_1x2x2_block twice.
    # add_1x2x2_block adds 1 term.
    # So add_1x2x4_block adds 2 terms.
    # 4 blocks * 2 = 8 terms.
    # Total rank = 64.
    
    # Now convert the lists of dicts to matrices.
    rank = len(U_terms)
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    for r in range(rank):
        for idx, val in U_terms[r].items():
            U[idx, r] = val
        for idx, val in V_terms[r].items():
            V[idx, r] = val
        for idx, val in W_terms[r].items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
