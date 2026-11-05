"""Matrix multiplication 3,3,7 - rank-minimal decomposition using block/Strassen structure."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix using block decomposition with Strassen
    for 2x2x2 sub-blocks.
    
    Strategy:
      - Split A (3x3) into blocks: A11 (1x1), A12 (1x2), A21 (2x1), A22 (2x2)
        where row split is 1|2 and col split is 1|2
      - Split B (3x7) into blocks: B11 (1x2), B12 (1x5), B21 (2x2), B22 (2x5)
        where row split is 1|2 and col split is 2|5
      
      Products needed:
        C11 = A11*B11 + A12*B21   (1x1 * 1x2 + 1x2 * 2x2) -> 1x2
        C12 = A11*B12 + A12*B22   (1x1 * 1x5 + 1x2 * 2x5) -> 1x5
        C21 = A21*B11 + A22*B21   (2x1 * 1x2 + 2x2 * 2x2) -> 2x2
        C22 = A21*B12 + A22*B22   (2x1 * 1x5 + 2x2 * 2x5) -> 2x5
      
      For each product, we use:
        - Schoolbook for small rectangular cases
        - Strassen (rank 7) for 2x2x2 case
        
      Let's compute ranks:
        A11*B11: 1x1x2 -> rank 2
        A12*B21: 1x2x2 -> rank 4 (schoolbook) or can do better?
          Actually 1x2x2: C[i,k] = sum_j A[i,j]*B[j,k], i fixed, j=0,1, k=0,1
          This is outer product structure, rank = 2*2 = 4? No, it's 1x2x2 tensor.
          T[i,j,k] = delta_{i,0} * A[0,j] * B[j,k]... 
          Actually for 1xm x mxp, the tensor has rank m*p if we do schoolbook.
          1x2x2: rank 4.
          
        A11*B12: 1x1x5 -> rank 5
        A12*B22: 1x2x5 -> rank 10
        
        A21*B11: 2x1x2 -> rank 4
        A22*B21: 2x2x2 -> rank 7 (Strassen)
        
        A21*B12: 2x1x5 -> rank 10
        A22*B22: 2x2x5 -> need to compute
        
      For 2x2x5: split B into 2x2 and 2x3
        2x2x2: rank 7 (Strassen)
        2x2x3: split B into 2x2 and 2x1
          2x2x2: rank 7
          2x2x1: rank 4
          Total: 11
        So 2x2x5: 7 + 11 = 18
        
      Total rank = 2 + 4 + 5 + 10 + 4 + 7 + 10 + 18 = 60
      
      Let me verify: 2+4=6, +5=11, +10=21, +4=25, +7=32, +10=42, +18=60.
      
      This gives rank 60, which is better than 63.
    """
    n, m, p = DIMENSIONS
    
    # We'll build U, V, W matrices column by column for each sub-problem
    # Each sub-problem contributes some number of rank-1 terms
    
    U_rows = n * m  # 9
    V_rows = m * p  # 21
    W_rows = p * n  # 21
    
    # We'll collect all rank-1 terms as (u_vec, v_vec, w_vec)
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a rank-1 term where only specific indices are non-zero."""
        u_vec = np.zeros(U_rows)
        v_vec = np.zeros(V_rows)
        w_vec = np.zeros(W_rows)
        u_vec[u_idx] = u_val
        v_vec[v_idx] = v_val
        w_vec[w_idx] = w_val
        terms.append((u_vec, v_vec, w_vec))
    
    def add_schoolbook(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset):
        """
        Add schoolbook terms for an n_sub x m_sub x p_sub sub-problem.
        i_offset: offset for i index in A (rows of A)
        j_offset: offset for j index (cols of A / rows of B)
        k_offset: offset for k index in B (cols of B)
        """
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_idx = (i_offset + i) * m + (j_offset + j)
                    v_idx = (j_offset + j) * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + (i_offset + i)
                    add_term(u_idx, 1.0, v_idx, 1.0, w_idx, 1.0)
    
    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        """
        Add Strassen rank-7 terms for a 2x2x2 sub-problem.
        i_offset: offset for i index in A
        j_offset: offset for j index
        k_offset: offset for k index in B
        
        Strassen's algorithm for 2x2x2:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B21 - B11)
        M4 = A22(B12 - B22)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 - M4 + M6 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M3 + M2 - M5
        
        We need to express each M as a combination of rank-1 terms.
        Each M is a product of two 2x2 matrices, but in the tensor decomposition
        context, we need to decompose the bilinear form.
        
        Actually, for the tensor decomposition, each "M" corresponds to a set of
        rank-1 terms. Let me think about this more carefully.
        
        In the tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r, each term
        corresponds to one multiplication. For Strassen, we have 7 "multiplier"
        terms, but each multiplier involves products of sums, so we need to
        expand them.
        
        Let me reconsider. The standard way to implement Strassen in tensor
        decomposition form:
        
        For the 2x2x2 case, we define 7 intermediate values:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B21 - B11)
        M4 = A22(B12 - B22)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        Then:
        C11 = M1 - M4 + M6 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M3 + M2 - M5
        
        In the tensor decomposition, each M_r is a bilinear form in A and B,
        and it contributes to C via the W matrix. The key insight is that each
        M_r can be written as a sum of rank-1 terms in the A-B-C space.
        
        Actually, let me think of it differently. Each M_r is a scalar that is
        a product of a linear form in A and a linear form in B. So M_r = l_A(r) * l_B(r)
        where l_A(r) is a linear combination of A entries and l_B(r) is a linear
        combination of B entries.
        
        Then M_r contributes to C[i,k] via coefficient c_{r,i,k} such that
        C[i,k] = sum_r c_{r,i,k} * l_A(r) * l_B(r)
        
        In the tensor decomposition T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r],
        we have:
        U[i,j,r] corresponds to the coefficient of A[i,j] in l_A(r)
        V[j,k,r] corresponds to the coefficient of B[j,k] in l_B(r)
        W[i,k,r] corresponds to c_{r,i,k}
        
        So for each Strassen multiplier r, we need:
        - The coefficients of A entries in l_A(r)
        - The coefficients of B entries in l_B(r)
        - The coefficients in the output C
        
        Let me define the 2x2 indices:
        A = [[A00, A01],
             [A10, A11]]
        B = [[B00, B01],
             [B10, B11]]
        C = [[C00, C01],
             [C10, C11]]
        
        Strassen multipliers:
        M1: l_A = A00 + A11, l_B = B00 + B11
        M2: l_A = A10 + A11, l_B = B00
        M3: l_A = A00, l_B = B10 - B00
        M4: l_A = A11, l_B = B01 - B11
        M5: l_A = A00 + A01, l_B = B11
        M6: l_A = A10 - A00, l_B = B00 + B01
        M7: l_A = A01 - A11, l_B = B10 + B11
        
        Output combinations:
        C00 = M1 - M4 + M6 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M3 + M2 - M5
        
        So W coefficients:
        For M1: W[00]=1, W[01]=0, W[10]=0, W[11]=1
        For M2: W[00]=0, W[01]=0, W[10]=1, W[11]=-1
        For M3: W[00]=0, W[01]=1, W[10]=0, W[11]=-1
        For M4: W[00]=-1, W[01]=0, W[10]=1, W[11]=0
        For M5: W[00]=0, W[01]=1, W[10]=0, W[11]=-1
        For M6: W[00]=1, W[01]=0, W[10]=0, W[11]=0
        For M7: W[00]=1, W[01]=0, W[10]=0, W[11]=0
        
        Now let me code this up.
        """
        # A indices in local 2x2: (0,0), (0,1), (1,0), (1,1)
        # B indices in local 2x2: (0,0), (0,1), (1,0), (1,1)
        # C indices in local 2x2: (0,0), (0,1), (1,0), (1,1)
        
        # Define the 7 multipliers as (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict mapping (i,j) -> coeff
        # B_coeffs: dict mapping (i,j) -> coeff
        # C_coeffs: dict mapping (i,j) -> coeff
        
        multipliers = [
            # M1
            (
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1}
            ),
            # M2
            (
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                {(1,0): 1, (1,1): -1}
            ),
            # M3
            (
                {(0,0): 1},
                {(1,0): 1, (0,0): -1},
                {(0,1): 1, (1,1): -1}
            ),
            # M4
            (
                {(1,1): 1},
                {(0,1): 1, (1,1): -1},
                {(0,0): -1, (1,0): 1}
            ),
            # M5
            (
                {(0,0): 1, (0,1): 1},
                {(1,1): 1},
                {(0,1): 1, (1,1): -1}
            ),
            # M6
            (
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (0,1): 1},
                {(0,0): 1}
            ),
            # M7
            (
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (1,1): 1},
                {(0,0): 1}
            ),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in multipliers:
            u_vec = np.zeros(U_rows)
            v_vec = np.zeros(V_rows)
            w_vec = np.zeros(W_rows)
            
            for (ai, aj), coeff in a_coeffs.items():
                u_idx = (i_offset + ai) * m + (j_offset + aj)
                u_vec[u_idx] = coeff
            
            for (bi, bj), coeff in b_coeffs.items():
                v_idx = (j_offset + bi) * p + (k_offset + bj)
                v_vec[v_idx] = coeff
            
            for (ci, ck), coeff in c_coeffs.items():
                w_idx = (k_offset + ck) * n + (i_offset + ci)
                w_vec[w_idx] = coeff
            
            terms.append((u_vec, v_vec, w_vec))
    
    # Block decomposition:
    # A split: rows 0 | rows 1,2; cols 0 | cols 1,2
    #   A11 = A[0:1, 0:1] (1x1), i_offset=0, j_offset=0
    #   A12 = A[0:1, 1:3] (1x2), i_offset=0, j_offset=1
    #   A21 = A[1:3, 0:1] (2x1), i_offset=1, j_offset=0
    #   A22 = A[1:3, 1:3] (2x2), i_offset=1, j_offset=1
    #
    # B split: rows 0 | rows 1,2; cols 0,1 | cols 2,3,4,5,6
    #   B11 = B[0:1, 0:2] (1x2), j_offset=0, k_offset=0
    #   B12 = B[0:1, 2:7] (1x5), j_offset=0, k_offset=2
    #   B21 = B[1:3, 0:2] (2x2), j_offset=1, k_offset=0
    #   B22 = B[1:3, 2:7] (2x5), j_offset=1, k_offset=2
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 1x1x2, i_offset=0, j_offset=0, k_offset=0
    add_schoolbook(1, 1, 2, 0, 0, 0)
    # A12*B21: 1x2x2, i_offset=0, j_offset=1, k_offset=0
    add_schoolbook(1, 2, 2, 0, 1, 0)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 1x1x5, i_offset=0, j_offset=0, k_offset=2
    add_schoolbook(1, 1, 5, 0, 0, 2)
    # A12*B22: 1x2x5, i_offset=0, j_offset=1, k_offset=2
    add_schoolbook(1, 2, 5, 0, 1, 2)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 2x1x2, i_offset=1, j_offset=0, k_offset=0
    add_schoolbook(2, 1, 2, 1, 0, 0)
    # A22*B21: 2x2x2, i_offset=1, j_offset=1, k_offset=0
    add_strassen_2x2x2(1, 1, 0)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 2x1x5, i_offset=1, j_offset=0, k_offset=2
    add_schoolbook(2, 1, 5, 1, 0, 2)
    # A22*B22: 2x2x5, i_offset=1, j_offset=1, k_offset=2
    # Split B22 into B22a (2x2, k_offset=2) and B22b (2x3, k_offset=4)
    # A22*B22a: 2x2x2, i_offset=1, j_offset=1, k_offset=2
    add_strassen_2x2x2(1, 1, 2)
    # A22*B22b: 2x2x3, i_offset=1, j_offset=1, k_offset=4
    # Split B22b into B22b1 (2x2, k_offset=4) and B22b2 (2x1, k_offset=6)
    # A22*B22b1: 2x2x2, i_offset=1, j_offset=1, k_offset=4
    add_strassen_2x2x2(1, 1, 4)
    # A22*B22b2: 2x2x1, i_offset=1, j_offset=1, k_offset=6
    add_schoolbook(2, 2, 1, 1, 1, 6)
    
    # Now assemble U, V, W from terms
    rank = len(terms)
    U = np.zeros((U_rows, rank), dtype=float)
    V = np.zeros((V_rows, rank), dtype=float)
    W = np.zeros((W_rows, rank), dtype=float)
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
