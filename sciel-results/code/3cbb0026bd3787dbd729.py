import numpy as np
import time

DIMENSIONS = [4, 5, 5]
_START = time.time()
_DEADLINE = 120.0  # seconds

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_term(i, k, j, coeff_u=1.0, coeff_v=1.0, coeff_w=1.0):
        """
        Adds a term: coeff_u * A[i,k] * coeff_v * B[k,j] -> coeff_w * C[i,j]
        u_vec has coeff_u at i*m+k
        v_vec has coeff_v at k*p+j
        w_vec has coeff_w at j*n+i
        """
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        u_vec[i * m + k] = coeff_u
        v_vec[k * p + j] = coeff_v
        w_vec[j * n + i] = coeff_w
        
        add_rank1(u_vec, v_vec, w_vec)
    
    def add_term_with_linear_forms(a_terms, b_terms, c_term, c_coeff):
        """
        Add a rank-1 term where:
        - a_terms: list of (i, k, coeff) for A
        - b_terms: list of (k, j, coeff) for B  
        - c_term: (i, j) for C
        - c_coeff: coefficient for C[i,j]
        """
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, k, coeff) in a_terms:
            u_vec[i * m + k] += coeff
        for (k, j, coeff) in b_terms:
            v_vec[k * p + j] += coeff
        (ci, cj) = c_term
        w_vec[cj * n + ci] = c_coeff
        
        add_rank1(u_vec, v_vec, w_vec)
    
    # Part 1: k=4 (4x1x5 schoolbook, rank 20)
    for i in range(n):
        for j in range(p):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    # Part 2: k in {0,1,2,3} (4x4x5 using Strassen on 4x4x4 + 4x4x1)
    # Split j into j in {0,1,2,3} (4x4x4 Strassen) and j=4 (4x4x1 schoolbook)
    
    # 4x4x1 schoolbook (rank 16)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # 4x4x4 Strassen (rank 28)
    # Define the 7 Strassen products with their linear forms
    # Each product is a bilinear form in A and B that contributes to C
    
    # P1: (A00+A11)(B00+B11)
    # A00: i in {0,1}, k in {0,1}; A11: i in {2,3}, k in {2,3}
    # B00: k in {0,1}, j in {0,1}; B11: k in {2,3}, j in {2,3}
    # C contributions: C00, C11
    
    # P2: (A10+A11)B00
    # A10: i in {2,3}, k in {0,1}; A11: i in {2,3}, k in {2,3}
    # B00: k in {0,1}, j in {0,1}
    # C contributions: C10, C11
    
    # P3: A00(B01-B11)
    # A00: i in {0,1}, k in {0,1}
    # B01: k in {0,1}, j in {2,3}; B11: k in {2,3}, j in {2,3}
    # C contributions: C01
    
    # P4: A11(B10-B00)
    # A11: i in {2,3}, k in {2,3}
    # B10: k in {2,3}, j in {0,1}; B00: k in {0,1}, j in {0,1}
    # C contributions: C00, C10
    
    # P5: (A00+A01)B11
    # A00: i in {0,1}, k in {0,1}; A01: i in {0,1}, k in {2,3}
    # B11: k in {2,3}, j in {2,3}
    # C contributions: C00, C01
    
    # P6: (A10-A00)(B00+B01)
    # A10: i in {2,3}, k in {0,1}; A00: i in {0,1}, k in {0,1}
    # B00: k in {0,1}, j in {0,1}; B01: k in {0,1}, j in {2,3}
    # C contributions: C11
    
    # P7: (A01-A11)(B10+B11)
    # A01: i in {0,1}, k in {2,3}; A11: i in {2,3}, k in {2,3}
    # B10: k in {2,3}, j in {0,1}; B11: k in {2,3}, j in {2,3}
    # C contributions: C00, C11
    
    # For each product, we need to expand into 4 schoolbook terms (2x2x2)
    # Then combine with the C reconstruction formulas
    
    # Let me define each product's 4 terms explicitly
    
    # P1 = (A00+A11)(B00+B11)
    # = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    # Each of these is a 2x2x2 product
    
    # Term P1_1: A00*B00
    # A00: i in {0,1}, k in {0,1}
    # B00: k in {0,1}, j in {0,1}
    for li in range(2):
        for lk in range(2):
            for lj in range(2):
                # C contribution from P1 to C00 (i=li, j=lj) and C11 (i=li+2, j=lj+2)
                # C00 = P1 + P4 - P5 + P7
                # C11 = P1 - P2 + P3 + P6
                add_term_with_linear_forms(
                    [(li, lk, 1.0)],
                    [(lk, lj, 1.0)],
                    (li, lj),
                    1.0  # Coefficient for C00
                )
                add_term_with_linear_forms(
                    [(li, lk, 1.0)],
                    [(lk, lj, 1.0)],
                    (li+2, lj+2),
                    1.0  # Coefficient for C11
                )
    
    # This approach is getting complex. Let me use a cleaner formulation.
    # I'll define the 28 terms directly.
    
    # Actually, let me restart with a cleaner approach.
    # Clear the columns and rebuild.
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()
    
    # Part 1: k=4 (4x1x5 schoolbook, rank 20)
    for i in range(n):
        for j in range(p):
            add_term(i, 4, j, 1.0, 1.0, 1.0)
    
    # Part 2: k in {0,1,2,3}, j=4 (4x4x1 schoolbook, rank 16)
    for i in range(4):
        for k in range(4):
            add_term(i, k, 4, 1.0, 1.0, 1.0)
    
    # Part 3: k in {0,1,2,3}, j in {0,1,2,3} (4x4x4 Strassen, rank 28)
    # Define the 7 Strassen products, each as a 2x2x2 schoolbook product
    
    # Helper to add a 2x2x2 schoolbook product with given A and B linear forms
    def add_2x2x2_product(a_form, b_form, c_form):
        """
        a_form: 2x2 array of coefficients for A[i,k] where i,k in {0,1}
        b_form: 2x2 array of coefficients for B[k,j] where k,j in {0,1}
        c_form: dict mapping (i,j) to coefficient for C[i,j]
        """
        for li in range(2):
            for lk in range(2):
                for lj in range(2):
                    a_coeff = a_form[li][lk]
                    b_coeff = b_form[lk][lj]
                    if a_coeff == 0 or b_coeff == 0:
                        continue
                    for (ci, cj), c_coeff in c_form.items():
                        if c_coeff == 0:
                            continue
                        add_term_with_linear_forms(
                            [(li, lk, a_coeff)],
                            [(lk, lj, b_coeff)],
                            (ci, cj),
                            c_coeff
                        )
    
    # P1: (A00+A11)(B00+B11)
    # A00: i in {0,1}, k in {0,1} -> a_form for A00 is identity
    # A11: i in {2,3}, k in {2,3} -> but we're working in local 2x2 coords
    # Wait, I need to be more careful.
    
    # Let me define each product as a combination of 2x2x2 products.
    # For P1 = (A00+A11)(B00+B11):
    # This equals A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each is a 2x2x2 product.
    
    # A00*B00: A[i,k] for i,k in {0,1}, B[k,j] for k,j in {0,1}
    # Contributes to C00[i,j] for i,j in {0,1} with coeff 1
    # And to C11[i+2,j+2] for i,j in {0,1} with coeff 1
    
    # This is getting too complex. Let me just use schoolbook for 4x4x4 (rank 64).
    # Total rank: 20 + 16 + 64 = 100. Same as before.
    
    # Actually, let me try a different approach: use the known rank-17 decomposition for 3x3x3.
    # But I don't have it hardcoded.
    
    # Let me just return the schoolbook decomposition for now.
    for i in range(4):
        for k in range(4):
            for j in range(4):
                add_term(i, k, j, 1.0, 1.0, 1.0)
    
    rank = len(U_cols)
    
    if rank == 0:
        add_term(0, 0, 0, 1.0, 1.0, 1.0)
        rank = 1
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
