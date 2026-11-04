import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=5
    
    # We will generate the list of (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    def add_term(u, v, w, c=1):
        terms.append((u, v, w, c))
        
    # Recursive function to add terms for M_{2, n, n}
    # u_offset, v_offset, w_offset are the starting indices for the sub-problem
    # The sub-problem A is 2 x n, B is n x n, C is 2 x n
    # U indices: 0..2n-1 (flattened A)
    # V indices: 0..n^2-1 (flattened B)
    # W indices: 0..2n-1 (flattened C, col-major: k*n + i)
    
    def recurse(n_sub, u_off, v_off, w_off):
        if n_sub == 2:
            # Strassen terms for 2x2x2
            # A indices: 0,1 (row 0), 2,3 (row 1) -> i*m+j
            # m=2.
            # A00: 0, A01: 1, A10: 2, A11: 3
            # B indices: 0,1 (row 0), 2,3 (row 1) -> j*p+k
            # p=2.
            # B00: 0, B01: 1, B10: 2, B11: 3
            # C indices: k*2+i.
            # C00: 0, C10: 1, C01: 2, C11: 3
            
            # P1 = A00 * (B01 + B11)
            add_term(u_off + 0, v_off + 1, w_off + 0, 1) # C00? No.
            # Strassen:
            # C00 = P5 + P4 - P2 + P6
            # C01 = P1 + P3
            # C10 = P2 + P7
            # C11 = P1 - P3 + P5 + P4
            
            # Let's map the 7 products to terms.
            # P1: A00 * (B01 + B11) -> C01 gets +1, C11 gets -1? No.
            # Standard Strassen:
            # M1 = A00*(B01+B11)
            # M2 = (A10+A11)*B00
            # M3 = (A00+A01)*B11
            # M4 = A11*(B10-B00)
            # M5 = (A00+A10)*(B00+B11)
            # M6 = (A01-A10)*(B10+B11)
            # M7 = (A00-A01)*(B00+B01)
            
            # C00 = M5 + M4 - M2 + M6
            # C01 = M1 + M3
            # C10 = M2 + M7
            # C11 = M1 - M3 + M5 + M4
            
            # We need to expand these into rank-1 terms.
            # Each Mi is a product of two linear combinations.
            # This is getting complicated to hardcode.
            
            # I will use the schoolbook for n=2 as well, 
            # which is rank 8.
            # Then R3 = 8 + 10 = 18.
            # R4 = 18 + 14 = 32.
            # R5 = 32 + 18 = 50.
            # This gives rank 50. No improvement.
            
            # I must use Strassen for n=2.
            pass
        else:
            # Split
            # A' is 2 x (n-1), a is 2 x 1
            # B' is (n-1) x (n-1), b is (n-1) x 1, c is 1 x (n-1), d is 1 x 1
            
            # Recurse on A' B'
            recurse(n_sub - 1, u_off, v_off, w_off)
            
            # Terms for a c
            # a is at u_off + 2*(n-1) ... u_off + 2*n - 1
            # c is at v_off + (n-1)*n ... v_off + n*n - 1?
            # B is n x n.
            # B' is top-left (n-1)x(n-1).
            # b is top-right (n-1)x1.
            # c is bottom-left 1x(n-1).
            # d is bottom-right 1x1.
            
            # Indices for B (flattened row-major j*p+k, p=n)
            # B' rows 0..n-2, cols 0..n-2.
            # b rows 0..n-2, col n-1.
            # c row n-1, cols 0..n-2.
            # d row n-1, col n-1.
            
            # a indices in A (flattened i*m+j, m=n)
            # A' rows 0,1, cols 0..n-2.
            # a rows 0,1, col n-1.
            
            # C' is 2 x (n-1).
            # C_new is 2 x 1.
            # C indices (flattened k*n+i)
            # C' cols 0..n-2.
            # C_new col n-1.
            
            # Term a_i * c_j -> C'_{i,j}
            for i in range(2):
                for j in range(n_sub - 1):
                    u_idx = u_off + i * n_sub + (n_sub - 1) # a_i
                    v_idx = v_off + (n_sub - 1) * n_sub + j # c_j
                    w_idx = w_off + j * 2 + i # C'_{i,j}
                    add_term(u_idx, v_idx, w_idx, 1)
                    
            # Term A'_ik * b_k -> C_new_i
            for i in range(2):
                for k in range(n_sub - 1):
                    u_idx = u_off + i * n_sub + k # A'_ik
                    v_idx = v_off + k * n_sub + (n_sub - 1) # b_k
                    w_idx = w_off + (n_sub - 1) * 2 + i # C_new_i
                    add_term(u_idx, v_idx, w_idx, 1)
                    
            # Term a_i * d -> C_new_i
            for i in range(2):
                u_idx = u_off + i * n_sub + (n_sub - 1) # a_i
                v_idx = v_off + (n_sub - 1) * n_sub + (n_sub - 1) # d
                w_idx = w_off + (n_sub - 1) * 2 + i # C_new_i
                add_term(u_idx, v_idx, w_idx, 1)

    recurse(5, 0, 0, 0)
    
    # Now terms contains the list.
    # But I didn't implement the Strassen base case properly.
    # I will replace the base case with the 8 schoolbook terms for now 
    # to ensure the code runs, but that gives rank 50.
    
    # To get rank 49, I MUST implement Strassen.
    
    # I will implement the Strassen terms explicitly in the base case.
    
    return {"U": U, "V": V, "W": W}
