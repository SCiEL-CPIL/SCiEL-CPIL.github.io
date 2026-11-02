import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists, then convert to arrays if needed?
    # The interface expects {'U', 'V', 'W'} with shapes.
    # The seed program uses lists of lists. I will stick to that or use numpy arrays.
    # Numpy is available.
    
    # Start with empty lists for the terms
    # Each term is a tuple (u_indices, v_indices, w_indices, coeffs)
    # Actually, a rank-1 term is defined by vectors u, v, w.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # So for each rank r, we have a vector U[:, r], V[:, r], W[:, r].
    
    terms = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        terms.append((u_vec, v_vec, w_vec))
        
    # Generate Strassen 4x4x4 terms
    # Recursive function to generate terms for size N x N x N
    def strassen_terms(n_size, offset_i, offset_j, offset_k, u_base, v_base, w_base):
        # This is tricky because the indices are global.
        # It's easier to generate the symbolic expressions for U, V, W for a 2x2x2 block
        # and then map them.
        
        pass

    # Let's hardcode the 49 terms for 4x4x4 using the 7x7 structure.
    # Or just use the known rank 49 decomposition.
    
    # Actually, I can just use the schoolbook for the 4x4x4 part if I can't generate Strassen easily?
    # No, schoolbook is 64. Strassen is 49.
    # 49 + 8 = 57.
    # 64 + 16 = 80.
    # So Strassen is much better.
    
    # Let's implement the Strassen term generation.
    
    # A term is defined by:
    # u: list of (index, coeff)
    # v: list of (index, coeff)
    # w: list of (index, coeff)
    
    def make_vec(size):
        return [0.0] * size
        
    def add_to_vec(vec, idx, coeff):
        vec[idx] += coeff
        
    # Strassen 2x2x2 terms (local indices 0,1)
    # Returns list of (u_local, v_local, w_local)
    def strassen_2x2x2_terms():
        # u indices: i*2+j
        # v indices: j*2+k
        # w indices: k*2+i
        
        t = []
        
        # M1: (A00+A11)(B00+B11)
        u = [0, 2, 0, 2] # A00, A11, A00, A11? No.
        # A is 2x2. A00 is (0,0). A11 is (1,1).
        # u indices: 0*2+0=0, 1*2+1=3.
        # So u has 1 at 0 and 1 at 3.
        
        # Let's define the linear forms.
        
        # M1
        u1 = [1, 0, 0, 1] # A00, A01, A10, A11 -> indices 0, 1, 2, 3
        v1 = [1, 0, 1, 0] # B00, B01, B10, B11 -> indices 0, 1, 2, 3
        w1 = [1, 1, 1, 1] # C00, C01, C10, C11? No.
        # W index: k*2+i.
        # C00: k=0, i=0 -> 0.
        # C01: k=1, i=0 -> 2.
        # C10: k=0, i=1 -> 1.
        # C11: k=1, i=1 -> 3.
        # M1 contributes to C00, C01, C10, C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M5
        # C10 = M3 + M6
        # C11 = M1 - M2 + M3 + M4
        
        # So W vector for M1 has 1 at C00, C11?
        # No, the tensor decomposition is $\sum U V W = T$.
        # $T_{u,v,w} = \sum_r U[u,r] V[v,r] W[w,r]$.
        # The target tensor has 1s at specific (u,v,w).
        
        # This approach is getting complicated to hardcode.
        # I will use a simpler strategy:
        # Use the known rank 49 decomposition for 4x4x4.
        # I can find it online or derive it.
        
        # For now, let's just output the schoolbook rank 80 to be safe?
        # No, the prompt asks for a stronger algorithm.
        # Rank 57 is a significant improvement.
        
        # I will implement the 49+8 construction.
        
        pass

    # I will write the code to generate the 49 terms for 4x4x4.
    # The 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B12-B22)
    # M3 = (A11-A22)(B21+B22)
    # M4 = A11(B22-B21)
    # M5 = (A12+A22)(B11+B21)
    # M6 = (A11+A12)(B22-B21)
    # M7 = (A21+A22)(B11-B21)
    
    # Each M is a 2x2 matrix product.
    # So each M expands to 7 scalar multiplications.
    
    # I will create a function that takes the linear forms for A and B for a specific M
    # and generates the 7 terms for the 2x2x2 multiplication.
    
    # The indices in the 4x4x4 problem:
    # A is 4x4.
    # A11 is A[0:2, 0:2].
    # A12 is A[0:2, 2:4].
    # etc.
    
    # I will hardcode the 49 terms.
    
    return {"U": [], "V": [], "W": []}
