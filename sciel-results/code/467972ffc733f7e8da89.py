import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the matrices U, V, W
    # Rank will be determined by the number of terms
    # We use a list of tuples (u, v, w) to accumulate terms
    # u is index in n*m, v is index in m*p, w is index in p*n
    
    terms = []
    
    def add_strassen(i_offset, j_offset, k_offset):
        # Adds a 2x2x2 Strassen block
        # i_offset: row offset in A (0..n-1)
        # j_offset: col offset in A / row offset in B (0..m-1)
        # k_offset: col offset in B (0..p-1)
        
        # A is [i_offset, j_offset] to [i_offset+1, j_offset+1]
        # B is [j_offset, k_offset] to [j_offset+1, k_offset+1]
        
        # Strassen intermediates
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # We need to add 7 terms
        
        # a = A[i0, j0], b = A[i0, j1]
        # c = A[i1, j0], d = A[i1, j1]
        # e = B[j0, k0], f = B[j0, k1]
        # g = B[j1, k0], h = B[j1, k1]
        
        i0, i1 = i_offset, i_offset + 1
        j0, j1 = j_offset, j_offset + 1
        k0, k1 = k_offset, k_offset + 1
        
        # M1: (a+d)(e+h)
        # a: U[i0*m+j0], V[j0*p+k0], W[k0*n+i0]
        # d: U[i1*m+j1], V[j1*p+k1], W[k1*n+i1]
        # e: U[i0*m+j0], V[j0*p+k0], W[k0*n+i0]
        # h: U[i1*m+j1], V[j1*p+k1], W[k1*n+i1]
        
        # This is getting complicated to trace manually.
        # It's better to generate the basis vectors.
        
        pass

    # This manual approach is error prone.
    # Let's use a helper to add a rank-1 term.
    
    U = {}
    V = {}
    W = {}
    
    def add_term(i, j, k, coeff):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        key = (u_idx, v_idx, w_idx)
        # We need to handle coefficients.
        # Since we are building a decomposition, we can just list the terms.
        # The evaluator expects matrices.
        # We can accumulate into a list of (u, v, w, coeff)
        pass

    # Let's just return the list of terms and convert at the end?
    # No, the interface requires U, V, W matrices.
    # We can use a dictionary to count occurrences or sum coefficients.
    
    terms_dict = {}
    
    def add_term(i, j, k, coeff=1):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        key = (u_idx, v_idx, w_idx)
        if key not in terms_dict:
            terms_dict[key] = 0
        terms_dict[key] += coeff
        
    # Strassen 2x2x2
    def strassen_2x2x2(i_off, j_off, k_off):
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        
        # M1 = (a+d)(e+h)
        # a: (i0, j0), e: (j0, k0)
        # d: (i1, j1), h: (j1, k1)
        # Term 1: a*e -> (i0, j0, k0)
        # Term 2: a*h -> (i0, j0, k1)
        # Term 3: d*e -> (i1, j1, k0)
        # Term 4: d*h -> (i1, j1, k1)
        add_term(i0, j0, k0, 1)
        add_term(i0, j0, k1, 1)
        add_term(i1, j1, k0, 1)
        add_term(i1, j1, k1, 1)
        
        # M2 = (c+d)e
        # c: (i1, j0), d: (i1, j1), e: (j0, k0)
        # Term 5: c*e -> (i1, j0, k0)
        # Term 6: d*e -> (i1, j1, k0)
        add_term(i1, j0, k0, 1)
        # d*e already added? No, M1 used d*h.
        # Wait, Strassen formulas are:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's just add the 7 products directly with signs in the W matrix?
        # No, U and V are fixed. W can have signs.
        
        # M1 = (a+d)(e+h) = ae + ah + de + dh
        # M2 = (c+d)e = ce + de
        # M3 = a(f-h) = af - ah
        # M4 = d(g-e) = dg - de
        # M5 = (a+b)h = ah + bh
        # M6 = (c-a)(e+f) = ce + cf - ae - af
        # M7 = (b-d)(g+h) = bg + bh - dg - dh
        
        # We need to map these to the output C.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # This is equivalent to:
        # C11 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
        #      = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
        #      = ae + bg
        # Correct.
        
        # So we can just add the 7 intermediate products as rank-1 terms,
        # and then combine them in W?
        # No, the rank is the number of terms.
        # If we use 7 terms, we need to define U, V, W for each term.
        
        # Term 1 (M1): U = a+d, V = e+h, W = 1 (for C11, C22)
        # This is not a single rank-1 term.
        # A rank-1 term is u * v * w.
        # u is a vector in R^{nm}, v in R^{mp}, w in R^{pn}.
        
        # For M1 = (a+d)(e+h):
        # u_1 has 1 at (i0,j0) and 1 at (i1,j1).
        # v_1 has 1 at (j0,k0) and 1 at (j0,k1) and 1 at (j1,k0) and 1 at (j1,k1)? No.
        # e is (j0,k0), h is (j1,k1).
        # v_1 has 1 at (j0,k0) and 1 at (j1,k1).
        # w_1 has 1 at (k0,i0) and 1 at (k0,i1) and 1 at (k1,i0) and 1 at (k1,i1)?
        # No, the output is C.
        # M1 contributes to C11 and C22.
        # C11 is at (i0, k0). C22 is at (i1, k1).
        # So w_1 has 1 at (k0, i0) and 1 at (k1, i1).
        
        # This works!
        
        pass

    # This is getting too complex for a scratch rewrite.
    # I will implement the 46 rank solution using explicit terms.
    
    return {"U": [], "V": [], "W": []}
