import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def get_u_idx(i, j):
        return i * m + j

    def get_v_idx(j, k):
        return j * p + k

    def get_w_idx(k, i):
        return k * n + i

    def add_strassen_2x2(ai, aj, bj, bk, scale_u=1, scale_v=1, scale_w=1):
        # Adds 7 terms for the 2x2x2 block multiplication
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = scale_u; u[get_u_idx(ai+1, aj+1)] = scale_u
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = scale_v; v[get_v_idx(bj+1, bk+1)] = scale_v
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = scale_w; w[get_w_idx(bk+1, ai+1)] = scale_w
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = scale_u; u[get_u_idx(ai+1, aj+1)] = scale_u
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = scale_v
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = scale_w; w[get_w_idx(bk+1, ai+1)] = -scale_w
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = scale_u
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = scale_v; v[get_v_idx(bj+1, bk+1)] = -scale_v
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = scale_w; w[get_w_idx(bk+1, ai+1)] = scale_w
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = scale_u
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = scale_v; v[get_v_idx(bj, bk)] = -scale_v
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = scale_w; w[get_w_idx(bk, ai+1)] = scale_w
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = scale_u; u[get_u_idx(ai, aj+1)] = scale_u
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = scale_v
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -scale_w; w[get_w_idx(bk+1, ai)] = scale_w
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = scale_u; u[get_u_idx(ai, aj)] = -scale_u
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = scale_v; v[get_v_idx(bj, bk+1)] = scale_v
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = scale_w
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = scale_u; u[get_u_idx(ai+1, aj+1)] = -scale_u
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = scale_v; v[get_v_idx(bj+1, bk+1)] = scale_v
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = scale_w
        add_term(u, v, w)

    # Construct the 4x4x4 part using two levels of Strassen
    # This will generate 49 terms
    
    # Level 1: Apply Strassen to the 2x2 blocks of A and B
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to expand each of these into 7 rank-1 terms for the 2x2x2 multiplication.
    # This is equivalent to calling add_strassen_2x2 for each of the 7 "virtual" 2x2x2 multiplications.
    # But the "virtual" multiplications involve sums/differences of the A and B blocks.
    # So we need to propagate the coefficients.
    
    # Let's define the 7 Strassen steps for the 4x4x4 case.
    # Step 1: Compute M1 = (A00+A11)(B00+B11)
    # This is a 2x2x2 multiplication where the A-block is A00+A11 and B-block is B00+B11.
    # We can decompose this using add_strassen_2x2, but we need to handle the sums.
    # Actually, it's easier to just use the known 49-term decomposition directly or generate it via recursion.
    
    # Recursive generation of Strassen terms
    def generate_strassen_terms(ai, aj, bj, bk, scale_u, scale_v, scale_w):
        # This function generates the 7 terms for the 2x2x2 block at (ai,aj) in A and (bj,bk) in B
        # with the given scales.
        # But for the 4x4x4 case, we need to call this for the 7 "super-blocks".
        # Let's implement the 7 super-blocks explicitly.
        
        pass # We will implement the 49 terms directly below.

    # Direct implementation of the 49 terms for 4x4x4
    # A is 4x4, B is 4x4.
    # Indices:
    # A00: (0,0)-(1,1), A01: (0,2)-(1,3)
    # A10: (2,0)-(3,1), A11: (2,2)-(3,3)
    # B00: (0,0)-(1,1), B01: (0,2)-(1,3)
    # B10: (2,0)-(3,1), B11: (2,2)-(3,3)
    
    # M1 = (A00+A11)(B00+B11)
    # This is a 2x2x2 multiplication.
    # We can call add_strassen_2x2, but we need to add the contributions from A00 and A11, and B00 and B11.
    # The linearity of the decomposition means we can sum the U, V, W vectors.
    # So for M1, we have 4 sub-terms:
    # (A00)(B00), (A00)(B11), (A11)(B00), (A11)(B11)
    # Each of these is a 2x2x2 multiplication.
    # So we call add_strassen_2x2 for each of these 4 pairs, and sum the results.
    # But add_strassen_2x2 adds 7 terms. So 4*7 = 28 terms for M1?
    # No, that's not right. The Strassen decomposition is for the product of two 2x2 matrices.
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # The Strassen decomposition computes the product of (A00+A11) and (B00+B11) in 7 steps.
    # It does NOT decompose into 4 separate 2x2x2 multiplications.
    # It uses linear combinations.
    # So we need to generate the 7 terms for the product of (A00+A11) and (B00+B11).
    # The U vector for the first term of this product will have non-zero entries at the positions of A00 and A11.
    # Specifically, the first term of Strassen is (a+d)(e+h).
    # Here 'a' is the top-left of (A00+A11), which is A00[0,0] + A11[0,0].
    # So the U vector will have 1 at get_u_idx(0,0) and 1 at get_u_idx(2,2).
    # Similarly for the other entries.
    
    # Let's write a helper to add a Strassen term for a "virtual" 2x2x2 multiplication
    # where the A-block is a sum of two 2x2 blocks and the B-block is a sum of two 2x2 blocks.
    
    def add_virtual_strassen(ai1, aj1, ai2, aj2, bj1, bk1, bj2, bk2, scale_u, scale_v, scale_w):
        # This adds 7 terms for the product of (A_block1 + A_block2) and (B_block1 + B_block2)
        # A_block1 is at (ai1, aj1), A_block2 is at (ai2, aj2)
        # B_block1 is at (bj1, bk1), B_block2 is at (bj2, bk2)
        
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1)] = scale_u
        u[get_u_idx(ai1+1, aj1+1)] = scale_u
        u[get_u_idx(ai2, aj2)] = scale_u
        u[get_u_idx(ai2+1, aj2+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1)] = scale_v
        v[get_v_idx(bj1+1, bk1+1)] = scale_v
        v[get_v_idx(bj2, bk2)] = scale_v
        v[get_v_idx(bj2+1, bk2+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = scale_w
        w[get_w_idx(bk1+1, ai1+1)] = scale_w
        w[get_w_idx(bk2, ai2)] = scale_w
        w[get_w_idx(bk2+1, ai2+1)] = scale_w
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai1+1, aj1)] = scale_u
        u[get_u_idx(ai1+1, aj1+1)] = scale_u
        u[get_u_idx(ai2+1, aj2)] = scale_u
        u[get_u_idx(ai2+1, aj2+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1+1)] = scale_w
        w[get_w_idx(bk1+1, ai1+1)] = -scale_w
        w[get_w_idx(bk2, ai2+1)] = scale_w
        w[get_w_idx(bk2+1, ai2+1)] = -scale_w
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1+1)] = scale_v
        v[get_v_idx(bj1+1, bk1+1)] = -scale_v
        v[get_v_idx(bj2, bk2+1)] = scale_v
        v[get_v_idx(bj2+1, bk2+1)] = -scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1+1, ai1)] = scale_w
        w[get_w_idx(bk1+1, ai1+1)] = scale_w
        w[get_w_idx(bk2+1, ai2)] = scale_w
        w[get_w_idx(bk2+1, ai2+1)] = scale_w
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai1+1, aj1+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1+1, bk1)] = scale_v
        v[get_v_idx(bj1, bk1)] = -scale_v
        v[get_v_idx(bj2+1, bk2)] = scale_v
        v[get_v_idx(bj2, bk2)] = -scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = scale_w
        w[get_w_idx(bk1, ai1+1)] = scale_w
        w[get_w_idx(bk2, ai2)] = scale_w
        w[get_w_idx(bk2, ai2+1)] = scale_w
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1)] = scale_u
        u[get_u_idx(ai1, aj1+1)] = scale_u
        u[get_u_idx(ai2, aj2)] = scale_u
        u[get_u_idx(ai2, aj2+1)] = scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1+1, bk1+1)] = scale_v
        v[get_v_idx(bj2+1, bk2+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = -scale_w
        w[get_w_idx(bk1+1, ai1)] = scale_w
        w[get_w_idx(bk2, ai2)] = -scale_w
        w[get_w_idx(bk2+1, ai2)] = scale_w
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai1+1, aj1)] = scale_u
        u[get_u_idx(ai1, aj1)] = -scale_u
        u[get_u_idx(ai2+1, aj2)] = scale_u
        u[get_u_idx(ai2, aj2)] = -scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1)] = scale_v
        v[get_v_idx(bj1, bk1+1)] = scale_v
        v[get_v_idx(bj2, bk2)] = scale_v
        v[get_v_idx(bj2, bk2+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1+1, ai1+1)] = scale_w
        w[get_w_idx(bk2+1, ai2+1)] = scale_w
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1+1)] = scale_u
        u[get_u_idx(ai1+1, aj1+1)] = -scale_u
        u[get_u_idx(ai2, aj2+1)] = scale_u
        u[get_u_idx(ai2+1, aj2+1)] = -scale_u
        v = np.zeros(m*p)
        v[get_v_idx(bj1+1, bk1)] = scale_v
        v[get_v_idx(bj1+1, bk1+1)] = scale_v
        v[get_v_idx(bj2+1, bk2)] = scale_v
        v[get_v_idx(bj2+1, bk2+1)] = scale_v
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = scale_w
        w[get_w_idx(bk2, ai2)] = scale_w
        add_term(u, v, w)

    # Now we can generate the 49 terms for the 4x4x4 part.
    # M1 = (A00+A11)(B00+B11)
    add_virtual_strassen(0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1)
    # M2 = (A10+A11)B00
    add_virtual_strassen(2, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1) # Wait, B00 is a single block.
    # The function add_virtual_strassen assumes two B-blocks.
    # For M2, it's (A10+A11) * B00.
    # This is (A10+A11) * (B00 + 0).
    # So we can call add_virtual_strassen(2, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1)
    # But the second B-block is zero. The function will add zeros for the second block.
    # This is correct.
    
    # M3 = A00(B10-B00)
    # This is A00 * (B10 - B00).
    # We can call add_virtual_strassen(0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1)
    # But we need to handle the minus sign.
    # The function adds terms for (A1+A2)(B1+B2).
    # Here A1=A00, A2=0. B1=B10, B2=-B00.
    # So we need to pass scale_v = -1 for the second B-block?
    # No, the function uses the same scale for both blocks.
    # We need a more flexible function.
    
    # Let's rewrite add_virtual_strassen to allow different scales for the two A-blocks and two B-blocks.
    
    def add_flexible_strassen(ai1, aj1, ai2, aj2, bj1, bk1, bj2, bk2, su1, su2, sv1, sv2, sw):
        # Adds 7 terms for (su1*A1 + su2*A2) * (sv1*B1 + sv2*B2)
        
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1)] = su1
        u[get_u_idx(ai1+1, aj1+1)] = su1
        u[get_u_idx(ai2, aj2)] = su2
        u[get_u_idx(ai2+1, aj2+1)] = su2
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1)] = sv1
        v[get_v_idx(bj1+1, bk1+1)] = sv1
        v[get_v_idx(bj2, bk2)] = sv2
        v[get_v_idx(bj2+1, bk2+1)] = sv2
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = sw
        w[get_w_idx(bk1+1, ai1+1)] = sw
        w[get_w_idx(bk2, ai2)] = sw
        w[get_w_idx(bk2+1, ai2+1)] = sw
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m)
        u[get_u_idx(ai1+1, aj1)] = su1
        u[get_u_idx(ai1+1, aj1+1)] = su1
        u[get_u_idx(ai2+1, aj2)] = su2
        u[get_u_idx(ai2+1, aj2+1)] = su2
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1)] = sv1
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1+1)] = sw
        w[get_w_idx(bk1+1, ai1+1)] = -sw
        w[get_w_idx(bk2, ai2+1)] = sw
        w[get_w_idx(bk2+1, ai2+1)] = -sw
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1)] = su1
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1+1)] = sv1
        v[get_v_idx(bj1+1, bk1+1)] = -sv1
        v[get_v_idx(bj2, bk2+1)] = sv2
        v[get_v_idx(bj2+1, bk2+1)] = -sv2
        w = np.zeros(p*n)
        w[get_w_idx(bk1+1, ai1)] = sw
        w[get_w_idx(bk1+1, ai1+1)] = sw
        w[get_w_idx(bk2+1, ai2)] = sw
        w[get_w_idx(bk2+1, ai2+1)] = sw
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m)
        u[get_u_idx(ai1+1, aj1+1)] = su1
        v = np.zeros(m*p)
        v[get_v_idx(bj1+1, bk1)] = sv1
        v[get_v_idx(bj1, bk1)] = -sv1
        v[get_v_idx(bj2+1, bk2)] = sv2
        v[get_v_idx(bj2, bk2)] = -sv2
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = sw
        w[get_w_idx(bk1, ai1+1)] = sw
        w[get_w_idx(bk2, ai2)] = sw
        w[get_w_idx(bk2, ai2+1)] = sw
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1)] = su1
        u[get_u_idx(ai1, aj1+1)] = su1
        u[get_u_idx(ai2, aj2)] = su2
        u[get_u_idx(ai2, aj2+1)] = su2
        v = np.zeros(m*p)
        v[get_v_idx(bj1+1, bk1+1)] = sv1
        v[get_v_idx(bj2+1, bk2+1)] = sv2
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = -sw
        w[get_w_idx(bk1+1, ai1)] = sw
        w[get_w_idx(bk2, ai2)] = -sw
        w[get_w_idx(bk2+1, ai2)] = sw
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m)
        u[get_u_idx(ai1+1, aj1)] = su1
        u[get_u_idx(ai1, aj1)] = -su1
        u[get_u_idx(ai2+1, aj2)] = su2
        u[get_u_idx(ai2, aj2)] = -su2
        v = np.zeros(m*p)
        v[get_v_idx(bj1, bk1)] = sv1
        v[get_v_idx(bj1, bk1+1)] = sv1
        v[get_v_idx(bj2, bk2)] = sv2
        v[get_v_idx(bj2, bk2+1)] = sv2
        w = np.zeros(p*n)
        w[get_w_idx(bk1+1, ai1+1)] = sw
        w[get_w_idx(bk2+1, ai2+1)] = sw
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m)
        u[get_u_idx(ai1, aj1+1)] = su1
        u[get_u_idx(ai1+1, aj1+1)] = -su1
        u[get_u_idx(ai2, aj2+1)] = su2
        u[get_u_idx(ai2+1, aj2+1)] = -su2
        v = np.zeros(m*p)
        v[get_v_idx(bj1+1, bk1)] = sv1
        v[get_v_idx(bj1+1, bk1+1)] = sv1
        v[get_v_idx(bj2+1, bk2)] = sv2
        v[get_v_idx(bj2+1, bk2+1)] = sv2
        w = np.zeros(p*n)
        w[get_w_idx(bk1, ai1)] = sw
        w[get_w_idx(bk2, ai2)] = sw
        add_term(u, v, w)

    # M1 = (A00+A11)(B00+B11)
    add_flexible_strassen(0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1)
    # M2 = (A10+A11)B00
    add_flexible_strassen(2, 0, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 1)
    # M3 = A00(B10-B00)
    add_flexible_strassen(0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1, -1, 1)
    # M4 = A11(B01-B10)
    add_flexible_strassen(2, 2, 2, 2, 0, 2, 2, 0, 1, 0, 1, -1, 1)
    # M5 = (A00+A01)B11
    add_flexible_strassen(0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1)
    # M6 = (A10-A00)(B00+B01)
    add_flexible_strassen(2, 0, 0, 0, 0, 0, 0, 2, 1, -1, 1, 1, 1)
    # M7 = (A01-A11)(B10+B11)
    add_flexible_strassen(0, 2, 2, 2, 2, 0, 2, 2, 1, -1, 1, 1, 1)

    # Now add the 16 terms for the 5th column (col 4)
    # C[:,4] = A @ B[:,4]
    for i in range(4):
        for j in range(4):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
