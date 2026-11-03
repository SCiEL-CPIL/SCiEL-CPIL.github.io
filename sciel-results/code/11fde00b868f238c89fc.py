import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # List of (i_start, j_start, k_start) for 2x2x2 blocks
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    blocks = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                blocks.append((i0, j0, k0))
                
    # Strassen coefficients for 2x2x2
    # Local indices 0,1
    # U_local (2x2), V_local (2x2), W_local (2x2)
    # Term 1: (A00+A11)(B00+B11)
    # A part: e0 + e1, B part: e0 + e1
    # C part: C00 + C11? No, C00 and C11 get +1, C01 and C10 get 0?
    # Let's use the explicit formulas.
    
    # We will build U, V, W as lists of columns.
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen terms for a block at (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        # Local 0 -> global i0, 1 -> global i0+1
        # Local 0 -> global j0, 1 -> global j0+1
        # Local 0 -> global k0, 1 -> global k0+1
        
        # Helper to create vectors
        def make_u(i_local, val):
            vec = [0.0] * (n * m)
            vec[(i0 + i_local) * m + (j0 + 0)] = val # Wait, j is also local?
            # No, U is n*m. Index is i*m + j.
            # The term is a product of a row of A and a row of B?
            # No, U is (n*m, rank). Each column is a vector in R^{n*m}.
            # The vector has non-zeros at positions corresponding to (i,j) pairs.
            # For a Strassen term, the "A part" is a linear combination of A_ij.
            # So the U vector has non-zeros at the (i,j) indices involved.
            
            # Let's define the 7 terms explicitly for local indices.
            # Term 1: (A00+A11)(B00+B11)
            # U: A00=1, A11=1. Others 0.
            # V: B00=1, B11=1. Others 0.
            # W: C00=1, C11=1. Others 0. (Since C00 += M1, C11 += M1)
            
            # Let's list the (u, v, w) coefficients for the 7 terms.
            # u is in R^4 (2x2), v in R^4 (2x2), w in R^4 (2x2).
            
            # T1
            u1 = [1, 0, 0, 1] # A00, A01, A10, A11
            v1 = [1, 0, 0, 1] # B00, B01, B10, B11
            w1 = [1, 0, 0, 1] # C00, C01, C10, C11 (W index k*n+i)
            # Wait, W index is k*n + i.
            # Local k=0, i=0 -> 0*n + 0 = 0.
            # Local k=0, i=1 -> 0*n + 1 = 1.
            # Local k=1, i=0 -> 1*n + 0 = n.
            # Local k=1, i=1 -> 1*n + 1 = n+1.
            # So W local indices:
            # C00 (k=0, i=0) -> 0
            # C10 (k=0, i=1) -> 1  (Note: C is n x p, so C[i,j]. Here output is C^T[k,i]?)
            # Let's stick to the global index mapping.
            
            # Global mapping:
            # U index: (i0+li) * m + (j0+lj)
            # V index: (j0+lj) * p + (k0+lk)
            # W index: (k0+lk) * n + (i0+li)
            
            # Term 1:
            # U: li=0, lj=0 (1); li=1, lj=1 (1)
            # V: lj=0, lk=0 (1); lj=1, lk=1 (1)
            # W: lk=0, li=0 (1); lk=1, li=1 (1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+0)*m + (j0+0)] = 1
            u_vec[(i0+1)*m + (j0+1)] = 1
            
            v_vec[(j0+0)*p + (k0+0)] = 1
            v_vec[(j0+1)*p + (k0+1)] = 1
            
            w_vec[(k0+0)*n + (i0+0)] = 1
            w_vec[(k0+1)*n + (i0+1)] = 1
            
            add_term(u_vec, v_vec, w_vec)
            
            # Term 2: (A10+A11)B00
            # U: li=1, lj=0 (1); li=1, lj=1 (1)
            # V: lj=0, lk=0 (1)
            # W: C10 += M2 -> lk=0, li=1 (1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+1)*m + (j0+0)] = 1
            u_vec[(i0+1)*m + (j0+1)] = 1
            
            v_vec[(j0+0)*p + (k0+0)] = 1
            
            w_vec[(k0+0)*n + (i0+1)] = 1
            
            add_term(u_vec, v_vec, w_vec)
            
            # Term 3: A00(B10-B11)
            # U: li=0, lj=0 (1)
            # V: lj=1, lk=0 (1); lj=1, lk=1 (-1)
            # W: C01 += M3 -> lk=1, li=0 (1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+0)*m + (j0+0)] = 1
            
            v_vec[(j0+1)*p + (k0+0)] = 1
            v_vec[(j0+1)*p + (k0+1)] = -1
            
            w_vec[(k0+1)*n + (i0+0)] = 1
            
            add_term(u_vec, v_vec, w_vec)
            
            # Term 4: A11(B01-B00)
            # U: li=1, lj=1 (1)
            # V: lj=0, lk=1 (1); lj=0, lk=0 (-1)
            # W: C00 += M4, C10 += M4 -> lk=0, li=0 (1); lk=0, li=1 (1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+1)*m + (j0+1)] = 1
            
            v_vec[(j0+0)*p + (k0+1)] = 1
            v_vec[(j0+0)*p + (k0+0)] = -1
            
            w_vec[(k0+0)*n + (i0+0)] = 1
            w_vec[(k0+0)*n + (i0+1)] = 1
            
            add_term(u_vec, v_vec, w_vec)
            
            # Term 5: (A00+A01)B11
            # U: li=0, lj=0 (1); li=0, lj=1 (1)
            # V: lj=1, lk=1 (1)
            # W: C00 -= M5, C01 += M5 -> lk=0, li=0 (-1); lk=1, li=0 (1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+0)*m + (j0+0)] = 1
            u_vec[(i0+0)*m + (j0+1)] = 1
            
            v_vec[(j0+1)*p + (k0+1)] = 1
            
            w_vec[(k0+0)*n + (i0+0)] = -1
            w_vec[(k0+1)*n + (i0+0)] = 1
            
            add_term(u_vec, v_vec, w_vec)
            
            # Term 6: (A10-A00)(B00+B01)
            # U: li=1, lj=0 (1); li=0, lj=0 (-1)
            # V: lj=0, lk=0 (1); lj=0, lk=1 (1)
            # W: C11 += M6 -> lk=1, li=1 (1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+1)*m + (j0+0)] = 1
            u_vec[(i0+0)*m + (j0+0)] = -1
            
            v_vec[(j0+0)*p + (k0+0)] = 1
            v_vec[(j0+0)*p + (k0+1)] = 1
            
            w_vec[(k0+1)*n + (i0+1)] = 1
            
            add_term(u_vec, v_vec, w_vec)
            
            # Term 7: (A01-A11)(B10+B11)
            # U: li=0, lj=1 (1); li=1, lj=1 (-1)
            # V: lj=1, lk=0 (1); lj=1, lk=1 (1)
            # W: C00 += M7, C11 -= M7 -> lk=0, li=0 (1); lk=1, li=1 (-1)
            
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            u_vec[(i0+0)*m + (j0+1)] = 1
            u_vec[(i0+1)*m + (j0+1)] = -1
            
            v_vec[(j0+1)*p + (k0+0)] = 1
            v_vec[(j0+1)*p + (k0+1)] = 1
            
            w_vec[(k0+0)*n + (i0+0)] = 1
            w_vec[(k0+1)*n + (i0+1)] = -1
            
            add_term(u_vec, v_vec, w_vec)
            
    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0)
        
    # Add Schoolbook terms for the rest
    # We need to identify which (i,j,k) terms are NOT covered by the Strassen blocks.
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # And specifically, the terms in the blocks are handled.
    # The Strassen blocks handle ALL terms for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5} that fall into the 2x2x2 grid.
    # Wait, the blocks are:
    # (0,0,0), (0,0,2), (0,0,4)
    # (0,2,0), (0,2,2), (0,2,4)
    # These cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    # Yes, every combination of i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5} is in exactly one block.
    # So we only need schoolbook terms for:
    # i=2, OR j=4, OR k=6.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if (i,j,k) is in a Strassen block
                covered = False
                for i0, j0, k0 in blocks:
                    if i0 <= i < i0+2 and j0 <= j < j0+2 and k0 <= k < k0+2:
                        covered = True
                        break
                if not covered:
                    u_vec = [0.0] * (n * m)
                    v_vec = [0.0] * (m * p)
                    w_vec = [0.0] * (p * n)
                    
                    u_vec[i * m + j] = 1
                    v_vec[j * p + k] = 1
                    w_vec[k * n + i] = 1
                    
                    add_term(u_vec, v_vec, w_vec)
                    
    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
