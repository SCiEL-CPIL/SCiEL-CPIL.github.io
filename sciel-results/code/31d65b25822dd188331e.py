import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor verification
    # T[x, y, z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)

    # Initialize zero vectors
    def zero_u():
        return [0.0] * nm
    def zero_v():
        return [0.0] * mp
    def zero_w():
        return [0.0] * pn

    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W.
    # For a specific 2x2x2 block, we define local indices.
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen Terms Definition
    # Each term is (U_dict, V_dict, W_dict)
    # Keys are (i,j) for U, (j,k) for V, (k,i) for W
    
    strassen_terms = [
        # M1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,0): 1}
        ),
        # M3
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # M4
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,1): -1, (1,0): 1}
        ),
        # M5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1, (0,1): 1}
        ),
        # M6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): -1, (0,1): -1}
        )
    ]
    
    # We have two 2x2x2 blocks for the A_1 (2x2) part.
    # Block 1: A cols 0,1. B rows 0,1. C cols 0,1.
    # Block 2: A cols 0,1. B rows 0,1. C cols 2,3.
    
    # Note: B is 3x4.
    # B rows 0,1 are used for A cols 0,1.
    # B row 2 is used for A col 2.
    
    # Block 1 Mapping:
    # A_local (i, j) -> A_global (i, j)
    # B_local (j, k) -> B_global (j, k) for k in 0,1
    # C_local (k, i) -> C_global (k, i) for k in 0,1
    
    for u_dict, v_dict, w_dict in strassen_terms:
        u_vec = zero_u()
        v_vec = zero_v()
        w_vec = zero_w()
        
        for (i, j), val in u_dict.items():
            # A index
            idx = i * m + j
            u_vec[idx] = val
            
        for (j, k), val in v_dict.items():
            # B index
            # k is local col 0 or 1
            idx = j * p + k
            v_vec[idx] = val
            
        for (k, i), val in w_dict.items():
            # C index
            # k is local col 0 or 1
            idx = k * n + i
            w_vec[idx] = val
            
        add_term(u_vec, v_vec, w_vec)
        
    # Block 2 Mapping:
    # A_local (i, j) -> A_global (i, j)
    # B_local (j, k) -> B_global (j, k+2) for k in 0,1
    # C_local (k, i) -> C_global (k+2, i) for k in 0,1
    
    for u_dict, v_dict, w_dict in strassen_terms:
        u_vec = zero_u()
        v_vec = zero_v()
        w_vec = zero_w()
        
        for (i, j), val in u_dict.items():
            idx = i * m + j
            u_vec[idx] = val
            
        for (j, k), val in v_dict.items():
            idx = j * p + (k + 2)
            v_vec[idx] = val
            
        for (k, i), val in w_dict.items():
            idx = (k + 2) * n + i
            w_vec[idx] = val
            
        add_term(u_vec, v_vec, w_vec)
        
    # Part 2: A col 2 (2x1) and B row 2 (1x4)
    # A index (i, 2), B index (2, k), C index (k, i)
    
    for i in range(n):
        for k in range(p):
            u_vec = zero_u()
            v_vec = zero_v()
            w_vec = zero_w()
            
            u_vec[i * m + 2] = 1.0
            v_vec[2 * p + k] = 1.0
            w_vec[k * n + i] = 1.0
            
            add_term(u_vec, v_vec, w_vec)
            
    return {"U": U, "V": V, "W": W}
