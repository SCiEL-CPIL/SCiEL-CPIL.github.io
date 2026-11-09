import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set a term
    def add_term(c, u_vals, v_vals, w_vals):
        for u_idx, val in u_vals.items():
            U[u_idx, c] = val
        for v_idx, val in v_vals.items():
            V[v_idx, c] = val
        for w_idx, val in w_vals.items():
            W[w_idx, c] = val

    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # i in 0,1; j in 0,1; k in 0,1
    
    # M1 = (a00+a11)(b00+b11)
    add_term(0, 
             {0: 1, 4: 1}, 
             {0: 1, 4: 1}, 
             {0: 1, 1: 1, 2: 1, 3: 1})
             
    # M2 = (a10+a11)b00
    add_term(1, 
             {3: 1, 4: 1}, 
             {0: 1}, 
             {1: 1, 3: -1})
             
    # M3 = a00(b01-b11)
    add_term(2, 
             {0: 1}, 
             {1: 1, 4: -1}, 
             {2: 1, 3: 1})
             
    # M4 = a11(b10-b00)
    add_term(3, 
             {4: 1}, 
             {3: 1, 0: -1}, 
             {0: 1, 1: 1})
             
    # M5 = (a00+a01)b11
    add_term(4, 
             {0: 1, 1: 1}, 
             {4: 1}, 
             {0: -1, 2: 1})
             
    # M6 = (a10-a00)(b00+b10)
    add_term(5, 
             {3: 1, 0: -1}, 
             {0: 1, 3: 1}, 
             {3: 1})
             
    # M7 = (a01-a11)(b11+b01)
    add_term(6, 
             {1: 1, 4: -1}, 
             {4: 1, 1: 1}, 
             {0: 1})

    # Part 2: A[0:2, 0:2] * B[0:2, 2]
    # i in 0,1; j in 0,1; k=2
    
    # C02 += a00*b02
    add_term(7, {0: 1}, {2: 1}, {4: 1})
    # C02 += a01*b12
    add_term(8, {1: 1}, {5: 1}, {4: 1})
    # C12 += a10*b02
    add_term(9, {3: 1}, {2: 1}, {5: 1})
    # C12 += a11*b12
    add_term(10, {4: 1}, {5: 1}, {5: 1})

    # Part 3: A[0:2, 2] * B[2, 0:3]
    # i in 0,1; j=2; k in 0,1,2
    
    # C00 += a02*b20
    add_term(11, {2: 1}, {6: 1}, {0: 1})
    # C01 += a02*b21
    add_term(12, {2: 1}, {7: 1}, {2: 1})
    # C02 += a02*b22
    add_term(13, {2: 1}, {8: 1}, {4: 1})
    # C10 += a12*b20
    add_term(14, {5: 1}, {6: 1}, {1: 1})
    # C11 += a12*b21
    add_term(15, {5: 1}, {7: 1}, {3: 1})
    # C12 += a12*b22
    add_term(16, {5: 1}, {8: 1}, {5: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
