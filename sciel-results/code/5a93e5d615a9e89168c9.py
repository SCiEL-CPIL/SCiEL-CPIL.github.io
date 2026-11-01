"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, values=1):
        nonlocal col
        if col >= rank:
            return
        for i in range(len(u_indices)):
            U[u_indices[i]][col] += values[i] if isinstance(values, list) else values
        for i in range(len(v_indices)):
            V[v_indices[i]][col] += values[i] if isinstance(values, list) else values
        for i in range(len(w_indices)):
            W[w_indices[i]][col] += values[i] if isinstance(values, list) else values
        col += 1

    # Function to add Strassen terms for 2x2x2 block
    # i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}
    def add_strassen(j0, k0):
        # U indices for i=0,1 and j=j0, j0+1
        # U[i*m + j]
        u00 = 0*m + j0
        u01 = 0*m + (j0+1)
        u10 = 1*m + j0
        u11 = 1*m + (j0+1)
        
        # V indices for j=j0, j0+1 and k=k0, k0+1
        # V[j*p + k]
        v00 = j0*p + k0
        v01 = j0*p + (k0+1)
        v10 = (j0+1)*p + k0
        v11 = (j0+1)*p + (k0+1)
        
        # W indices for k=k0, k0+1 and i=0,1
        # W[k*n + i]
        w00 = k0*n + 0
        w01 = k0*n + 1
        w10 = (k0+1)*n + 0
        w11 = (k0+1)*n + 1
        
        # P1 = (A00 + A11)(B00 + B11) -> C00 += P1, C11 += P1
        add_term([u00, u11], [v00, v11], [w00, w11])
        
        # P2 = (A10 + A11)B00 -> C10 += P2, C11 -= P2
        add_term([u10, u11], [v00], [w01, w11], [1, -1])
        
        # P3 = A00(B01 - B11) -> C01 += P3, C11 += P3
        add_term([u00], [v01, v11], [w10, w11], [1, 1])
        
        # P4 = A11(B10 - B00) -> C00 += P4, C10 += P4
        add_term([u11], [v10, v00], [w00, w01], [1, 1])
        
        # P5 = (A00 + A01)B11 -> C00 -= P5, C01 += P5
        add_term([u00, u01], [v11], [w00, w10], [-1, 1])
        
        # P6 = (A10 - A00)(B00 + B01) -> C11 += P6
        add_term([u10, u00], [v00, v01], [w11], [1, -1])
        
        # P7 = (A01 - A11)(B10 + B11) -> C00 += P7
        add_term([u01, u11], [v10, v11], [w00], [1, -1])

    # Function to add schoolbook terms for 2x2x1 block
    # i in {0,1}, j in {j0, j0+1}, k=k0
    def add_schoolbook_221(j0, k0):
        u00 = 0*m + j0
        u01 = 0*m + (j0+1)
        u10 = 1*m + j0
        u11 = 1*m + (j0+1)
        
        v00 = j0*p + k0
        v10 = (j0+1)*p + k0
        
        w00 = k0*n + 0
        w01 = k0*n + 1
        
        # A00 B00 -> C00
        add_term([u00], [v00], [w00])
        # A01 B10 -> C00
        add_term([u01], [v10], [w00])
        # A10 B00 -> C10
        add_term([u10], [v00], [w01])
        # A11 B10 -> C10
        add_term([u11], [v10], [w01])

    # Function to add rank-2 terms for 2x1x3 block
    # i in {0,1}, j=j0, k in {k0, k0+1, k0+2}
    def add_rank2_213(j0, k0):
        u0 = 0*m + j0
        u1 = 1*m + j0
        
        v0 = j0*p + k0
        v1 = j0*p + (k0+1)
        v2 = j0*p + (k0+2)
        
        w00 = k0*n + 0
        w01 = (k0+1)*n + 0
        w02 = (k0+2)*n + 0
        
        w10 = k0*n + 1
        w11 = (k0+1)*n + 1
        w12 = (k0+2)*n + 1
        
        # A0j (Bj0, Bj1, Bj2) -> C00, C01, C02
        add_term([u0], [v0, v1, v2], [w00, w01, w02])
        
        # A1j (Bj0, Bj1, Bj2) -> C10, C11, C12
        add_term([u1], [v0, v1, v2], [w10, w11, w12])

    # Block 1: k in {0, 1, 2}
    add_strassen(0, 0)       # j in {0,1}, k in {0,1}
    add_schoolbook_221(0, 2) # j in {0,1}, k=2
    add_rank2_213(2, 0)      # j=2, k in {0,1,2}
    
    # Block 2: k in {3, 4, 5}
    add_strassen(0, 3)       # j in {0,1}, k in {3,4}
    add_schoolbook_221(0, 5) # j in {0,1}, k=5
    add_rank2_213(2, 3)      # j=2, k in {3,4,5}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
