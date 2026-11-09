"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 60
    # Block 1 (2x2x7): 7 (2x2x2) + 7 (2x2x2) + 11 (2x2x3) = 25
    # Block 2 (2x1x7): 14
    # Block 3 (1x2x7): 14
    # Block 4 (1x1x7): 7
    # Total: 25 + 14 + 14 + 7 = 60
    
    total_rank = 60
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # i_off, j_off: top-left index of A block
    # k_off: start index of B columns (which are also C columns)
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal rank_offset
        # Local indices 0,1 map to global:
        # A: (i_off+0, j_off+0), (i_off+0, j_off+1), (i_off+1, j_off+0), (i_off+1, j_off+1)
        # B: (j_off+0, k_off+0), (j_off+0, k_off+1), (j_off+1, k_off+0), (j_off+1, k_off+1)
        # C: (k_off+0, i_off+0), (k_off+0, i_off+1), (k_off+1, i_off+0), (k_off+1, i_off+1)
        
        # U indices
        u00 = (i_off) * m + (j_off)
        u01 = (i_off) * m + (j_off + 1)
        u10 = (i_off + 1) * m + (j_off)
        u11 = (i_off + 1) * m + (j_off + 1)
        
        # V indices
        v00 = (j_off) * p + (k_off)
        v01 = (j_off) * p + (k_off + 1)
        v10 = (j_off + 1) * p + (k_off)
        v11 = (j_off + 1) * p + (k_off + 1)
        
        # W indices
        w00 = (k_off) * n + (i_off)
        w01 = (k_off) * n + (i_off + 1)
        w10 = (k_off + 1) * n + (i_off)
        w11 = (k_off + 1) * n + (i_off + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u00][rank_offset] += 1
        U[u11][rank_offset] += 1
        V[v00][rank_offset] += 1
        V[v11][rank_offset] += 1
        W[w00][rank_offset] += 1
        W[w11][rank_offset] += 1
        W[w01][rank_offset] -= 1
        W[w10][rank_offset] -= 1
        rank_offset += 1
        
        # M2 = (A10 + A11)B00
        U[u10][rank_offset] += 1
        U[u11][rank_offset] += 1
        V[v00][rank_offset] += 1
        W[w10][rank_offset] += 1
        W[w11][rank_offset] -= 1
        rank_offset += 1
        
        # M3 = A00(B10 - B11)
        U[u00][rank_offset] += 1
        V[v10][rank_offset] += 1
        V[v11][rank_offset] -= 1
        W[w01][rank_offset] += 1
        W[w11][rank_offset] += 1
        rank_offset += 1
        
        # M4 = A11(B01 - B00)
        U[u11][rank_offset] += 1
        V[v01][rank_offset] += 1
        V[v00][rank_offset] -= 1
        W[w00][rank_offset] += 1
        W[w10][rank_offset] += 1
        rank_offset += 1
        
        # M5 = (A00 + A01)B11
        U[u00][rank_offset] += 1
        U[u01][rank_offset] += 1
        V[v11][rank_offset] += 1
        W[w00][rank_offset] -= 1
        W[w01][rank_offset] += 1
        rank_offset += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u10][rank_offset] += 1
        U[u00][rank_offset] -= 1
        V[v00][rank_offset] += 1
        V[v01][rank_offset] += 1
        W[w11][rank_offset] += 1
        rank_offset += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u01][rank_offset] += 1
        U[u11][rank_offset] -= 1
        V[v10][rank_offset] += 1
        V[v11][rank_offset] += 1
        W[w00][rank_offset] += 1
        rank_offset += 1

    # Helper to add Schoolbook 2x2x1 decomposition
    # For the 3rd column of a 2x2x3 block
    def add_schoolbook_221(i_off, j_off, k_off):
        nonlocal rank_offset
        u00 = i_off * m + j_off
        u01 = i_off * m + (j_off + 1)
        u10 = (i_off + 1) * m + j_off
        u11 = (i_off + 1) * m + (j_off + 1)
        
        v00 = j_off * p + k_off
        v10 = (j_off + 1) * p + k_off
        
        w00 = k_off * n + i_off
        w01 = k_off * n + (i_off + 1)
        w10 = (k_off + 1) * n + i_off
        w11 = (k_off + 1) * n + (i_off + 1)
        
        # C00 += A00 B00
        U[u00][rank_offset] += 1
        V[v00][rank_offset] += 1
        W[w00][rank_offset] += 1
        rank_offset += 1
        
        # C01 += A01 B00
        U[u01][rank_offset] += 1
        V[v00][rank_offset] += 1
        W[w01][rank_offset] += 1
        rank_offset += 1
        
        # C10 += A10 B10
        U[u10][rank_offset] += 1
        V[v10][rank_offset] += 1
        W[w10][rank_offset] += 1
        rank_offset += 1
        
        # C11 += A11 B10
        U[u11][rank_offset] += 1
        V[v10][rank_offset] += 1
        W[w11][rank_offset] += 1
        rank_offset += 1

    # Helper to add Outer Product 2x1x7 (A is 2x1, B is 1x7)
    # A indices: i_off, i_off+1; j=j_val
    # B indices: j=j_val; k=k_off to k_off+6
    # C indices: i_off, i_off+1; k=k_off to k_off+6
    def add_outer_217(i_off, j_val, k_off):
        nonlocal rank_offset
        u0 = i_off * m + j_val
        u1 = (i_off + 1) * m + j_val
        v = j_val * p
        
        for k in range(7):
            w0 = (k_off + k) * n + i_off
            w1 = (k_off + k) * n + (i_off + 1)
            
            # Term for C[i_off, k] += A[i_off, j] B[j, k]
            U[u0][rank_offset] += 1
            V[v + (k_off + k)][rank_offset] += 1
            W[w0][rank_offset] += 1
            rank_offset += 1
            
            # Term for C[i_off+1, k] += A[i_off+1, j] B[j, k]
            U[u1][rank_offset] += 1
            V[v + (k_off + k)][rank_offset] += 1
            W[w1][rank_offset] += 1
            rank_offset += 1

    # Helper to add Outer Product 1x2x7 (A is 1x2, B is 2x7)
    # A indices: i=i_val; j=j_off, j_off+1
    # B indices: j=j_off, j_off+1; k=k_off to k_off+6
    # C indices: i=i_val; k=k_off to k_off+6
    def add_outer_127(i_val, j_off, k_off):
        nonlocal rank_offset
        v0 = j_off * p
        v1 = (j_off + 1) * p
        
        for j_idx, j_val in enumerate([j_off, j_off + 1]):
            u = i_val * m + j_val
            for k in range(7):
                w = (k_off + k) * n + i_val
                
                # Term for C[i, k] += A[i, j] B[j, k]
                U[u][rank_offset] += 1
                V[(v0 if j_idx==0 else v1) + (k_off + k)][rank_offset] += 1
                W[w][rank_offset] += 1
                rank_offset += 1

    # Helper to add Outer Product 1x1x7 (A is 1x1, B is 1x7)
    def add_outer_117(i_val, j_val, k_off):
        nonlocal rank_offset
        u = i_val * m + j_val
        v = j_val * p
        
        for k in range(7):
            w = (k_off + k) * n + i_val
            U[u][rank_offset] += 1
            V[v + (k_off + k)][rank_offset] += 1
            W[w][rank_offset] += 1
            rank_offset += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # Split B columns into 2, 2, 3
    add_strassen_222(0, 0, 0) # Cols 0-1
    add_strassen_222(0, 0, 2) # Cols 2-3
    add_strassen_222(0, 0, 4) # Cols 4-5 (Strassen part of 2x2x3)
    add_schoolbook_221(0, 0, 6) # Col 6 (Schoolbook part of 2x2x3)
    
    # Block 2: A[0:2, 2] * B[2, 0:7]
    add_outer_217(0, 2, 0)
    
    # Block 3: A[2, 0:2] * B[0:2, 0:7]
    add_outer_127(2, 0, 0)
    
    # Block 4: A[2, 2] * B[2, 0:7]
    add_outer_117(2, 2, 0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
