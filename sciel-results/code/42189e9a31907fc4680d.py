"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrices partitioned into 3x3 blocks of 2x2 matrices
    block_size = 2
    num_blocks = n // block_size # 3
    
    # Strassen's 7 products for 2x2 matrices
    # u_s: linear form in A (2x2)
    # v_s: linear form in B (2x2)
    # w_s: linear form in C (2x2) coefficients
    # C = sum_s alpha_s * M_s
    # M_s = u_s(A) * v_s(B)
    
    # Indices for 2x2: 0,1
    # A: [[a,b],[c,d]]
    # B: [[e,f],[g,h]]
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (c+d)e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 0, (0,1): 0, (1,1): 0}, {(1,0): 1, (1,1): 0}),
        # M3 = a(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 0}),
        # M4 = d(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1, (1,1): -1}),
        # M5 = (a+b)h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (c-a)(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (b-d)(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]
    
    rank = num_blocks * num_blocks * num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in range(num_blocks):
        for jb in range(num_blocks):
            for kb in range(num_blocks):
                for s in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                    
                    # Fill U
                    for (il, jl), val in u_coeffs.items():
                        i = ib * block_size + il
                        j = jb * block_size + jl
                        U[i * m + j][col] = val
                    
                    # Fill V
                    for (jl, kl), val in v_coeffs.items():
                        j = jb * block_size + jl
                        k = kb * block_size + kl
                        V[j * p + k][col] = val
                    
                    # Fill W
                    # Note: W index is k*n + i
                    for (il, kl), val in w_coeffs.items():
                        i = ib * block_size + il
                        k = kb * block_size + kl
                        W[k * n + i][col] = val
                    
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
