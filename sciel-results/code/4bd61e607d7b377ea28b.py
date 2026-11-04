"""
Program for rank-minimal exact tensor decomposition of 2x4 * 4x8 matrix multiplication.

Approach:
- Split the 4 intermediate indices j into two groups of 2.
- Split the 8 output indices k into four groups of 2.
- For each (j-group, k-group) pair, we compute a 2x2 * 2x2 sub-multiplication.
- Use Strassen's rank-7 algorithm for each 2x2 * 2x2 sub-problem.
- Total rank = 2 * 4 * 7 = 56, which is less than the schoolbook 64.

The factor matrices U, V, W are built so that:
  C[i,k] = sum_j A[i,j] * B[j,k]
is exactly reproduced via the tensor decomposition.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen coefficients for 2x2 * 2x2
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    strassen_A = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    strassen_B = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    strassen_C = [
        [1, 0, 0, 1],   # C11: M1+M4-M5+M7 (M1:1, M4:1, M5:-1, M7:1) -> Wait, indices in strassen_C are for M_s
        # Let's redefine strassen_C as matrix of size 4x7
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
    ]
    
    # Correct strassen_C matrix (4 rows for C entries, 7 cols for M terms)
    strassen_C = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
    ]
    
    rank = 2 * 4 * 7  # 56
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill matrices
    for jg in range(2):
        for kg in range(4):
            base_c = (jg * 4 + kg) * 7
            for s in range(7):
                c = base_c + s
                
                # U: A[i, j] where j = jg*2 + jl
                for i in range(2):
                    for jl in range(2):
                        a_idx = i * 2 + jl
                        val = strassen_A[s][a_idx]
                        if val != 0:
                            U[i * m + jg * 2 + jl][c] += val
                
                # V: B[j, k] where j = jg*2 + jl, k = kg*2 + kl
                for jl in range(2):
                    for kl in range(2):
                        b_idx = jl * 2 + kl
                        val = strassen_B[s][b_idx]
                        if val != 0:
                            V[(jg * 2 + jl) * p + kg * 2 + kl][c] += val
                
                # W: C[i, k] where k = kg*2 + kl
                for i in range(2):
                    for kl in range(2):
                        c_idx = i * 2 + kl
                        val = strassen_C[c_idx][s]
                        if val != 0:
                            W[(kg * 2 + kl) * n + i][c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
