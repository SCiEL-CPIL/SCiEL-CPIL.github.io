"""
Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy:
- Partition the 4 intermediate indices j into two groups of 2.
- Partition the 8 output indices k into four groups of 2.
- Each (j-group, k-group) pair defines a 2x2 * 2x2 sub-multiplication.
- Apply Strassen's rank-7 algorithm to each sub-problem.
- Total rank = 2 * 4 * 7 = 56 (vs. schoolbook 64).

The factor matrices U, V, W satisfy the tensor identity:
  C[i,k] = sum_j A[i,j] B[j,k]  ==  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
using the official trace convention W[k*n+i, r].
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen rank-7 scheme for 2x2 * 2x2:
    #   A = [[a,b],[c,d]],  B = [[e,f],[g,h]]
    #   M1=(a+d)(e+h)  M2=(c+d)e  M3=a(f-h)  M4=d(g-e)
    #   M5=(a+b)h      M6=(c-a)(e+f)  M7=(b-d)(g+h)
    #   C11=M1+M4-M5+M7  C12=M3+M5  C21=M2+M4  C22=M1-M2+M3+M6

    # A-side coefficients per Strassen term (keys: 0=a,1=b,2=c,3=d)
    A_side = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1},
    ]
    # B-side coefficients per Strassen term (keys: 0=e,1=f,2=g,3=h)
    B_side = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1},
    ]
    # C-side combination per output position (keys: Strassen term index 0..6)
    # C11: M1+M4-M5+M7, C12: M3+M5, C21: M2+M4, C22: M1-M2+M3+M6
    C_side = [
        {0: 1, 3: 1, 4: -1, 6: 1},   # C11
        {2: 1, 4: 1},                 # C12
        {1: 1, 3: 1},                 # C21
        {0: 1, 1: -1, 2: 1, 5: 1},   # C22
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for jg in range(2):
        for kg in range(4):
            for s in range(7):
                c = (jg * 4 + kg) * 7 + s
                a_map = A_side[s]
                b_map = B_side[s]

                # U: A[i, jg*2 + jl]
                for i in range(2):
                    for jl in range(2):
                        coef = a_map.get(i * 2 + jl, 0)
                        if coef:
                            U[i * m + jg * 2 + jl][c] = coef

                # V: B[jg*2 + jl, kg*2 + kl]
                for jl in range(2):
                    for kl in range(2):
                        coef = b_map.get(jl * 2 + kl, 0)
                        if coef:
                            V[(jg * 2 + jl) * p + kg * 2 + kl][c] = coef

                # W: C[i, kg*2 + kl]  ->  W[(kg*2+kl)*n + i]
                for i in range(2):
                    for kl in range(2):
                        coef = C_side[i * 2 + kl].get(s, 0)
                        if coef:
                            W[(kg * 2 + kl) * n + i][c] = coef

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
