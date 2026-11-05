"""Matrix multiplication 3x4x4 with rank-44 decomposition using Strassen on 2x2x2 blocks."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # U[i*m+j], V[j*p+k], W[k*n+i]
    
    terms = []  # list of (U_dict, V_dict, W_dict) where each is {index: coeff}
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen 2x2x2: X(2x2)*Y(2x2)=Z(2x2)
    # X[i,j] at A[i0+i, j0+j], Y[j,k] at B[j0+j, k0+k], Z[k,i] at C[k0+k, i0+i]
    # 7 rank-1 terms, each is (linear form in X) * (linear form in Y) contributing to Z
    def strassen(i0, j0, k0):
        # M1 = (X00+X11)*(Y00+Y11), contributes +1 to Z00 and Z11
        add_term({i0*m+j0: 1, (i0+1)*m+(j0+1): 1}, 
                 {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                 {k0*n+i0: 1, (k0+1)*n+(i0+1): 1})
        # M2 = (X10+X11)*Y00, contributes -1 to Z10 and Z11
        add_term({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                 {j0*p+k0: 1},
                 {(k0+1)*n+i0: -1, (k0+1)*n+(i0+1): -1})
        # M3 = X00*(Y01-Y11), contributes +1 to Z01 and Z11
        add_term({i0*m+j0: 1},
                 {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                 {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): 1})
        # M4 = X11*(Y10-Y00), contributes +1 to Z00 and Z10
        add_term({(i0+1)*m+(j0+1): 1},
                 {(j0+1)*p+k0: 1, j0*p+k0: -1},
                 {k0*n+i0: 1, (k0+1)*n+i0: 1})
        # M5 = (X00+X01)*Y11, contributes -1 to Z00 and +1 to Z01
        add_term({i0*m+j0: 1, i0*m+(j0+1): 1},
                 {(j0+1)*p+(k0+1): 1},
                 {k0*n+i0: -1, k0*n+(i0+1): 1})
        # M6 = (X10-X00)*(Y00+Y01), contributes +1 to Z11
        add_term({(i0+1)*m+j0: 1, i0*m+j0: -1},
                 {j0*p+k0: 1, j0*p+(k0+1): 1},
                 {(k0+1)*n+(i0+1): 1})
        # M7 = (X01-X11)*(Y10+Y11), contributes +1 to Z00
        add_term({i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                 {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                 {k0*n+i0: 1})
    
    # Schoolbook 1x2x2: X(1x2)*Y(2x2)=Z(1x2)
    # X[j] at A[i0, j0+j], Y[j,k] at B[j0+j, k0+k], Z[k] at C[k0+k, i0]
    def schoolbook_1x2x2(i0, j0, k0):
        # Z[0] = X0*Y00 + X1*Y10
        add_term({i0*m+j0: 1}, {j0*p+k0: 1}, {k0*n+i0: 1})
        add_term({i0*m+(j0+1): 1}, {(j0+1)*p+k0: 1}, {k0*n+i0: 1})
        # Z[1] = X0*Y01 + X1*Y11
        add_term({i0*m+j0: 1}, {j0*p+(k0+1): 1}, {(k0+1)*n+i0: 1})
        add_term({i0*m+(j0+1): 1}, {(j0+1)*p+(k0+1): 1}, {(k0+1)*n+i0: 1})
    
    # Apply to our 3x4x4 problem:
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    strassen(0, 0, 0)
    strassen(0, 2, 0)
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    strassen(0, 0, 2)
    strassen(0, 2, 2)
    # C[2:3,0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    schoolbook_1x2x2(2, 0, 0)
    schoolbook_1x2x2(2, 2, 0)
    # C[2:3,2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
    schoolbook_1x2x2(2, 0, 2)
    schoolbook_1x2x2(2, 2, 2)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (ud, vd, wd) in enumerate(terms):
        for idx, c in ud.items():
            U[idx][col] = c
        for idx, c in vd.items():
            V[idx][col] = c
        for idx, c in wd.items():
            W[idx][col] = c
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
