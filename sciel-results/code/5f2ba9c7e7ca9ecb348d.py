"""
Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
The output factor follows the official trace convention W[k*n+i,r].
Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
"""

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = 3, 4, 5
    
    # Blocked decomposition:
    # A(3x4): rows[2,1], cols[2,2] -> A11(2x2), A12(2x2), A21(1x2), A22(1x2)
    # B(4x5): rows[2,2], cols[2,3] -> B11(2x2), B12(2x3), B21(2x2), B22(2x3)
    # C(3x5): rows[2,1], cols[2,3] -> C11(2x2), C12(2x3), C21(1x2), C22(1x3)
    #
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x2*2x3)
    # C21 = A21*B11 + A22*B21  (1x2 = 1x2*2x2 + 1x2*2x2)
    # C22 = A21*B12 + A22*B22  (1x3 = 1x2*2x3 + 1x2*2x3)
    
    # Ranks: 2x2x2=7 (Strassen), 2x2x3=12, 1x2x2=4, 1x2x3=6
    # Total: 2*7 + (7+12) + (4+4) + (6+6) = 14+19+8+12 = 53
    
    rank = 53
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, coeffs):
        nonlocal col
        for (idx, coeff) in zip(u_indices, coeffs):
            U[idx][col] += coeff
        for (idx, coeff) in zip(v_indices, coeffs):
            V[idx][col] += coeff
        for (idx, coeff) in zip(w_indices, coeffs):
            W[idx][col] += coeff
        col += 1
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # i0,j0,k0 are top-left of 2x2 blocks
        # A: rows i0,i0+1; cols j0,j0+1
        # B: rows j0,j0+1; cols k0,k0+1
        # C: rows i0,i0+1; cols k0,k0+1
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1])
        # Contributes to C[i0,k0] and C[i0+1,k0+1]
        add_term(
            [i0*m+j0, (i0+1)*m+(j0+1)],
            [j0*p+k0, (j0+1)*p+(k0+1)],
            [k0*n+i0, (k0+1)*n+(i0+1)],
            [1, 1]
        )
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0]
        # Contributes to C[i0+1,k0]
        add_term(
            [(i0+1)*m+j0, (i0+1)*m+(j0+1)],
            [j0*p+k0],
            [k0*n+(i0+1)],
            [1, 1]
        )
        
        # M3 = A[i0,j0]*(B[j0+1,k0]-B[j0+1,k0+1])
        # Contributes to C[i0,k0+1]
        add_term(
            [i0*m+j0],
            [(j0+1)*p+k0, (j0+1)*p+(k0+1)],
            [(k0+1)*n+i0],
            [1, -1]
        )
        
        # M4 = A[i0+1,j0+1]*(B[j0,k0]-B[j0,k0+1])
        # Contributes to -C[i0,k0] and C[i0+1,k0]
        add_term(
            [(i0+1)*m+(j0+1)],
            [j0*p+k0, j0*p+(k0+1)],
            [k0*n+i0, k0*n+(i0+1)],
            [1, -1]
        )
        
        # M5 = (A[i0,j0]+A[i0,j0+1])*B[j0+1,k0+1]
        # Contributes to C[i0,k0+1]
        add_term(
            [i0*m+j0, i0*m+(j0+1)],
            [(j0+1)*p+(k0+1)],
            [(k0+1)*n+i0],
            [1, 1]
        )
        
        # M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
        # Contributes to C[i0,k0]
        add_term(
            [(i0+1)*m+j0, i0*m+j0],
            [j0*p+k0, j0*p+(k0+1)],
            [k0*n+i0],
            [1, -1]
        )
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
        # Contributes to C[i0+1,k0+1]
        add_term(
            [i0*m+(j0+1), (i0+1)*m+(j0+1)],
            [(j0+1)*p+k0, (j0+1)*p+(k0+1)],
            [(k0+1)*n+(i0+1)],
            [1, -1]
        )
    
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    # C11 = A11*B11 + A12*B21
    add_strassen(0, 0, 0)   # A11(0,1)x(0,1) * B11(0,1)x(0,1) -> C11(0,1)x(0,1)
    add_strassen(0, 2, 0)   # A12(0,1)x(2,3) * B21(2,3)x(0,1) -> C11(0,1)x(0,1)
    
    # C12 = A11*B12 + A12*B22
    for di in range(2):
        for dj in range(2):
            for dk in range(3):
                add_schoolbook(di, dj, 2+dk)
    for di in range(2):
        for dj in range(2):
            for dk in range(3):
                add_schoolbook(di, 2+dj, 2+dk)
    
    # C21 = A21*B11 + A22*B21
    for dj in range(2):
        for dk in range(2):
            add_schoolbook(2, dj, dk)
    for dj in range(2):
        for dk in range(2):
            add_schoolbook(2, 2+dj, dk)
    
    # C22 = A21*B12 + A22*B22
    for dj in range(2):
        for dk in range(3):
            add_schoolbook(2, dj, 2+dk)
    for dj in range(2):
        for dk in range(3):
            add_schoolbook(2, 2+dj, 2+dk)
    
    return {"U": U, "V": V, "W": W}
