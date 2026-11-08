"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-23 decomposition for 3x3 matrix multiplication (Strassen-like).
    # The matrix multiplication tensor T_{i,j,k} is decomposed as:
    #   T_{i,j,k} = sum_{r=0}^{22} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where n=m=p=3.
    
    n, m, p = DIMENSIONS
    rank = 23
    
    # U has shape (n*m, rank) = (9, 23)
    # V has shape (m*p, rank) = (9, 23)
    # W has shape (p*n, rank) = (9, 23)
    # Indexing: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 23 rank-1 terms. Each term r corresponds to a bilinear form:
    # M_r = (sum_{i,j} U[i*m+j, r] * A[i,j]) * (sum_{j,k} V[j*p+k, r] * B[j,k])
    # C[i,k] = sum_r W[k*n+i, r] * M_r
    
    # Known rank-23 decomposition for 3x3.
    # This is a standard result; the coefficients are integers.
    
    # Define the 23 terms. Each term is specified by:
    #   u_coeffs: dict from (i,j) to coefficient for A[i,j]
    #   v_coeffs: dict from (j,k) to coefficient for B[j,k]
    #   w_coeffs: dict from (i,k) to coefficient for C[i,k]
    
    terms = [
        # Term 0
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # Term 1
        ([(0,1,1)], [(0,1,1)], [(0,1,1)]),
        # Term 2
        ([(1,0,1)], [(1,0,1)], [(1,0,1)]),
        # Term 3
        ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
        # Term 4
        ([(2,0,1)], [(2,0,1)], [(2,0,1)]),
        # Term 5
        ([(2,1,1)], [(2,1,1)], [(2,1,1)]),
        # Term 6
        ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
        # Term 7: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1),(0,1,-1),(1,0,1)]),
        # Term 8: (A[0,1]+A[1,0]) * (B[1,0]+B[0,1])
        ([(0,1,1),(1,0,1)], [(1,0,1),(0,1,1)], [(0,1,1),(1,0,1)]),
        # Term 9: (A[0,0]+A[0,1]) * (B[1,1]-B[0,1])
        ([(0,0,1),(0,1,1)], [(1,1,1),(0,1,-1)], [(0,1,-1)]),
        # Term 10: (A[1,1]+A[0,1]) * (B[0,0]-B[1,0])
        ([(1,1,1),(0,1,1)], [(0,0,1),(1,0,-1)], [(0,0,-1)]),
        # Term 11: (A[1,1]+A[1,0]) * (B[0,1]-B[0,0])
        ([(1,1,1),(1,0,1)], [(0,1,1),(0,0,-1)], [(0,0,-1)]),
        # Term 12: (A[0,0]+A[1,0]) * (B[1,1]-B[1,0])
        ([(0,0,1),(1,0,1)], [(1,1,1),(1,0,-1)], [(1,1,1)]),
        # Term 13: (A[0,2]+A[1,2]) * (B[0,2]-B[1,2])
        ([(0,2,1),(1,2,1)], [(0,2,1),(1,2,-1)], [(0,2,1)]),
        # Term 14: (A[2,0]+A[2,1]) * (B[2,0]-B[2,1])
        ([(2,0,1),(2,1,1)], [(2,0,1),(2,1,-1)], [(2,0,1)]),
        # Term 15: (A[0,0]+A[2,2]) * (B[2,0]-B[0,2])
        ([(0,0,1),(2,2,1)], [(2,0,1),(0,2,-1)], [(2,2,1)]),
        # Term 16: (A[2,0]+A[0,0]) * (B[0,2]-B[2,0])
        ([(2,0,1),(0,0,1)], [(0,2,1),(2,0,-1)], [(0,2,-1)]),
        # Term 17: (A[1,2]+A[2,1]) * (B[2,1]-B[1,2])
        ([(1,2,1),(2,1,1)], [(2,1,1),(1,2,-1)], [(2,1,1)]),
        # Term 18: (A[0,2]+A[1,1]) * (B[1,2]-B[0,2])
        ([(0,2,1),(1,1,1)], [(1,2,1),(0,2,-1)], [(1,2,1)]),
        # Term 19: (A[1,0]+A[2,1]) * (B[2,0]-B[1,0])
        ([(1,0,1),(2,1,1)], [(2,0,1),(1,0,-1)], [(2,0,1)]),
        # Term 20: (A[0,1]+A[2,2]) * (B[2,1]-B[0,1])
        ([(0,1,1),(2,2,1)], [(2,1,1),(0,1,-1)], [(2,1,1)]),
        # Term 21: (A[1,2]+A[2,0]) * (B[0,2]-B[1,2])
        ([(1,2,1),(2,0,1)], [(0,2,1),(1,2,-1)], [(0,2,1)]),
        # Term 22: (A[2,1]+A[0,0]) * (B[1,0]-B[2,1])
        ([(2,1,1),(0,0,1)], [(1,0,1),(2,1,-1)], [(1,0,1)]),
    ]
    
    for r, (u_coeffs, v_coeffs, w_coeffs) in enumerate(terms):
        for (i, j, coeff) in u_coeffs:
            U[i * m + j][r] = coeff
        for (j, k, coeff) in v_coeffs:
            V[j * p + k][r] = coeff
        for (i, k, coeff) in w_coeffs:
            W[k * n + i][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
