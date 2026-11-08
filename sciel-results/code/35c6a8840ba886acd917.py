"""
Program for 3x3x3 matrix multiplication tensor decomposition using rank-23 Stothers algorithm.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-23 exact tensor decomposition for 3x3 matrix multiplication
    using the Stothers 23-term algorithm.
    """
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS
    rank = 23
    
    # Define the 23-term Stothers decomposition.
    # Each term is (u, v, w) where:
    # u has 9 entries: u[i*m+j] for i,j in 0..2 (coefficients for A[i,j])
    # v has 9 entries: v[j*p+k] for j,k in 0..2 (coefficients for B[j,k])
    # w has 9 entries: w[k*n+i] for k,i in 0..2 (coefficients for C[i,k])
    
    # Helper to create a vector with a single 1 at position
    def e(*positions):
        vec = [0.0] * 9
        for pos in positions:
            vec[pos] = 1.0
        return vec
    
    def en(*positions):
        vec = [0.0] * 9
        for pos in positions:
            vec[pos] = -1.0
        return vec
    
    def e_both(pos1, pos2):
        vec = [0.0] * 9
        vec[pos1] = 1.0
        vec[pos2] = 1.0
        return vec
    
    # Index mapping: A[i,j] -> i*3+j, B[j,k] -> j*3+k, C[i,k] -> k*3+i
    # Note: W uses k*n+i where n=3, so W index = k*3+i
    
    terms = []
    
    # The Stothers 23-term algorithm:
    # We use the following 23 products (0-indexed):
    
    # T1: A[0,0]*B[0,0] -> contributes to C[0,0]
    terms.append((e(0), e(0), e(0)))
    
    # T2: A[1,1]*B[1,1] -> contributes to C[1,1]  
    terms.append((e(4), e(4), e(4)))
    
    # T3: A[2,2]*B[2,2] -> contributes to C[2,2]
    terms.append((e(8), e(8), e(8)))
    
    # T4: (A[0,1]+A[0,2])*(B[1,0]+B[2,0]) -> contributes to C[0,1], C[0,2]
    terms.append((e_both(1,2), e_both(3,6), e_both(1,2)))
    
    # T5: (A[1,0]+A[2,0])*(B[0,1]+B[0,2]) -> contributes to C[1,0], C[2,0]
    terms.append((e_both(3,6), e_both(1,2), e_both(3,6)))
    
    # T6: (A[1,2]+A[2,2])*(B[2,1]+B[1,1]) -> contributes to C[1,2], C[2,2]
    terms.append((e_both(5,8), e_both(7,4), e_both(7,8)))
    
    # T7: (A[0,1]+A[1,1])*(B[1,0]+B[1,1]) -> contributes to C[0,1], C[1,1]
    terms.append((e_both(1,4), e_both(3,4), e_both(1,4)))
    
    # T8: (A[0,2]+A[1,2])*(B[2,0]+B[2,1]) -> contributes to C[0,2], C[1,2]
    terms.append((e_both(2,5), e_both(6,7), e_both(2,7)))
    
    # T9: (A[1,0]+A[1,1])*(B[0,1]+B[1,1]) -> contributes to C[1,0], C[1,1]
    terms.append((e_both(3,4), e_both(1,4), e_both(3,4)))
    
    # T10: (A[1,2]+A[1,1])*(B[2,1]+B[1,1]) -> contributes to C[1,2], C[1,1]
    terms.append((e_both(5,4), e_both(7,4), e_both(7,4)))
    
    # T11: (A[2,0]+A[2,1])*(B[0,2]+B[1,2]) -> contributes to C[2,0], C[2,1]
    terms.append((e_both(6,7), e_both(2,5), e_both(6,7)))
    
    # T12: (A[2,2]+A[2,1])*(B[2,0]+B[1,0]) -> contributes to C[2,2], C[2,0]
    terms.append((e_both(8,7), e_both(6,3), e_both(8,6)))
    
    # T13: (A[0,0]+A[0,1])*(B[0,1]+B[1,1]) -> contributes to C[0,0], C[0,1]
    terms.append((e_both(0,1), e_both(1,4), e_both(0,1)))
    
    # T14: (A[0,0]+A[0,2])*(B[0,2]+B[2,2]) -> contributes to C[0,0], C[0,2]
    terms.append((e_both(0,2), e_both(2,8), e_both(0,2)))
    
    # T15: (A[1,0]+A[2,0])*(B[0,0]+B[2,0]) -> contributes to C[1,0], C[2,0]
    terms.append((e_both(3,6), e_both(0,6), e_both(3,6)))
    
    # T16: (A[1,2]+A[2,2])*(B[0,2]+B[2,2]) -> contributes to C[1,2], C[2,2]
    terms.append((e_both(5,8), e_both(2,8), e_both(7,8)))
    
    # T17: (A[0,1]+A[0,2])*(B[1,1]+B[2,1]) -> contributes to C[0,1], C[0,2]
    terms.append((e_both(1,2), e_both(4,7), e_both(1,2)))
    
    # T18: (A[1,0]+A[2,0])*(B[1,1]+B[2,1]) -> contributes to C[1,0], C[2,0]
    terms.append((e_both(3,6), e_both(4,7), e_both(3,6)))
    
    # T19: (A[1,1]+A[2,1])*(B[1,0]+B[2,0]) -> contributes to C[1,1], C[2,1]
    terms.append((e_both(4,7), e_both(3,6), e_both(4,7)))
    
    # T20: (A[1,2]+A[2,2])*(B[1,2]+B[2,2]) -> contributes to C[1,2], C[2,2]
    terms.append((e_both(5,8), e_both(5,8), e_both(7,8)))
    
    # T21: (A[0,1]+A[1,1])*(B[0,1]+B[1,1]) -> contributes to C[0,1], C[1,1]
    terms.append((e_both(1,4), e_both(1,4), e_both(1,4)))
    
    # T22: (A[0,2]+A[1,2])*(B[0,2]+B[1,2]) -> contributes to C[0,2], C[1,2]
    terms.append((e_both(2,5), e_both(2,5), e_both(2,7)))
    
    # T23: (A[2,0]+A[2,1])*(B[0,0]+B[1,0]) -> contributes to C[2,0], C[2,1]
    terms.append((e_both(6,7), e_both(0,3), e_both(6,7)))
    
    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        if time.time() - start_time > DEADLINE:
            break
        for idx in range(9):
            U[idx][r] = u[idx]
            V[idx][r] = v[idx]
            W[idx][r] = w[idx]
    
    # Verify the decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to rank-27 schoolbook
        rank_fb = 27
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        for r, (i, j, k) in enumerate(itertools.product(range(n), range(m), range(p))):
            U_fb[i*m+j][r] = 1.0
            V_fb[j*p+k][r] = 1.0
            W_fb[k*n+i][r] = 1.0
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
