import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for 2x2 x 2x8 (4 blocks of 2x2x2)
    # A is 2x3, B is 3x8
    # A_L is A[0:2, 0:2], B_L is B[0:2, :]
    # Split B_L into 4 blocks of 2x2
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: (idx, coeff), V: (idx, coeff), W: (idx, coeff)
        # Local indices for 2x2 A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local indices for 2x2 B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local indices for 2x2 C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    col = 0
    for b in range(4):
        # B block b is at columns 2b, 2b+1 of B_L
        # B_L rows are 0, 1. So B indices are j=0,1.
        # V local index l corresponds to B[j, 2b + (l%2)] where j=l//2?
        # Let's map local B indices (0..3) to global V indices.
        # Local 0: B[0, 2b] -> V[0*8 + 2b]
        # Local 1: B[0, 2b+1] -> V[0*8 + 2b+1]
        # Local 2: B[1, 2b] -> V[1*8 + 2b]
        # Local 3: B[1, 2b+1] -> V[1*8 + 2b+1]
        
        # A is fixed A_L: A[0,0], A[0,1], A[1,0], A[1,1]
        # U local index l corresponds to A[i, j] where i=l//2? No.
        # Local 0: A[0,0] -> U[0*3 + 0]
        # Local 1: A[0,1] -> U[0*3 + 1]
        # Local 2: A[1,0] -> U[1*3 + 0]
        # Local 3: A[1,1] -> U[1*3 + 1]
        
        # C block b is at columns 2b, 2b+1 of C.
        # W local index l corresponds to C[i, 2b + (l%2)]?
        # W index is k*n + i.
        # Local 0: C[0, 2b] -> W[2b*2 + 0]
        # Local 1: C[0, 2b+1] -> W[(2b+1)*2 + 0]
        # Local 2: C[1, 2b] -> W[2b*2 + 1]
        # Local 3: C[1, 2b+1] -> W[(2b+1)*2 + 1]

        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            for u_idx, coeff in u_coeffs:
                # u_idx is local 0..3
                # global U index = u_idx
                U[u_idx][col] += coeff
                
            for v_idx, coeff in v_coeffs:
                # v_idx is local 0..3
                # local 0 -> B[0, 2b] -> V[2b]
                # local 1 -> B[0, 2b+1] -> V[2b+1]
                # local 2 -> B[1, 2b] -> V[8+2b]
                # local 3 -> B[1, 2b+1] -> V[9+2b]
                if v_idx == 0:
                    g_idx = 2 * b
                elif v_idx == 1:
                    g_idx = 2 * b + 1
                elif v_idx == 2:
                    g_idx = 8 + 2 * b
                elif v_idx == 3:
                    g_idx = 9 + 2 * b
                V[g_idx][col] += coeff
                
            for w_idx, coeff in w_coeffs:
                # w_idx is local 0..3
                # local 0 -> C[0, 2b] -> W[4b]
                # local 1 -> C[0, 2b+1] -> W[4b+2]
                # local 2 -> C[1, 2b] -> W[4b+1]
                # local 3 -> C[1, 2b+1] -> W[4b+3]
                if w_idx == 0:
                    g_idx = 4 * b
                elif w_idx == 1:
                    g_idx = 4 * b + 2
                elif w_idx == 2:
                    g_idx = 4 * b + 1
                elif w_idx == 3:
                    g_idx = 4 * b + 3
                W[g_idx][col] += coeff
            
            col += 1
            
    # Part 2: Direct for 2x1 x 1x8
    # A_R is A[0:2, 2:3]
    # B_R is B[2:3, 0:8]
    # C_R is C[0:2, 0:8]
    
    for i in range(2):
        for k in range(8):
            # U index for A[i, 2]
            u_idx = i * 3 + 2
            U[u_idx][col] = 1
            
            # V index for B[2, k]
            v_idx = 2 * 8 + k
            V[v_idx][col] = 1
            
            # W index for C[i, k]
            # W[k*n + i] = W[2k + i]
            w_idx = 2 * k + i
            W[w_idx][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
