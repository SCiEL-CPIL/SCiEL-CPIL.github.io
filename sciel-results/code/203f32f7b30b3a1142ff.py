"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    Uses Strassen's algorithm on 2x2x2 blocks where possible, with schoolbook for leftovers.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # Leave 10s margin for 150s limit
    
    # Strategy: Use Strassen on 2x2x2 blocks
    # Split j (middle dimension) into pairs: (0,1), (2,3), and leftover 4
    # Split k (output dimension) into pairs: (0,1), (2,3), (4,5)
    
    # For each pair of j's and pair of k's, we have a 2x2x2 subproblem
    # Strassen gives rank 7 instead of 8 for 2x2x2
    # Leftover j=4 with each k: schoolbook rank 1 each
    
    # Total: 2 j-pairs * 3 k-pairs * 7 (Strassen) + 1 leftover j * 6 k's * 1 (schoolbook)
    # = 2 * 3 * 7 + 6 = 42 + 6 = 48
    
    rank = 48
    U = [[0.0] * rank for _ in range(n * m)]  # 10 x 48
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 48
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 48
    
    col = 0
    
    # Strassen's 7 products for 2x2x2:
    # Given A = [[a, b], [c, d]] (2x2) and B = [[e, f], [g, h]] (2x2)
    # P1 = (a + d)(e + h)
    # P2 = (c + d)e
    # P3 = a(f - h)
    # P4 = d(g - e)
    # P5 = (a + b)h
    # P6 = (c - a)(e + f)
    # P7 = (b - d)(g + h)
    # Result: [[P1 + P4 - P5 + P7, P3 + P5], [P2 + P4, P1 - P2 + P3 + P6]]
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
            # Contributes to: C[i0,k0] += 1, C[i0,k1] += 1, C[i1,k0] += 1, C[i1,k1] += 1
            # Wait, let me be more careful.
            
            # P1 = (A[0,j0] + A[1,j1]) * (B[j0,k0] + B[j1,k1])
            # This product contributes to:
            #   C[0,k0] gets +1 from this term
            #   C[0,k1] gets +1 from this term  
            #   C[1,k0] gets +1 from this term
            #   C[1,k1] gets +1 from this term
            # But wait, that's not right. Let me think about this more carefully.
            
            # For the 2x2x2 subproblem with rows i0=0, i1=1, cols j0, j1, and output cols k0, k1:
            # The sub-multiplication is:
            #   C[0,k0] += A[0,j0]*B[j0,k0] + A[0,j1]*B[j1,k0]
            #   C[0,k1] += A[0,j0]*B[j0,k1] + A[0,j1]*B[j1,k1]
            #   C[1,k0] += A[1,j0]*B[j0,k0] + A[1,j1]*B[j1,k0]
            #   C[1,k1] += A[1,j0]*B[j0,k1] + A[1,j1]*B[j1,k1]
            
            # Strassen for this 2x2x2:
            # Let a=A[0,j0], b=A[0,j1], c=A[1,j0], d=A[1,j1]
            # Let e=B[j0,k0], f=B[j0,k1], g=B[j1,k0], h=B[j1,k1]
            
            # P1 = (a+d)(e+h)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            # P1 contributes to: C[0,k0]+, C[0,k1]+, C[1,k0]+, C[1,k1]+
            # Actually, P1 = (a+d)(e+h) = ae + ah + de + dh
            # This gives contributions to all four outputs
            W[k0*n + 0, col] += 1
            W[k1*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2 = (c+d)e = (A[1,j0] + A[1,j1]) * B[j0,k0]
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            # P2 contributes to: C[1,k0]
            W[k0*n + 1, col] += 1
            col += 1
            
            # P3 = a(f-h) = A[0,j0] * (B[j0,k1] - B[j1,k1])
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            # P3 contributes to: C[0,k1]
            W[k1*n + 0, col] += 1
            col += 1
            
            # P4 = d(g-e) = A[1,j1] * (B[j1,k0] - B[j0,k0])
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            # P4 contributes to: C[1,k0]
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5 = (a+b)h = (A[0,j0] + A[0,j1]) * B[j1,k1]
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            # P5 contributes to: C[0,k1]
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6 = (c-a)(e+f) = (A[1,j0] - A[0,j0]) * (B[j0,k0] + B[j0,k1])
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            # P6 contributes to: C[1,k0] and C[1,k1]
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7 = (b-d)(g+h) = (A[0,j1] - A[1,j1]) * (B[j1,k0] + B[j1,k1])
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            # P7 contributes to: C[0,k0] and C[0,k1]
            W[k0*n + 0, col] += 1
            W[k1*n + 0, col] += 1
            col += 1
    
    # Leftover j=4: schoolbook for each k
    for k in range(p):
        for i in range(n):
            U[i*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + i, col] += 1
            col += 1
    
    # Verify: col should be 48
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Validate the decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook if validation fails
                    r_school = n * m * p
                    U_s = np.zeros((n * m, r_school))
                    V_s = np.zeros((m * p, r_school))
                    W_s = np.zeros((p * n, r_school))
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_s[ii*m+jj, c] = 1
                                V_s[jj*p+kk, c] = 1
                                W_s[kk*n+ii, c] = 1
                                c += 1
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
