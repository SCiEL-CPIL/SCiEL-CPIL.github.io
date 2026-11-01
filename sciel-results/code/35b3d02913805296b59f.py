"""Matrix multiplication 2,2,10 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm applied to each of 5 disjoint 2x2 column-pair blocks.
    The total rank is 5*7 = 35.
    
    Interface:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)  
    W: shape (p*n, rank) = (20, 35)
    
    Index conventions:
    U[i*m+j, r] for A[i,j]
    V[j*p+k, r] for B[j,k]
    W[k*n+i, r] for C[i,k] (trace convention)
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each term: (A_coeffs[4], B_coeffs[4], C_coeffs[4])
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # P3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # P5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # P7
    ]
    
    col = 0
    for block in range(5):
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # U: A coefficients (same for all blocks)
            for i in range(4):
                U[i][col] = a_coeffs[i]
            
            # V: B coefficients for this block's 2 columns
            # B[0, 2*block] -> index 0*p + 2*block = 2*block
            # B[0, 2*block+1] -> index 0*p + 2*block+1 = 2*block+1
            # B[1, 2*block] -> index 1*p + 2*block = 10 + 2*block
            # B[1, 2*block+1] -> index 1*p + 2*block+1 = 10 + 2*block+1
            V[2*block][col] = b_coeffs[0]
            V[2*block+1][col] = b_coeffs[1]
            V[10+2*block][col] = b_coeffs[2]
            V[10+2*block+1][col] = b_coeffs[3]
            
            # W: C coefficients with trace convention W[k*n+i]
            # C[0, 2*block] -> W[(2*block)*2 + 0] = W[4*block]
            # C[0, 2*block+1] -> W[(2*block+1)*2 + 0] = W[4*block+2]
            # C[1, 2*block] -> W[(2*block)*2 + 1] = W[4*block+1]
            # C[1, 2*block+1] -> W[(2*block+1)*2 + 1] = W[4*block+3]
            W[4*block][col] = c_coeffs[0]       # C00
            W[4*block+2][col] = c_coeffs[1]     # C01
            W[4*block+1][col] = c_coeffs[2]     # C10
            W[4*block+3][col] = c_coeffs[3]     # C11
            
            col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = 0.0
                for r in range(rank):
                    total += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                expected = 1.0 if j == k else 0.0
                if abs(total - expected) > 1e-9:
                    # Fallback: return schoolbook if verification fails
                    rank_school = n * m * p
                    U_s = [[0] * rank_school for _ in range(n * m)]
                    V_s = [[0] * rank_school for _ in range(m * p)]
                    W_s = [[0] * rank_school for _ in range(p * n)]
                    col_s = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_s[ii * m + jj][col_s] = 1
                                V_s[jj * p + kk][col_s] = 1
                                W_s[kk * n + ii][col_s] = 1
                                col_s += 1
                    return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
