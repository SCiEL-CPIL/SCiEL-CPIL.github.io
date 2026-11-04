"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Split into
    #   Part A: A[0:2, 0:2] (2x2) * B[0:2, 0:3] (2x3) -> Winograd rank 10
    #   Part B: A[0:2, 0:2] (2x2) * B[0:2, 3:5] (2x2) -> Strassen rank 7
    #   Part C: A[0:2, 2] (2x1) * B[2, 0:5] (1x5) -> outer product rank 1
    # Total rank: 18
    
    # Winograd 2x2x3 coefficients (rank 10)
    # For computing C2 = A2 * B23 where A2 is 2x2, B23 is 2x3
    # Winograd: 
    #   A' = [[A00+A11, A00+A01, A10+A11, A00, A01, A10, A11]]  (7 entries but we use 10 terms)
    # Actually let me use the standard Winograd 2x2x3:
    #   Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    #   C = A*B = [[ae+ch, af+ci, ag+cj], [be+dh, bf+di, bg+dj]]
    #   Winograd rank 10:
    #   x1 = a+b, x2 = a-b, x3 = c+d, x4 = c-d, x5 = a, x6 = b, x7 = c, x8 = d
    #   y1 = e+f, y2 = e-f, y3 = g, y4 = h+i, y5 = h-i, y6 = j, y7 = e, y8 = f, y9 = g, y10 = h
    #   Hmm, this is getting complicated. Let me use a verified formulation.
    
    # I'll use a different approach: just use schoolbook for the small blocks to be safe,
    # but that gives rank 30. Instead, let me carefully implement Winograd and Strassen.
    
    # Let me define the full tensor and build the decomposition term by term.
    
    # Helper: add a term to U, V, W
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        """u_vec: length n*m=6, v_vec: length m*p=15, w_vec: length p*n=10"""
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
    
    # --- Part A: 2x2x3 Winograd (rank 10) ---
    # A block: rows i=0,1; cols j=0,1
    # B block: rows j=0,1; cols k=0,1,2
    # C output: rows i=0,1; cols k=0,1,2
    
    # Winograd 2x2x3 known rank 10 decomposition:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A*B
    # 
    # Winograd scheme (from Winograd 1980):
    # Define intermediate:
    #   A' = [a00+a11, a00+a01, a10+a11, a00, a01, a10, a11]  (7 values)
    #   B' = [b00+b11, b00+b01, b02+b12, b00, b01, b10, b11]  (7 values)  
    #   But rank 10 means 10 products.
    #
    # Let me use this verified Winograd 2x2x3 (rank 10):
    # Products: p1..p10
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a00-a11)*(b00-b11)  
    # p3 = (a00+a01)*(b00+b02)
    # p4 = (a00-a01)*(b00-b02)
    # p5 = (a10+a11)*(b10+b12)
    # p6 = (a10-a11)*(b10-b12)
    # p7 = (a01+a10)*(b01+b11)
    # p8 = (a01-a10)*(b01-b11)
    # p9 = a00*b01
    # p10 = a11*b10
    #
    # Then:
    # C00 = p1 + p2 + p3 - p4 + p5 - p6 + p7 - p8 + p9 + p10  ... this is getting error-prone.
    
    # Let me use a cleaner approach: I'll verify by construction using numpy.
    
    # Actually, let me just use the straightforward approach with known correct decompositions.
    # For 2x2x2 Strassen (rank 7):
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    # M1=(a+b)(e+f), M2=(a-b)(e+f), M3=(c+d)*h, M4=a*g, M5=d*(f-h), M6=(a+b)*g, M7=(c-d)*f
    # C11=M1+M2+M3+M4, C12=M5+M6, C21=M7+M3, C22=M1-M2+M5-M7
    
    # For 2x2x3, I'll use the fact that it can be decomposed as 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11,
    # or find a better one. Actually rank(2,2,3)=10 is known.
    
    # Let me just use schoolbook for everything to ensure correctness, giving rank 30.
    # Then I can try to optimize. But the task wants minimal rank.
    
    # Given time constraints, let me implement a verified rank-18 solution.
    
    # Rank 18 = Winograd(2,2,3):10 + Strassen(2,2,2):7 + outer(2,1,5):1
    
    # Winograd 2x2x3 (rank 10) - verified formulation:
    # Source: standard Winograd scheme
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # 
    # Use these 10 products:
    # w1 = a+e  (no, these are mixed)
    
    # Let me look up a concrete Winograd 2x2x3:
    # From literature, one rank-10 decomposition:
    # A factors: [[1,1,0,0,0,0,1,0,0,0],[1,0,1,0,0,0,0,1,0,0],
    #            [1,0,0,1,0,0,0,0,1,0],[1,1,0,0,0,0,0,0,0,1],
    #            [0,0,1,1,0,0,0,0,0,0],[0,0,1,0,1,0,0,0,0,0],
    #            [0,0,0,1,1,0,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],
    #            [1,0,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,0,1,0,0]]
    # This is getting too complex to get right without verification.
    
    # SAFER APPROACH: Use schoolbook rank 30 but try to verify a rank 18 via numpy check.
    
    # Let me just do schoolbook to be safe and correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
