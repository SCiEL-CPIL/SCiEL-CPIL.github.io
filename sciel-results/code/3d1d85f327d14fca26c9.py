"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p = 52
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # For 2x2 * 2xp, there's a known construction achieving rank 3p+1 = 40
    # This uses the fact that we can share some multiplications
    # The construction:
    # Let A = [[a,b],[c,d]], B = [[b1,b2,...,bp],[c1,c2,...,cp]]
    # Output C = [[a*b1+b*c1, a*b2+b*c2, ..., a*bp+b*cp],
    #             [c*b1+d*c1, c*b2+d*c2, ..., c*bp+d*cp]]
    #
    # We can write:
    # C[0,k] = a*b[k] + b*c[k]
    # C[1,k] = c*b[k] + d*c[k]
    #
    # Using rank 3p+1 construction:
    # Columns 0..p-1: handle a*b[k] terms
    #   U[0,k]=1 (for a), V[k,k]=1 (for b[k]), W[k,0]=1 (contributes to C[0,k])
    # Columns p..2p-1: handle b*c[k] terms  
    #   U[1,k-p]=1 (for b), V[k,k-p]=1 (for c[k]), W[k,0]=1 (contributes to C[0,k])
    # Columns 2p..3p-1: handle c*b[k] terms
    #   U[2,k-2p]=1 (for c), V[k,k-2p]=1 (for b[k]), W[k,1]=1 (contributes to C[1,k])
    # Columns 3p..4p-1: handle d*c[k] terms
    #   U[3,k-3p]=1 (for d), V[k,k-3p]=1 (for c[k]), W[k,1]=1 (contributes to C[1,k])
    #
    # Wait, that's 4p not 3p+1. Let me reconsider.
    #
    # Actually for 2x2 * 2xp, the tensor rank is known to be at most 3p+1.
    # A better construction:
    # We can combine some terms. Let's use the standard result.
    # 
    # For multiplying (2x2) by (2xp), we have 4p scalar multiplications in schoolbook.
    # The rank of the matrix multiplication tensor M_{2,2,p} is at most 3p+1.
    #
    # Construction using rank 3p+1:
    # We need 3p+1 rank-1 tensors.
    # 
    # Let me think about this more carefully. The output is 2xp = 2p entries.
    # Each output entry C[i,k] is a bilinear form.
    #
    # C[0,k] = a*b[k] + b*c[k]
    # C[1,k] = c*b[k] + d*c[k]
    #
    # We can write these using shared rank-1 terms.
    #
    # Consider:
    # Term 1: a * b[0] -> contributes to C[0,0]
    # Term 2: b * c[0] -> contributes to C[0,0]
    # Term 3: c * b[0] -> contributes to C[1,0]
    # Term 4: d * c[0] -> contributes to C[1,0]
    # ... and so on for k=1,...,p-1
    #
    # That's 4p terms. To get 3p+1, we need to save p-1 terms.
    #
    # One approach: use the identity that we can combine some multiplications.
    # For example, (a+d)*b[k] = a*b[k] + d*b[k], but we don't have d*b[k] directly.
    #
    # Actually, let me just try to construct it explicitly for small p to verify.
    # For p=1: 2x2 * 2x1, rank should be 4 (since 3*1+1=4).
    # For p=2: 2x2 * 2x2, rank should be 7 (since 3*2+1=7).
    #
    # Let me try a different approach: use the known formula.
    # The rank of M_{2,2,p} is 3p+1 for p >= 1.
    #
    # Construction:
    # For k = 0 to p-1:
    #   U[0, 3k] = 1, V[0, 3k] = 1, W[k*n+0, 3k] = 1  # a * b[k] -> C[0,k]
    #   U[1, 3k+1] = 1, V[1, 3k+1] = 1, W[k*n+0, 3k+1] = 1  # b * c[k] -> C[0,k]
    #   U[2, 3k+2] = 1, V[0, 3k+2] = 1, W[k*n+1, 3k+2] = 1  # c * b[k] -> C[1,k]
    #   U[3, 3k+2] = 1, V[1, 3k+2] = 1, W[k*n+1, 3k+2] = 1  # d * c[k] -> C[1,k]
    #
    # Wait, that uses 3p+1 columns? No, that uses 3p+1? Let me count: 3 per k, so 3p total, plus 1 extra.
    # 
    # Actually, I think the construction is:
    # For k = 0 to p-1:
    #   Column 3k: a * b[k] -> C[0,k]
    #   Column 3k+1: b * c[k] -> C[0,k]  
    #   Column 3k+2: (c+d) * something... no.
    #
    # Let me just return the schoolbook result to ensure validity.
    # The 3p+1 construction is complex and I don't want to introduce errors.
    
    return best_result
# EVOLVE-BLOCK-END
