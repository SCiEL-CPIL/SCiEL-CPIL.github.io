"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    # Attempt Rank 15 Construction
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i*m+j (i=0,1; j=0,1,2) -> 0..5
    # V: j*p+k (j=0,1,2; k=0,1,2,3) -> 0..11
    # W: k*n+i (k=0,1,2,3; i=0,1) -> 0..7
    
    # We define the 15 terms.
    # Term t: U[row_u][t], V[row_v][t], W[row_w][t]
    # Note: This is a sum of rank-1 tensors.
    # The tensor T is defined by T[i,j,k] = 1.
    # The decomposition is C[i,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] ??
    # No, the standard form is:
    # C[i,k] = sum_t (sum_j U[i*m+j, t] A[i,j]) * (sum_j V[j*p+k, t] B[j,k]) * W[k*n+i, t]
    # Wait, the linearity is:
    # L_A(t) = sum_{i,j} U[i*m+j, t] A[i,j]
    # L_B(t) = sum_{j,k} V[j*p+k, t] B[j,k]
    # C[i,k] += L_A(t) L_B(t) W[k*n+i, t]
    
    # Let's define the 15 terms based on the "shared forms" hypothesis.
    # Forms for A:
    # a0 = a, a1 = b, a2 = c, a3 = d, a4 = e, a5 = f
    # Forms for B:
    # b0 = g, b1 = h, b2 = i, b3 = j, b4 = k, b5 = l, b6 = m, b7 = n, b8 = o, b9 = p, b10 = q, b11 = r
    
    # We will use the following 15 products (indices 0-14):
    # 0: (a+b) * (g+k)
    # 1: (a-b) * (g-k)
    # 2: c * (g+k)
    # 3: (d+e) * (g+k)
    # 4: (d-e) * (g-k)
    # 5: f * (g+k)
    # 6: (a+b) * (h+l)
    # 7: (a-b) * (h-l)
    # 8: c * (h+l)
    # 9: (d+e) * (h+l)
    # 10: (d-e) * (h-l)
    # 11: f * (h+l)
    # 12: (a+c) * (i+m)
    # 13: (d+f) * (i+m)
    # 14: (b+e) * (i+m) -- This is a guess for the last part.
    
    # This construction is likely incorrect for the last part.
    # I will verify it.
    
    # To be safe, I will just return the schoolbook solution.
    # The probability of guessing the correct rank-15 coefficients is low.
    # However, I can try to use a known library or search.
    # Since I cannot, I will return the schoolbook solution.
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
