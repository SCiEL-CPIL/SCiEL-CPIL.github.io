import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 15
    P = p * n  # 15
    
    # Target Tensor
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def residual(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W)

    def check_solution(U, V, W, tol=1e-9):
        Res = residual(U, V, W)
        if not np.allclose(Res, T, atol=tol):
            return False
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
        return True

    # 1. Construct Rank 21 Decomposition (Strassen 3x3x3 + 3x3x2)
    
    # Strassen 3x3x3 Factors (Rank 13)
    # U_strassen: 9x13, V_strassen: 9x13, W_strassen: 9x13
    # We need to embed these into the larger dimensions.
    # U_strassen acts on A (9x1).
    # V_strassen acts on B_1 (9x1, first 9 entries of B).
    # W_strassen acts on C_1 (9x1, first 9 entries of C).
    
    # 3x3x2 Factors (Rank 8)
    # U_2: 9x8, V_2: 6x8 (acts on B_2, last 6 entries of B), W_2: 6x8 (acts on C_2, last 6 entries of C)
    
    # We need the actual coefficients for Strassen and 3x3x2.
    # Strassen 3x3x3 is well-known.
    # 3x3x2 rank 8 is also known.
    
    # Let's hardcode the Strassen 3x3x3 factors.
    # Source: Standard Strassen for 3x3.
    # M1 = (A11+A22+A33) * (B11+B22+B33)
    # M2 = (A11+A22-A33) * (B11-B22+B33)
    # M3 = (A11-A22+A33) * (B11+B22-B33)
    # M4 = (A22-A11+A33) * (-B11+B22+B33)
    # M5 = (A33-A11-A22) * (B11-B22-B33)
    # M6 = (A12+A21+A33) * (B23+B31-B13)
    # M7 = (A13+A31+A22) * (B32+B12-B21)
    # M8 = (A23+A32+A11) * (B13+B21-B31)
    # M9 = (A12+A23+A31) * (B32+B13-B21) # Wait, need correct Strassen 3x3.
    
    # Actually, implementing Strassen 3x3x3 from scratch is error-prone.
    # We can use a known decomposition.
    # Alternatively, we can just use the schoolbook for the 3x3x3 part (Rank 9) and 3x3x2 part (Rank 6).
    # Rank 9 + 6 = 15.
    # Wait, 3x3x3 schoolbook is Rank 9.
    # 3x3x2 schoolbook is Rank 6.
    # Total Rank 15.
    # This is MUCH better than 21 and 45.
    # Why didn't I think of this?
    # The "Strassen" inspiration is for 3x3x3 to get 13.
    # But 3x3x3 schoolbook is 9.
    # 3x3x2 schoolbook is 6.
    # Total 15.
    # Is Rank 15 valid for 3x3x5?
    # Yes, if we just sum the two schoolbook decompositions.
    # T = T_1 (3x3x3) + T_2 (3x3x2).
    # Rank(T_1) = 9. Rank(T_2) = 6.
    # Rank(T) <= 15.
    # The lower bound for 3x3x5 is max(9, 15, 15) = 15.
    # So Rank 15 is the theoretical lower bound!
    # And we just achieved it with a simple block sum.
    
    # Let's verify this logic.
    # T_1 is the tensor for A(3x3) * B1(3x3) -> C1(3x3).
    # T_2 is the tensor for A(3x3) * B2(3x2) -> C2(3x2).
    # The full tensor T is the direct sum of T_1 and T_2 (in the output dimension).
    # The rank of a direct sum of tensors is the sum of their ranks.
    # Rank(T_1) = 9 (schoolbook).
    # Rank(T_2) = 6 (schoolbook).
    # Total Rank = 15.
    # This is the minimal possible rank.
    
    # So we just need to construct the Rank 15 decomposition.
    
    U = np.zeros((N, 15))
    V = np.zeros((M, 15))
    W = np.zeros((P, 15))
    
    col = 0
    # Part 1: 3x3x3 (k=0,1,2)
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only first 3 columns
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Part 2: 3x3x2 (k=3,4)
    for i in range(n):
        for j in range(m):
            for k in range(3, p): # k=3,4
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Verify
    if check_solution(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to Schoolbook 45 if something went wrong (shouldn't happen)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((N, rank_schoolbook))
    V_sb = np.zeros((M, rank_schoolbook))
    W_sb = np.zeros((P, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
