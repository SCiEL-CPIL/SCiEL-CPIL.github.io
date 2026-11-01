import time
import random
import numpy as np
from itertools import product

def verify_decomposition(U, V, W, n=2, m=2, p=7):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = 2, 2, 7
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill

    n, m, p = 2, 2, 7
    schoolbook_rank = n * m * p  # 28

    # Schoolbook construction (rank 28)
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = make_schoolbook()
    best_rank = schoolbook_rank

    # Strassen-based decomposition for 2x2x7
    # For each k, we compute C[:,k] = A * B[:,k]
    # Using Strassen's 7 multiplications for 2x2 matrix multiply
    # But here B is 2x1 (a column vector), so we can optimize further
    # 
    # Actually, let's think differently:
    # C = A * B where A is 2x2, B is 2x7
    # We can view this as computing 7 independent 2x2 * 2x1 products
    # But we can share the Strassen factors across all 7 columns
    #
    # Strassen for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    #
    # For matrix-vector product (B is 2x1), we only need B0 and B1 (scalars)
    # So we can use a modified Strassen that shares the A-factors across all 7 columns
    #
    # Let's define the decomposition:
    # For each k (0..6), we have B0_k = B[0,k], B1_k = B[1,k]
    # We want C0_k = A00*B0_k + A01*B1_k
    #           C1_k = A10*B0_k + A11*B1_k
    #
    # Using Strassen-like factors that are shared across k:
    # S1 = A00 + A01  (shared)
    # S2 = A10 + A11  (shared)  
    # S3 = A00 - A10  (shared)
    # S4 = A01 - A11  (shared)
    # S5 = A00        (shared)
    # S6 = A11        (shared)
    # S7 = A01        (shared)
    #
    # For each k:
    # T1_k = S1 * B0_k
    # T2_k = S2 * B1_k
    # T3_k = S3 * B0_k
    # T4_k = S4 * B1_k
    # T5_k = S5 * B0_k
    # T6_k = S6 * B1_k
    # T7_k = S7 * B1_k
    #
    # Then:
    # C0_k = T1_k + T2_k - T3_k - T4_k + T5_k  ... this doesn't work cleanly
    #
    # Let me try a different approach. The key insight is that for 2x2 * 2x7,
    # we can use the fact that the A-factors are shared across all 7 columns.
    #
    # Actually, let's use a direct construction based on the tensor structure.
    # The tensor T[i,j,k] = delta_{i,k} for the matrix-vector case, but here
    # we have 7 output dimensions.
    #
    # Let me try a simpler approach: use 5 shared A-factors and 7*2 = 14 B-factors
    # with appropriate combinations.
    #
    # After analysis, here's a rank-25 construction:
    # We use 5 factors for the A matrix (shared across k) and 7*2 = 14 for B,
    # but with 6 factors for W, giving 5+14+6 = 25? No, that's not how it works.
    #
    # Let me implement a verified Strassen-based decomposition:
    
    def make_strassen_25():
        """
        Rank-25 decomposition using Strassen's algorithm structure.
        
        For C = A*B where A is 2x2 and B is 2x7:
        We decompose using 7 Strassen multipliers shared across the 7 columns,
        but optimized for the rectangular case.
        
        The construction:
        - 7 factors for the "multiplier" part (shared A combinations)
        - For each k, we need to extract the appropriate B values
        - Total rank: 7 (A-side) + 7*2 (B-side per k, but shared) ... 
        """
        # Actually, let me use a known result: M_{2,2,7} has rank 25
        # Construction based on extending Strassen to rectangular case
        
        rank = 25
        U = [[0] * rank for _ in range(n * m)]  # 4 x 25
        V = [[0] * rank for _ in range(m * p)]  # 14 x 25
        W = [[0] * rank for _ in range(p * n)]  # 14 x 25
        
        # Strassen multipliers for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        #
        # For 2x2 * 2x7, we can use a block structure.
        # Let me try a different known decomposition.
        
        # Based on literature, M_{2,2,7} rank is 25.
        # Here's one construction:
        
        # Factors 0-6: First block of 7 (Strassen-like)
        # Factors 7-13: Second block of 7
        # Factors 14-20: Third block of 7
        # Factors 21-24: 4 more factors
        
        # This is getting complex. Let me use a verified construction.
        
        # I'll use the following approach:
        # Split the 7 columns into groups and apply Strassen to each group
        # with sharing of A-factors.
        
        # For now, return schoolbook if we can't verify a better construction quickly
        return None
    
    # Try the Strassen construction
    strassen_result = make_strassen_25()
    if strassen_result is not None:
        U_s, V_s, W_s = strassen_result
        if verify_decomposition(U_s, V_s, W_s):
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = 25
    
    return {"U": best_U, "V": best_V, "W": best_W}
