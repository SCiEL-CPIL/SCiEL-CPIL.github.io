"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 18 decomposition for 2x3 x 3x4
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, r):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val

    # We split the 3x4 matrix B into two 3x2 matrices B1 and B2.
    # C = A * B = A * [B1, B2] = [A*B1, A*B2]
    # A is 2x3. B1 is 3x2. A*B1 is 2x2.
    # This is M_{2,3,2} which is isomorphic to M_{2,2,3}.
    # r(M_{2,2,3}) = 9.
    # So we need two rank-9 blocks.
    
    # Block 1: Compute A * B1 (cols 0,1 of B)
    # B1 indices: j in 0..2, k in 0..1
    # A indices: i in 0..1, j in 0..2
    # C1 indices: i in 0..1, k in 0..1
    
    # We use the rank-9 decomposition for M_{2,2,3} mapped to M_{2,3,2}.
    # M_{2,2,3}: A (2x2), B (2x3).
    # Here A is 2x3, B is 3x2.
    # This is M_{2,3,2}.
    # We can just use the schoolbook for the 2x2 part? No, we need rank 9.
    # Let's use the explicit rank-9 decomposition for M_{2,2,3}.
    # M_{2,2,3}: A (2x2), B (2x3).
    # C (2x3).
    # r = 7 (for first 2 cols) + 2 (for 3rd col).
    
    # For M_{2,3,2}, we can view it as M_{2,2,3} by swapping indices?
    # M_{2,3,2} is isomorphic to M_{2,2,3} by swapping m and p?
    # M_{n,m,p} ~ M_{n,p,m}.
    # So M_{2,3,2} ~ M_{2,2,3}.
    # The decomposition for M_{2,2,3} can be adapted.
    
    # Let's construct M_{2,2,3} rank-9 decomposition first.
    # A: (i,j) i=0,1 j=0,1
    # B: (j,k) j=0,1 k=0,1,2
    # C: (i,k) i=0,1 k=0,1,2
    
    # Part 1: M_{2,2,2} for k=0,1 (Rank 7)
    # Part 2: M_{2,2,1} for k=2 (Rank 2)
    
    # We will generate the terms for Block 1 (cols 0,1 of original B)
    # and Block 2 (cols 2,3 of original B)
    
    # Block 1: k_orig = 0, 1. Corresponds to k_local = 0, 1 in M_{2,2,3}?
    # No, M_{2,2,3} has k=0,1,2.
    # Here we have 2 columns.
    # So we use M_{2,3,2}.
    # Let's just use the rank-9 decomposition for M_{2,2,3} and map it.
    
    # Actually, simpler:
    # Use rank-7 for 2x2x2 and rank-2 for 2x2x1.
    # For M_{2,3,2}:
    # A is 2x3. B is 3x2.
    # We can split A into A1 (2x2) and A2 (2x1)?
    # No, B is 3x2.
    # Let's use the known rank-9 for M_{2,2,3} and apply it to M_{2,3,2} via isomorphism.
    
    # Isomorphism M_{2,3,2} -> M_{2,2,3}:
    # A_{i,j} (i=0,1; j=0,1,2) -> A'_{i,j'} (i=0,1; j'=0,1)
    # B_{j,k} (j=0,1,2; k=0,1) -> B'_{j',k'} (j'=0,1; k'=0,1,2)
    # This mapping is not straightforward.
    
    # Let's just implement the rank-9 for M_{2,2,3} and use it twice?
    # No, M_{2,3,2} is not M_{2,2,3}.
    
    # Let's use the rank-18 decomposition directly.
    # It is known that r(M_{2,3,4}) = 18.
    # We will hardcode the 18 terms.
    
    # Since I don't have the explicit 18 terms memorized, I will use a fallback
    # or a simpler construction if possible.
    # But rank 18 is the target.
    
    # If I cannot construct rank 18, I should return the schoolbook (24).
    # But the task asks for improvement.
    
    # Let's try to construct rank 18 using the 7+2+7+2 structure?
    # Block 1 (cols 0,1): M_{2,3,2}.
    # Block 2 (cols 2,3): M_{2,3,2}.
    
    # r(M_{2,3,2}) = 9.
    # How to get rank 9 for M_{2,3,2}?
    # M_{2,3,2} ~ M_{2,2,3}.
    # r(M_{2,2,3}) = 9.
    
    # Let's assume the rank-9 decomposition for M_{2,2,3} is:
    # 7 terms for M_{2,2,2} (cols 0,1)
    # 2 terms for M_{2,2,1} (col 2)
    
    # For M_{2,3,2}, we have 2 columns.
    # So we use 7 terms for the first column? No.
    # M_{2,3,2} has 2 columns.
    # We can use 7 terms for the 2x2x2 part?
    # A is 2x3. B is 3x2.
    # This is not 2x2x2.
    
    # I will return the schoolbook decomposition to ensure validity, 
    # as constructing the exact rank-18 terms without lookup is error-prone.
    # However, the score will be 1.0.
    
    # To improve, I must find the rank-18 terms.
    # Given the complexity, I will stick to the schoolbook method 
    # unless I can verify the rank-18 construction.
    
    # Wait, the inspiration says "uses _check_decomposition".
    # I should add a check.
    
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
