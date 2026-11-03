import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 17 (for 2x4x4) + 14 (for 2x4x2) = 31
    rank = 31
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x4x4 using restricted 4x4x4 rank-17 decomposition
    # We need the 17 factors for 4x4x4.
    # U4, V4, W4 are lists of 17 4x4 matrices.
    
    # Hardcoded 4x4x4 Rank-17 Decomposition (Laser Method)
    # Note: These are specific integer/half-integer coefficients.
    # I will use a known valid set.
    
    # If I cannot guarantee the 17-rank factors, I will use a fallback.
    # However, for this task, I will assume I can provide them.
    # Let's use a simpler, verifiable approach if possible.
    # Actually, I will use the schoolbook for 2x4x4 (rank 32) if I am unsure about 17.
    # But 32+14=46 is only slightly better than 48.
    # Let's try to use the 17-rank one.
    
    # I will use a standard 4x4x4 rank-17 decomposition.
    # Factors are 4x4 matrices.
    # Due to complexity, I will implement the 2x4x2 part (rank 14) and use schoolbook for 2x4x4 (rank 32)
    # Total 46.
    # Wait, 2x4x4 schoolbook is 32.
    # 32 + 14 = 46.
    # Score 48/46 = 1.043.
    
    # Can I do better for 2x4x4?
    # 2x4x4 can be done with rank 28 (two 14s).
    # 28 + 14 = 42.
    # Score 48/42 = 1.142.
    
    # Let's implement the rank-28 for 2x4x4.
    # Split A (2x4) into A1 (2x2) and A2 (2x2).
    # Split B (4x4) into B1 (2x4) and B2 (2x4).
    # C = A1 B1 + A2 B2.
    # Rank of 2x2x4 is 14.
    # Total 28.
    
    # So total rank 42.
    
    col = 0
    # Add 2x4x4 decomposition (rank 28)
    # A1: i in 0,1; j in 0,1
    # A2: i in 0,1; j in 2,3
    # B1: j in 0,1; k in 0..3
    # B2: j in 2,3; k in 0..3
    
    # We need rank-14 decomposition for 2x2x4.
    # 2x2x4 is two 2x2x2 multiplications.
    # Rank 7 + 7 = 14.
    
    def add_strassen_2x2x2(offset_u, offset_v, offset_w, col_start, A_i_range, A_j_range, B_j_range, B_k_range):
        # A is 2x2, B is 2x2
        # Strassen factors
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-e)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices in local 2x2 space:
        # a: (0,0), b: (0,1), c: (1,0), d: (1,1)
        # e: (0,0), f: (0,1), g: (1,0), h: (1,1)
        
        # U factors (2x2)
        U1 = [[0,0],[0,0]]; U1[0][0]=1; U1[1][1]=1
        U2 = [[0,0],[0,0]]; U2[1][0]=1; U2[1][1]=1
        U3 = [[0,0],[0,0]]; U3[0][0]=1
        U4 = [[0,0],[0,0]]; U4[1][1]=1
        U5 = [[0,0],[0,0]]; U5[0][0]=1; U5[0][1]=1
        U6 = [[0,0],[0,0]]; U6[1][0]=1; U6[0][0]=-1
        U7 = [[0,0],[0,0]]; U7[0][1]=1; U7[1][1]=-1
        
        # V factors (2x2)
        V1 = [[0,0],[0,0]]; V1[0][0]=1; V1[1][1]=1
        V2 = [[0,0],[0,0]]; V2[0][0]=1
        V3 = [[0,0],[0,0]]; V3[0][1]=1; V3[1][1]=-1
        V4 = [[0,0],[0,0]]; V4[1][0]=1; V4[0][0]=-1
        V5 = [[0,0],[0,0]]; V5[1][1]=1
        V6 = [[0,0],[0,0]]; V6[0][0]=1; V6[0][1]=1
        V7 = [[0,0],[0,0]]; V7[1][0]=1; V7[1][1]=1
        
        # W factors (2x2) - Output C = [[C00, C01], [C10, C11]]
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        W1 = [[1,0],[0,1]]
        W2 = [[0,0],[1,0]]
        W3 = [[0,1],[0,1]]
        W4 = [[1,0],[1,0]]
        W5 = [[-1,1],[0,0]]
        W6 = [[0,0],[0,1]]
        W7 = [[1,0],[0,0]]
        
        Us = [U1, U2, U3, U4, U5, U6, U7]
        Vs = [V1, V2, V3, V4, V5, V6, V7]
        Ws = [W1, W2, W3, W4, W5, W6, W7]
        
        for t in range(7):
            # Map local indices to global
            # A global: i in A_i_range, j in A_j_range
            # B global: j in B_j_range, k in B_k_range
            # W global: k in B_k_range, i in A_i_range
            
            for li in range(2):
                for lj in range(2):
                    val = Us[t][li][lj]
                    if val != 0:
                        gi = A_i_range[li]
                        gj = A_j_range[lj]
                        U[gi*m + gj][col_start + t] += val
                        
            for lj in range(2):
                for lk in range(2):
                    val = Vs[t][lj][lk]
                    if val != 0:
                        gj = B_j_range[lj]
                        gk = B_k_range[lk]
                        V[gj*p + gk][col_start + t] += val
                        
            for lk in range(2):
                for li in range(2):
                    val = Ws[t][lk][li]
                    if val != 0:
                        gk = B_k_range[lk]
                        gi = A_i_range[li]
                        W[gk*n + gi][col_start + t] += val
                        
        return col_start + 7

    # 2x4x4 part:
    # Split into 2x2x4 (A1, B1) and 2x2x4 (A2, B2)
    # A1: i in 0,1; j in 0,1
    # B1: j in 0,1; k in 0..3
    # A2: i in 0,1; j in 2,3
    # B2: j in 2,3; k in 0..3
    
    # 2x2x4 is two 2x2x2:
    # B1 split into B1a (k 0,1) and B1b (k 2,3)
    # B2 split into B2a (k 0,1) and B2b (k 2,3)
    
    # 1. A1 B1a
    col = add_strassen_2x2x2(0, 0, 0, col, [0,1], [0,1], [0,1], [0,1])
    # 2. A1 B1b
    col = add_strassen_2x2x2(0, 0, 0, col, [0,1], [0,1], [0,1], [2,3])
    # 3. A2 B2a
    col = add_strassen_2x2x2(0, 0, 0, col, [0,1], [2,3], [2,3], [0,1])
    # 4. A2 B2b
    col = add_strassen_2x2x2(0, 0, 0, col, [0,1], [2,3], [2,3], [2,3])
    
    # Now col is 28.
    # 2x4x2 part:
    # A1: i 0,1; j 0,1. B1: j 0,1; k 4,5
    # A2: i 0,1; j 2,3. B2: j 2,3; k 4,5
    
    # 5. A1 B1 (k 4,5)
    col = add_strassen_2x2x2(0, 0, 0, col, [0,1], [0,1], [0,1], [4,5])
    # 6. A2 B2 (k 4,5)
    col = add_strassen_2x2x2(0, 0, 0, col, [0,1], [2,3], [2,3], [4,5])
    
    # Total col should be 42.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
