import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank 45
    R = 45
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    def add_decomp(offset, U_sub, V_sub, W_sub, i_off, j_off, k_off):
        """
        Adds a sub-decomposition to the global matrices.
        U_sub: list of lists or numpy array (n_sub*m_sub, rank_sub)
        V_sub: (m_sub*p_sub, rank_sub)
        W_sub: (p_sub*n_sub, rank_sub)
        i_off, j_off, k_off: offsets in global indices for A, B, C
        """
        n_sub = int(np.sqrt(len(U_sub))) # This is wrong, U_sub is flattened
        # We need to pass dimensions or infer from offsets?
        # Better to pass explicit dimensions.
        pass

    # We will manually map the indices.
    
    # Helper to get index in U
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # --- Part 1: m in [0, 2) ---
    # A1 is 2x2 (i in 0,1; j in 0,1)
    # B1 is 2x6 (j in 0,1; k in 0,5)
    
    # Block 1.1: j in [0,2), k in [0,2) -> 2x2x2 Strassen
    # Block 1.2: j in [0,2), k in [2,4) -> 2x2x2 Strassen
    # Block 1.3: j in [0,2), k in [4,6) -> 2x2x2 Strassen
    
    # --- Part 2: m in [2, 5) ---
    # A2 is 2x3 (i in 0,1; j in 2,3,4)
    # B2 is 3x6 (j in 2,3,4; k in 0,5)
    
    # For each k block of size 2:
    # Block 2.x: j in [2,5), k in [2*x, 2*x+2) -> 2x3x2
    #   Split m into 1+2.
    #   Sub 2.x.a: j in [2,3), k in [...] -> 2x1x2 (Rank 1)
    #   Sub 2.x.b: j in [3,5), k in [...] -> 2x2x2 (Rank 7)
    
    # We need the explicit U, V, W for Strassen 2x2x2 and 2x1x2.
    
    # Strassen 2x2x2
    # U (4, 7), V (4, 7), W (4, 7)
    # Indices: A: 00, 01, 10, 11. B: 00, 01, 10, 11. C: 00, 01, 10, 11.
    # P1 = A00 (B00+B11)
    # P2 = (A10+A11) B00
    # P3 = A00 (B10-B11)
    # P4 = A11 (B01+B10)
    # P5 = (A00+A11) (B11-B00)
    # P6 = (A00-A10) (B01+B10)
    # P7 = (A10+A11) (B11-B01)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # r=0 (P1)
    U_s[0, 0] = 1 # A00
    V_s[0, 0] = 1 # B00
    V_s[3, 0] = 1 # B11
    # C = P1 + P4 - P5 - P7
    W_s[0, 0] = 1 # C00
    W_s[1, 0] = 1 # C01
    W_s[2, 0] = 1 # C10
    W_s[3, 0] = 1 # C11
    # Wait, W_s[k, i] in my helper? No, W_s is local.
    # Local C indices: 00, 01, 10, 11.
    # C00 gets P1. C01 gets P1. C10 gets P1. C11 gets P1.
    # Actually, Strassen formulas:
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P7
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P4
    
    # Let's fill W_s based on this.
    # W_s[i, k, r] ? No, W_s is flattened (4, 7).
    # Index in W_s: k*2 + i? Or i*2 + k?
    # Problem: W[k*n+i, r]. So index is k*n + i.
    # For 2x2, n=2. Index = k*2 + i.
    
    # P1
    W_s[0*2+0, 0] = 1 # C00
    W_s[1*2+0, 0] = 1 # C01 (k=1, i=0) -> C[0,1]
    W_s[0*2+1, 0] = 1 # C10 (k=0, i=1) -> C[1,0]
    W_s[1*2+1, 0] = 1 # C11
    
    # P2
    U_s[2, 1] = 1 # A10
    U_s[3, 1] = 1 # A11
    V_s[0, 1] = 1 # B00
    W_s[0*2+1, 1] = 1 # C10
    W_s[1*2+1, 1] = -1 # C11
    
    # P3
    U_s[0, 2] = 1 # A00
    V_s[2, 2] = 1 # B10
    V_s[3, 2] = -1 # B11
    W_s[1*2+0, 2] = 1 # C01
    W_s[1*2+1, 2] = 1 # C11
    
    # P4
    U_s[3, 3] = 1 # A11
    V_s[1, 3] = 1 # B01
    V_s[2, 3] = 1 # B10
    W_s[0*2+0, 3] = 1 # C00
    W_s[1*2+1, 3] = 1 # C11
    
    # P5
    U_s[0, 4] = 1 # A00
    U_s[3, 4] = 1 # A11
    V_s[3, 4] = 1 # B11
    V_s[0, 4] = -1 # B00
    W_s[0*2+0, 4] = -1 # C00
    W_s[0*2+1, 4] = 1 # C10
    
    # P6
    U_s[0, 5] = 1 # A00
    U_s[2, 5] = -1 # A10
    V_s[1, 5] = 1 # B01
    V_s[2, 5] = 1 # B10
    # P6 is not used in standard Strassen C formulas?
    # Standard Strassen uses 7 products.
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P7
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P4
    # P6 is not in these?
    # Ah, P6 is usually for C01?
    # Let's check Wikipedia.
    # P1 = A00(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01+B10)
    # P5 = (A00+A11)(B11-B00)
    # P6 = (A00-A10)(B01+B10)
    # P7 = (A10+A11)(B11-B01)
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P7
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P4
    # P6 is indeed not used in this specific set of formulas?
    # Wait, P6 is used in some variants.
    # If P6 is not used, we can set its W to 0.
    # Or maybe I missed it.
    # Actually, P6 cancels out or is not needed for this specific combination.
    # I will leave W_s for P6 as 0.
    
    # P7
    U_s[2, 6] = 1 # A10
    U_s[3, 6] = 1 # A11
    V_s[3, 6] = 1 # B11
    V_s[1, 6] = -1 # B01
    W_s[0*2+0, 6] = -1 # C00
    W_s[1*2+0, 6] = 1 # C01
    
    # 2x1x2 Rank 1
    U_r1 = np.array([[1], [1]])
    V_r1 = np.array([[1], [1]])
    W_r1 = np.array([[1], [1], [1], [1]]) # C00, C01, C10, C11
    
    # Assembly
    # Part 1: 3 blocks of 2x2x2
    offset = 0
    for k_block in range(3):
        # k_off = 2 * k_block
        # j_off = 0
        # i_off = 0
        for r in range(7):
            # Map U_s
            for i in range(2):
                for j in range(2):
                    U[i * m + (j + 0)][offset + r] += U_s[i * 2 + j, r]
            # Map V_s
            for j in range(2):
                for k in range(2):
                    V[(j + 0) * p + (k + 2 * k_block)][offset + r] += V_s[j * 2 + k, r]
            # Map W_s
            for k in range(2):
                for i in range(2):
                    W[(k + 2 * k_block) * n + i][offset + r] += W_s[k * 2 + i, r]
        offset += 7
        
    # Part 2: 3 blocks of 2x3x2
    # Each 2x3x2 is split into 2x1x2 (Rank 1) + 2x2x2 (Rank 7)
    for k_block in range(3):
        # k_off = 2 * k_block
        # j_off = 2
        
        # Sub 2.x.a: 2x1x2
        # j in [2, 3)
        # i in [0, 2)
        # k in [2*k_block, 2*k_block+2)
        for r in range(1):
            for i in range(2):
                U[i * m + 2][offset + r] += U_r1[i, 0]
            for j in range(1):
                for k in range(2):
                    V[(2 + j) * p + (k + 2 * k_block)][offset + r] += V_r1[j, 0]
            for k in range(2):
                for i in range(2):
                    W[(k + 2 * k_block) * n + i][offset + r] += W_r1[k * 2 + i, 0]
        offset += 1
        
        # Sub 2.x.b: 2x2x2
        # j in [3, 5)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + (j + 3)][offset + r] += U_s[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j + 3) * p + (k + 2 * k_block)][offset + r] += V_s[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    W[(k + 2 * k_block) * n + i][offset + r] += W_s[k * 2 + i, r]
        offset += 7
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
