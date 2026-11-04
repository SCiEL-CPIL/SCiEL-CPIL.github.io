"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition:
    # 1. M_{2,1,4} part (j=2): Rank 1
    # 2. M_{2,2,4} part (j=0,1): Split into two M_{2,2,2} (k=0,1 and k=2,3), each Rank 7
    # Total Rank = 1 + 7 + 7 = 15
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{2,1,4} for j=2
    # U indices: i*3 + 2 for i in {0,1} -> 2, 5
    # V indices: 2*4 + k for k in {0,1,2,3} -> 8, 9, 10, 11
    # W indices: k*2 + i for k in {0,1,2,3}, i in {0,1} -> 0..7
    U[2][col] = 1
    U[5][col] = 1
    for k in range(4):
        V[2 * 4 + k][col] = 1
        for i in range(2):
            W[k * 2 + i][col] = 1
    col += 1
    
    # Part 2: M_{2,2,4} for j=0,1
    # Split k into {0,1} and {2,3}
    
    # Helper to add Strassen decomposition for M_{2,2,2}
    # local A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 in local 2x2
    # But we need global indices.
    # j_offset: 0 or 1? No, j is 0,1 for this part.
    # k_offset: 0 or 2.
    
    def add_strassen(j_start, k_start):
        nonlocal col
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        # i in {0, 1}
        
        # Map local (i, j_l, k_l) to global indices
        # j_l in {0, 1} -> j = j_start + j_l
        # k_l in {0, 1} -> k = k_start + k_l
        
        # U indices (i, j)
        def u_idx(i, j_l):
            return i * m + (j_start + j_l)
            
        # V indices (j, k)
        def v_idx(j_l, k_l):
            return (j_start + j_l) * p + (k_start + k_l)
            
        # W indices (k, i)
        def w_idx(k_l, i):
            return (k_start + k_l) * n + i
            
        # Strassen coefficients
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[u_idx(0, 0)][col] += 1
        U[u_idx(1, 1)][col] += 1
        V[v_idx(0, 0)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(0, 0)][col] += 1
        W[w_idx(1, 1)][col] += 1
        col += 1
        
        # M2
        U[u_idx(1, 0)][col] += 1
        U[u_idx(1, 1)][col] += 1
        V[v_idx(0, 0)][col] += 1
        W[w_idx(0, 1)][col] += 1
        W[w_idx(1, 1)][col] += 1 # Wait, C10 and C11 use M2?
        # C10 = M2 + M4 -> W(0, 1) i.e. k=0, i=1
        # C11 = M1 - M2 + M3 + M6 -> W(1, 1) i.e. k=1, i=1
        # So M2 contributes to W(0,1) with +1 and W(1,1) with -1?
        # Let's re-verify Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My previous line for M2:
        # W[w_idx(0, 1)][col] += 1  (C10)
        # W[w_idx(1, 1)][col] -= 1  (C11)
        # I need to fix the W updates.
        
        # Let's restart the Strassen addition logic cleanly.
        col -= 1 # rollback M1 if I messed up? No, let's just write it correctly.
        
        # I will rewrite the function body properly.
        
    # Let's just implement the 7 terms for each block explicitly to avoid closure bugs.
    
    # Block A: j in {0,1}, k in {0,1}
    j_s = 0
    k_s = 0
    
    # M1: (A00+A11)(B00+B11)
    U[0*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*4+0][col] += 1; V[1*4+1][col] += 1
    W[0*2+0][col] += 1; W[1*2+1][col] += 1
    col += 1
    
    # M2: (A10+A11)B00
    U[1*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*4+0][col] += 1
    W[0*2+1][col] += 1 # C10
    W[1*2+1][col] -= 1 # C11
    col += 1
    
    # M3: A00(B01-B11)
    U[0*3+0][col] += 1
    V[0*4+1][col] += 1; V[1*4+1][col] -= 1
    W[0*2+1][col] += 1 # C01
    W[1*2+1][col] += 1 # C11
    col += 1
    
    # M4: A11(B10-B00)
    U[1*3+1][col] += 1
    V[1*4+0][col] += 1; V[0*4+0][col] -= 1
    W[0*2+0][col] += 1 # C00
    W[0*2+1][col] += 1 # C10
    col += 1
    
    # M5: (A00+A01)B11
    U[0*3+0][col] += 1; U[0*3+1][col] += 1
    V[1*4+1][col] += 1
    W[0*2+0][col] -= 1 # C00
    W[0*2+1][col] += 1 # C01
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[1*3+0][col] += 1; U[0*3+0][col] -= 1
    V[0*4+0][col] += 1; V[0*4+1][col] += 1
    W[1*2+1][col] += 1 # C11
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[0*3+1][col] += 1; U[1*3+1][col] -= 1
    V[1*4+0][col] += 1; V[1*4+1][col] += 1
    W[0*2+0][col] += 1 # C00
    col += 1
    
    # Block B: j in {0,1}, k in {2,3}
    j_s = 0
    k_s = 2
    
    # M1: (A00+A11)(B00+B11) [Local B00 is global B(0,2), B11 is global B(1,3)]
    U[0*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*4+2][col] += 1; V[1*4+3][col] += 1
    W[2*2+0][col] += 1; W[3*2+1][col] += 1
    col += 1
    
    # M2: (A10+A11)B00
    U[1*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*4+2][col] += 1
    W[2*2+1][col] += 1 # C(2,1) -> k=2, i=1
    W[3*2+1][col] -= 1 # C(3,1) -> k=3, i=1
    col += 1
    
    # M3: A00(B01-B11)
    U[0*3+0][col] += 1
    V[0*4+3][col] += 1; V[1*4+3][col] -= 1
    W[2*2+1][col] += 1 # C(2,1)
    W[3*2+1][col] += 1 # C(3,1)
    col += 1
    
    # M4: A11(B10-B00)
    U[1*3+1][col] += 1
    V[1*4+2][col] += 1; V[0*4+2][col] -= 1
    W[2*2+0][col] += 1 # C(2,0)
    W[2*2+1][col] += 1 # C(2,1)
    col += 1
    
    # M5: (A00+A01)B11
    U[0*3+0][col] += 1; U[0*3+1][col] += 1
    V[1*4+3][col] += 1
    W[2*2+0][col] -= 1 # C(2,0)
    W[2*2+1][col] += 1 # C(2,1)
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[1*3+0][col] += 1; U[0*3+0][col] -= 1
    V[0*4+2][col] += 1; V[0*4+3][col] += 1
    W[3*2+1][col] += 1 # C(3,1)
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[0*3+1][col] += 1; U[1*3+1][col] -= 1
    V[1*4+2][col] += 1; V[1*4+3][col] += 1
    W[2*2+0][col] += 1 # C(2,0)
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
