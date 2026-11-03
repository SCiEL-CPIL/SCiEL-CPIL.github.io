import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4x7 matrix multiplication.
    
    Strategy: Partition the problem into:
    - 4x4x4 sub-problem using Strassen (rank 49)
    - 4x4x3 sub-problem using schoolbook (rank 48)
    Total rank: 97
    
    U: shape (n*m, rank) = (16, 97)
    V: shape (m*p, rank) = (28, 97)  
    W: shape (p*n, rank) = (28, 97)
    
    Indexing conventions:
    - U[i*m + j, r]: coefficient for A[i,j] in term r
    - V[j*p + k, r]: coefficient for B[j,k] in term r
    - W[k*n + i, r]: coefficient for C[i,k] in term r (trace convention: W[k*n+i, r])
    """
    n, m, p = 4, 4, 7
    
    # Part 1: 4x4x4 Strassen (rank 49)
    # Part 2: 4x4x3 Schoolbook (rank 48)
    total_rank = 49 + 48
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * total_rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * total_rank for _ in range(p * n)]  # 28 x 97
    
    # ===== PART 1: 4x4x4 Strassen (terms 0-48) =====
    # Decompose A(4x4), B(4x4), C(4x4) into 2x2 blocks of size 2x2
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = [[C00, C01], [C10, C11]] where each block is 2x2
    
    # Strassen formulas for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4x4, each Mi is a 2x2 matrix product, which itself needs 7 terms
    # So 7 * 7 = 49 total terms
    
    # Helper functions for global indices
    def a_global(br, bc, lr, lc):
        """Global A index: block (br,bc), local (lr,lc) -> row = br*2+lr, col = bc*2+lc"""
        return (br * 2 + lr) * m + (bc * 2 + lc)
    
    def b_global_4x4(br, bc, lr, lc):
        """Global B index for 4x4 part: block (br,bc), local (lr,lc) -> row = br*2+lr, col = bc*2+lc"""
        # B is 4x4 in this part, so col index is bc*2+lc (0-3)
        return (br * 2 + lr) * p + (bc * 2 + lc)
    
    def c_global(br, bc, lr, lc):
        """Global C index: block (br,bc), local (lr,lc) -> row = br*2+lr, col = bc*2+lc
        W index = k*n + i where k=col, i=row"""
        col = bc * 2 + lc
        row = br * 2 + lr
        return col * n + row
    
    # Strassen 2x2x2 base decomposition (U, V, W for 4x7 each)
    # Local indices: 0=00, 1=01, 2=10, 3=11
    base_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    base_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    base_W = [
        [1, 0, 0, 1, 0, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, 0, 0, -1, 1, -1]  # C11
    ]
    
    # For each Strassen step (7 steps) and each base term (7 terms):
    for step in range(7):
        # Determine which A blocks and B blocks are involved in this Strassen step
        if step == 0:
            # M1 = (A00+A11)(B00+B11)
            a_blocks = { (0,0): 1, (1,1): 1 }
            b_blocks = { (0,0): 1, (1,1): 1 }
        elif step == 1:
            # M2 = (A10+A11)B00
            a_blocks = { (1,0): 1, (1,1): 1 }
            b_blocks = { (0,0): 1 }
        elif step == 2:
            # M3 = A00(B01-B11)
            a_blocks = { (0,0): 1 }
            b_blocks = { (0,1): 1, (1,1): -1 }
        elif step == 3:
            # M4 = A11(B10-B00)
            a_blocks = { (1,1): 1 }
            b_blocks = { (1,0): 1, (0,0): -1 }
        elif step == 4:
            # M5 = (A00+A01)B11
            a_blocks = { (0,0): 1, (0,1): 1 }
            b_blocks = { (1,1): 1 }
        elif step == 5:
            # M6 = (A10-A00)(B00+B01)
            a_blocks = { (1,0): 1, (0,0): -1 }
            b_blocks = { (0,0): 1, (0,1): 1 }
        elif step == 6:
            # M7 = (A01-A11)(B10+B11)
            a_blocks = { (0,1): 1, (1,1): -1 }
            b_blocks = { (1,0): 1, (1,1): 1 }
        
        for base_term in range(7):
            term_idx = step * 7 + base_term
            
            # Fill U: for each local A element, accumulate contributions from relevant blocks
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_u = base_U[local_idx][base_term]
                    if coeff_u == 0:
                        continue
                    for (br, bc), coeff_a in a_blocks.items():
                        if coeff_a == 0:
                            continue
                        g_idx = a_global(br, bc, lr, lc)
                        U[g_idx][term_idx] += coeff_u * coeff_a
            
            # Fill V: for each local B element, accumulate contributions from relevant blocks
            for lr in range(2):
                for lc in range(2):
                    local_idx = lr * 2 + lc
                    coeff_v = base_V[local_idx][base_term]
                    if coeff_v == 0:
                        continue
                    for (br, bc), coeff_b in b_blocks.items():
                        if coeff_b == 0:
                            continue
                        g_idx = b_global_4x4(br, bc, lr, lc)
                        V[g_idx][term_idx] += coeff_v * coeff_b
            
            # Fill W: for each local C element, the output goes to the corresponding C block
            # The Strassen step produces a 2x2 result that gets added to C blocks
            # We need to know which C blocks this step contributes to and with what sign
            # 
            # From Strassen recombination:
            # C00 = M1 - M4 + M6 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M5 - M6
            #
            # So for each step, we know which C blocks it contributes to:
            c_contributions = {
                0: { (0,0): 1, (1,1): 1 },      # M1 -> C00(+1), C11(+1)
                1: { (1,0): 1, (1,1): -1 },      # M2 -> C10(+1), C11(-1)
                2: { (0,1): 1 },                 # M3 -> C01(+1)
                3: { (0,0): -1, (1,0): 1 },      # M4 -> C00(-1), C10(+1)
                4: { (0,1): 1, (1,1): -1 },      # M5 -> C01(+1), C11(-1)
                5: { (0,0): 1, (1,1): -1 },      # M6 -> C00(+1), C11(-1)
                6: { (0,0): -1, (1,1): -1 }      # M7 -> C00(-1), C11(-1)
            }
            
            for (br, bc), coeff_c in c_contributions[step].items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        coeff_w = base_W[local_idx][base_term]
                        if coeff_w == 0:
                            continue
                        g_idx = c_global(br, bc, lr, lc)
                        W[g_idx][term_idx] += coeff_w * coeff_c
    
    # ===== PART 2: 4x4x3 Schoolbook (terms 49-96) =====
    # The remaining 3 columns of B (cols 4,5,6) are handled with schoolbook
    # A is 4x4, B_remaining is 4x3, C_remaining is 4x3
    # Schoolbook: 4*4*3 = 48 terms
    
    offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(3):  # columns 4,5,6 of B
                global_k = k + 4  # actual column index in B (4,5,6)
                
                u_idx = i * m + j
                v_idx = j * p + global_k
                w_idx = global_k * n + i
                
                U[u_idx][offset] = 1
                V[v_idx][offset] = 1
                W[w_idx][offset] = 1
                offset += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
