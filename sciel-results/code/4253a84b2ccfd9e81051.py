import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Pre-computed rank-36 decomposition for 2x4x5
    # Structure: Split m=4 into 2+2.
    # For each half, we have 2x2x5.
    # 2x2x5 split into 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11? 
    # Wait, 2x2x2 is rank 7. 2x2x1 is rank 4. Total 11.
    # Two halves: 11 * 2 = 22? 
    # No, the previous code used 2x2x2 (rank 7) for the 2x2 blocks of B?
    # Let's re-evaluate the rank calculation in the previous code.
    
    # Previous code:
    # "strassen_terms" has 7 terms.
    # "standard_terms_2x2x1" has 4 terms.
    # Loop:
    # for q in range(2): # Split A columns into 2 blocks of 2
    #   for s in range(3): # Split B columns into 3 blocks: 2, 2, 1
    #     if s < 2: # First two blocks of B are 2 wide
    #       Use Strassen (7 terms)
    #     else: # Last block of B is 1 wide
    #       Use Standard (4 terms)
    
    # So for each A-block (q), we process B-blocks s=0,1,2.
    # s=0 (2 cols): 7 terms
    # s=1 (2 cols): 7 terms
    # s=2 (1 col): 4 terms
    # Total per A-block: 18 terms.
    # Two A-blocks: 36 terms.
    
    # This confirms the rank is 36.
    
    # Can we do better?
    # If we used a rank-10 decomposition for 2x2x3 (covering s=0 and s=1 together? No, s=0 and s=1 are 2+2=4 cols.
    # 2x2x4 rank is 14 (7+7).
    # 2x2x1 rank is 4.
    # Total 18.
    
    # If we treated B as 2x5 directly.
    # R(2,2,5) = 17 (using 2x2x3 rank 10 + 2x2x2 rank 7).
    # Then total rank 17*2 = 34.
    
    # I will implement the 36 solution as it is safer.
    
    def strassen_block_solution():
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # 2x2x2 Strassen terms (indices 0-3 for A, 0-3 for B, 0-3 for C)
        # A idx: 00=0, 01=1, 10=2, 11=3
        # B idx: 00=0, 01=1, 10=2, 11=3
        # C idx: 00=0, 01=1, 10=2, 11=3
        
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00+=, C11-=, C10-=, C01+=? 
            # Standard Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 - M6
            
            # Term 1: M1
            # U: {0:1, 3:1} (A00+A11)
            # V: {0:1, 3:1} (B00+B11)
            # W: {0:1, 3:-1, 2:-1, 1:1} ? 
            # Let's map W carefully.
            # W[k*n+i]. C00->idx 0, C01->idx 1, C10->idx 2, C11->idx 3.
            # C00 += M1 => W[0] += 1
            # C11 -= M1 => W[3] -= 1
            # C10 -= M1? No, C10 = M2+M4. C11 = M1-M2+M3-M6.
            # Wait, standard Strassen formulas:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 - M6
            
            # So M1 appears in C00 (+1) and C11 (+1).
            # W coeffs for M1: {0: 1, 3: 1}
            
            # M2 appears in C10 (+1) and C11 (-1).
            # W coeffs for M2: {2: 1, 3: -1}
            # U: {2:1, 3:1} (A10+A11)
            # V: {0:1} (B00)
            
            # M3 appears in C01 (+1) and C11 (+1).
            # W coeffs for M3: {1: 1, 3: 1}
            # U: {0:1} (A00)
            # V: {2:1, 3:-1} (B10-B11)
            
            # M4 appears in C00 (+1) and C10 (+1).
            # W coeffs for M4: {0: 1, 2: 1}
            # U: {3:1} (A11)
            # V: {1:1, 0:-1} (B01-B00)
            
            # M5 appears in C00 (-1) and C01 (+1).
            # W coeffs for M5: {0: -1, 1: 1}
            # U: {0:1, 1:1} (A00+A01)
            # V: {3:1} (B11)
            
            # M6 appears in C11 (-1).
            # W coeffs for M6: {3: -1}
            # U: {2:1, 0:-1} (A10-A00)
            # V: {0:1, 1:1} (B00+B01)
            
            # M7 appears in C00 (+1).
            # W coeffs for M7: {0: 1}
            # U: {1:1, 3:-1} (A01-A11)
            # V: {2:1, 3:1} (B10+B11)
            
        ]
        
        # Corrected Strassen terms based on standard formulas
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
            ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),       # M3
            ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),       # M4
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: -1}),      # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})        # M7
        ]
        
        standard_terms_2x2x1 = [
            ({0: 1}, {0: 1}, {0: 1}), # A00 B0 -> C0
            ({1: 1}, {1: 1}, {0: 1}), # A01 B1 -> C0
            ({2: 1}, {0: 1}, {1: 1}), # A10 B0 -> C1
            ({3: 1}, {1: 1}, {1: 1})  # A11 B1 -> C1
        ]
        
        # Map local indices to global
        # A: 2 rows, 4 cols. Split into q=0 (cols 0,1), q=1 (cols 2,3)
        # B: 4 rows, 5 cols. Split rows into s=0 (rows 0,1), s=1 (rows 2,3).
        # B cols: s=0 uses cols 0,1 (Strassen), s=1 uses cols 2,3 (Strassen), s=2 uses col 4 (Standard)
        # Wait, B is 4x5.
        # The inner dimension is 4.
        # So B is split into two blocks of 2 rows each?
        # Yes, j=0,1 and j=2,3.
        # For each block, B is 2x5.
        # We split the 5 columns of B into 2, 2, 1.
        
        for q in range(2):
            # q=0: A cols 0,1. q=1: A cols 2,3.
            for s in range(3):
                # s=0: B rows 0,1, cols 0,1 (2x2 block) -> Strassen
                # s=1: B rows 0,1, cols 2,3 (2x2 block) -> Strassen
                # s=2: B rows 0,1, cols 4 (2x1 block) -> Standard
                
                # Wait, the loop `for s in range(3)` in the original code
                # combined s (B row block) and B col block?
                # Let's re-read the original code logic.
                
                # Original:
                # for q in range(2): # A col block
                #   for s in range(3): # ?
                #     if s < 2: # Strassen
                #     else: # Standard
                
                # The original code mapped s to B row blocks AND B col blocks?
                # No, `v_global` used `gk = 2*s + lk`.
                # If s=0, gk=0,1. If s=1, gk=2,3. If s=2, gk=4.
                # And `gj = 2*q + lj`? No, `gj` was for A?
                # `u_global`: `gi = li`, `gj = 2*q + lj`.
                # So q determines the A column block.
                # s determines the B column block?
                # But B also has rows.
                # `v_global`: `gj = 2*q + lj`? No.
                # Original code:
                # `lj = idx // 2` (B row local)
                # `lk = idx % 2` (B col local)
                # `gj = 2*q + lj` -> This maps B row to global row?
                # `gk = 2*s + lk` -> This maps B col to global col?
                
                # This implies:
                # q selects the B row block?
                # s selects the B col block?
                # And A is not split by q in the same way?
                # `u_global`: `gi = li` (A row), `gj = 2*q + lj` (A col).
                # So q splits A columns AND B rows?
                # This means we are doing block matrix multiplication:
                # A is 2x4. Split cols into 2 blocks of 2.
                # B is 4x5. Split rows into 2 blocks of 2.
                # The product $A_q B_q$ is defined for matching q.
                # So we have two independent sub-problems:
                # 1. A[:, 0:2] * B[0:2, :]
                # 2. A[:, 2:4] * B[2:4, :]
                
                # For each sub-problem, A is 2x2, B is 2x5.
                # We split B cols (5) into 2, 2, 1.
                # s=0: B cols 0,1 (2x2) -> Strassen
                # s=1: B cols 2,3 (2x2) -> Strassen
                # s=2: B cols 4 (2x1) -> Standard
                
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_global = {}
                        v_global = {}
                        w_global = {}
                        
                        for idx, val in u_t.items():
                            li = idx // 2
                            lj = idx % 2
                            gi = li
                            gj = 2*q + lj
                            u_global[gi * m + gj] = val
                        
                        for idx, val in v_t.items():
                            lj = idx // 2
                            lk = idx % 2
                            gj = 2*q + lj # B row
                            gk = 2*s + lk # B col
                            v_global[gj * p + gk] = val
                        
                        for idx, val in w_t.items():
                            li = idx // 2
                            lk = idx % 2
                            gi = li
                            gk = 2*s + lk
                            w_global[gk * n + gi] = val
                        
                        add_term(u_global, v_global, w_global)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_global = {}
                        v_global = {}
                        w_global = {}
                        
                        for idx, val in u_t.items():
                            li = idx // 2
                            lj = idx % 2
                            gi = li
                            gj = 2*q + lj
                            u_global[gi * m + gj] = val
                        
                        for idx, val in v_t.items():
                            lj = idx # 0 or 1
                            gk = 4
                            gj = 2*q + lj
                            v_global[gj * p + gk] = val
                        
                        for idx, val in w_t.items():
                            li = idx # 0 or 1
                            gk = 4
                            gi = li
                            w_global[gk * n + gi] = val
                        
                        add_term(u_global, v_global, w_global)

        return {"U": U, "V": V, "W": W}

    return strassen_block_solution()
# EVOLVE-BLOCK-END
