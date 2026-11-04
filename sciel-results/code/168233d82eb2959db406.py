import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Rank 65 decomposition:
    # Part 1: 4x4x4 using Recursive Strassen (rank 49)
    # Part 2: 4x4x1 using schoolbook (rank 16)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]   # (16, 65)
    V = [[0] * rank for _ in range(m * p)]   # (20, 65)
    W = [[0] * rank for _ in range(p * n)]   # (20, 65)
    
    col = 0
    
    # Helper to add a 2x2 Strassen decomposition with mapped indices
    def add_2x2_strassen_mapped(a_map, b_map, c_map, col_idx):
        # a_map, b_map, c_map are dicts: (r,c) -> list of (global_idx, coeff)
        
        # M1: (A00+A11)(B00+B11)
        # U: x00+x11, V: y00+y11, W: z00+z11
        for (r, c) in [(0,0), (1,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col_idx] += 1 * mult
        for (r, c) in [(0,0), (1,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col_idx] += 1 * mult
        for (r, c) in [(0,0), (1,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col_idx] += 1 * mult
        col_idx += 1
        
        # M2: (A10+A11)B00
        # U: x10+x11, V: y00, W: z10 - z11 (based on current code's pattern? 
        # Let's stick to the standard Strassen W coefficients for the 2x2 block output)
        # Standard 2x2 Strassen W coefficients:
        # M1: C00+1, C11+1
        # M2: C10+1, C11-1  <-- Wait, current code used this. 
        # Let's re-verify if current code's W pattern is standard or custom.
        # Current code M2: W[c10]+=1, W[c11]-=1.
        # If I use this, I must ensure the 4x4 level C_map accounts for this.
        # The 2x2 Strassen produces a 2x2 matrix M.
        # The entries of M are the local C00, C01, C10, C11.
        # So the W updates in add_2x2_strassen_mapped should define the local C matrix.
        # Then c_map distributes the local C to global C.
        
        # So, for M2, the local C is:
        # C10 += 1, C11 -= 1? 
        # This implies the 2x2 Strassen decomposition in the current code is NOT the standard one that outputs the product matrix directly, 
        # but one where the W coefficients are already mixed?
        # No, the W coefficients in the 2x2 decomposition *are* the product matrix entries.
        # If the current code says M2 contributes -1 to C11 of the 2x2 block, then the 2x2 block product has that property.
        # I will replicate the *exact* W updates from the current code's M1-M7 to be safe.
        
        # Reset col_idx logic: I should increment col_idx 7 times total.
        # The function above incremented it once. I need to do this for all 7.
        
        # Let's rewrite the helper to take col_start and return col_end, or just use nonlocal col.
        pass

    # Correct implementation of helper
    def add_2x2_strassen_mapped(a_map, b_map, c_map):
        nonlocal col
        
        # M1
        for (r, c) in [(0,0), (1,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(0,0), (1,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(0,0), (1,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        col += 1
        
        # M2
        for (r, c) in [(1,0), (1,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(0,0)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(1,0)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        for (r, c) in [(1,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] -= 1 * mult
        col += 1
        
        # M3
        for (r, c) in [(0,0)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(0,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(1,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] -= 1 * mult
        for (r, c) in [(0,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        for (r, c) in [(1,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        col += 1
        
        # M4
        for (r, c) in [(1,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(1,0)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(0,0)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] -= 1 * mult
        for (r, c) in [(0,0)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        for (r, c) in [(1,0)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        col += 1
        
        # M5
        for (r, c) in [(0,0), (0,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(1,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(0,0)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] -= 1 * mult
        for (r, c) in [(0,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        col += 1
        
        # M6
        for (r, c) in [(1,0)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(0,0)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] -= 1 * mult
        for (r, c) in [(0,0), (0,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(1,1)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        col += 1
        
        # M7
        for (r, c) in [(0,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] += 1 * mult
        for (r, c) in [(1,1)]:
            for (idx, mult) in a_map[r,c]:
                U[idx][col] -= 1 * mult
        for (r, c) in [(1,0), (1,1)]:
            for (idx, mult) in b_map[r,c]:
                V[idx][col] += 1 * mult
        for (r, c) in [(0,0)]:
            for (idx, mult) in c_map[r,c]:
                W[idx][col] += 1 * mult
        col += 1

    # Define block index helpers
    # A is 4x4. Blocks are 2x2.
    # A00: r 0-1, c 0-1. Indices: 0,1,4,5
    # A01: r 0-1, c 2-3. Indices: 2,3,6,7
    # A10: r 2-3, c 0-1. Indices: 8,9,12,13
    # A11: r 2-3, c 2-3. Indices: 10,11,14,15
    
    def get_a_map(coeffs):
        # coeffs: 2x2 list of coefficients for A00, A01, A10, A11
        # Returns dict (r,c) -> [(idx, mult)]
        m = {}
        # A00
        if coeffs[0][0] != 0:
            m[(0,0)] = m.get((0,0), []) + [(0, coeffs[0][0])]
            m[(0,1)] = m.get((0,1), []) + [(1, coeffs[0][0])]
            m[(1,0)] = m.get((1,0), []) + [(4, coeffs[0][0])]
            m[(1,1)] = m.get((1,1), []) + [(5, coeffs[0][0])]
        # A01
        if coeffs[0][1] != 0:
            m[(0,0)] = m.get((0,0), []) + [(2, coeffs[0][1])]
            m[(0,1)] = m.get((0,1), []) + [(3, coeffs[0][1])]
            m[(1,0)] = m.get((1,0), []) + [(6, coeffs[0][1])]
            m[(1,1)] = m.get((1,1), []) + [(7, coeffs[0][1])]
        # A10
        if coeffs[1][0] != 0:
            m[(0,0)] = m.get((0,0), []) + [(8, coeffs[1][0])]
            m[(0,1)] = m.get((0,1), []) + [(9, coeffs[1][0])]
            m[(1,0)] = m.get((1,0), []) + [(12, coeffs[1][0])]
            m[(1,1)] = m.get((1,1), []) + [(13, coeffs[1][0])]
        # A11
        if coeffs[1][1] != 0:
            m[(0,0)] = m.get((0,0), []) + [(10, coeffs[1][1])]
            m[(0,1)] = m.get((0,1), []) + [(11, coeffs[1][1])]
            m[(1,0)] = m.get((1,0), []) + [(14, coeffs[1][1])]
            m[(1,1)] = m.get((1,1), []) + [(15, coeffs[1][1])]
        # Ensure all keys exist
        for r in range(2):
            for c in range(2):
                if (r,c) not in m:
                    m[(r,c)] = []
        return m

    # B is 4x5. 
    # Wait, B is 4x5. The Strassen applies to the 4x4 part of B?
    # The task is 4x4 * 4x5.
    # We can split B into B_left (4x4) and B_right (4x1).
    # The 4x4 part of B (cols 0-3) can use Strassen.
    # The 4x1 part (col 4) uses schoolbook.
    
    # B indices for 4x4 part (cols 0-3):
    # B00: r 0-1, c 0-1. Indices: 0,1, 5,6
    # B01: r 0-1, c 2-3. Indices: 2,3, 7,8
    # B10: r 2-3, c 0-1. Indices: 10,11, 15,16
    # B11: r 2-3, c 2-3. Indices: 12,13, 17,18
    
    def get_b_map(coeffs):
        m = {}
        if coeffs[0][0] != 0:
            m[(0,0)] = m.get((0,0), []) + [(0, coeffs[0][0])]
            m[(0,1)] = m.get((0,1), []) + [(1, coeffs[0][0])]
            m[(1,0)] = m.get((1,0), []) + [(5, coeffs[0][0])]
            m[(1,1)] = m.get((1,1), []) + [(6, coeffs[0][0])]
        if coeffs[0][1] != 0:
            m[(0,0)] = m.get((0,0), []) + [(2, coeffs[0][1])]
            m[(0,1)] = m.get((0,1), []) + [(3, coeffs[0][1])]
            m[(1,0)] = m.get((1,0), []) + [(7, coeffs[0][1])]
            m[(1,1)] = m.get((1,1), []) + [(8, coeffs[0][1])]
        if coeffs[1][0] != 0:
            m[(0,0)] = m.get((0,0), []) + [(10, coeffs[1][0])]
            m[(0,1)] = m.get((0,1), []) + [(11, coeffs[1][0])]
            m[(1,0)] = m.get((1,0), []) + [(15, coeffs[1][0])]
            m[(1,1)] = m.get((1,1), []) + [(16, coeffs[1][0])]
        if coeffs[1][1] != 0:
            m[(0,0)] = m.get((0,0), []) + [(12, coeffs[1][1])]
            m[(0,1)] = m.get((0,1), []) + [(13, coeffs[1][1])]
            m[(1,0)] = m.get((1,0), []) + [(17, coeffs[1][1])]
            m[(1,1)] = m.get((1,1), []) + [(18, coeffs[1][1])]
        for r in range(2):
            for c in range(2):
                if (r,c) not in m:
                    m[(r,c)] = []
        return m

    # C is 4x5. W index is k*n + i = col*4 + row.
    # C 4x4 part (cols 0-3):
    # C00: col 0-1, row 0-1. 
    # W indices: 
    # (0,0): 0*4+0=0. (0,1): 0*4+1=1. (1,0): 1*4+0=4. (1,1): 1*4+1=5.
    # C01: col 2-3, row 0-1.
    # (0,0): 2*4+0=8. (0,1): 2*4+1=9. (1,0): 3*4+0=12. (1,1): 3*4+1=13.
    # C10: col 0-1, row 2-3.
    # (0,0): 0*4+2=2. (0,1): 0*4+3=3. (1,0): 1*4+2=6. (1,1): 1*4+3=7.
    # C11: col 2-3, row 2-3.
    # (0,0): 2*4+2=10. (0,1): 2*4+3=11. (1,0): 3*4+2=14. (1,1): 3*4+3=15.
    
    def get_c_map(coeffs):
        # coeffs: 2x2 list for C00, C01, C10, C11
        m = {}
        if coeffs[0][0] != 0:
            m[(0,0)] = m.get((0,0), []) + [(0, coeffs[0][0])]
            m[(0,1)] = m.get((0,1), []) + [(1, coeffs[0][0])]
            m[(1,0)] = m.get((1,0), []) + [(4, coeffs[0][0])]
            m[(1,1)] = m.get((1,1), []) + [(5, coeffs[0][0])]
        if coeffs[0][1] != 0:
            m[(0,0)] = m.get((0,0), []) + [(8, coeffs[0][1])]
            m[(0,1)] = m.get((0,1), []) + [(9, coeffs[0][1])]
            m[(1,0)] = m.get((1,0), []) + [(12, coeffs[0][1])]
            m[(1,1)] = m.get((1,1), []) + [(13, coeffs[0][1])]
        if coeffs[1][0] != 0:
            m[(0,0)] = m.get((0,0), []) + [(2, coeffs[1][0])]
            m[(0,1)] = m.get((0,1), []) + [(3, coeffs[1][0])]
            m[(1,0)] = m.get((1,0), []) + [(6, coeffs[1][0])]
            m[(1,1)] = m.get((1,1), []) + [(7, coeffs[1][0])]
        if coeffs[1][1] != 0:
            m[(0,0)] = m.get((0,0), []) + [(10, coeffs[1][1])]
            m[(0,1)] = m.get((0,1), []) + [(11, coeffs[1][1])]
            m[(1,0)] = m.get((1,0), []) + [(14, coeffs[1][1])]
            m[(1,1)] = m.get((1,1), []) + [(15, coeffs[1][1])]
        for r in range(2):
            for c in range(2):
                if (r,c) not in m:
                    m[(r,c)] = []
        return m

    # 7 Strassen steps for 4x4
    # M1
    add_2x2_strassen_mapped(
        get_a_map([[1,0],[0,1]]), 
        get_b_map([[1,0],[0,1]]), 
        get_c_map([[1,0],[0,1]])
    )
    
    # M2
    add_2x2_strassen_mapped(
        get_a_map([[0,0],[1,1]]), 
        get_b_map([[1,0],[0,0]]), 
        get_c_map([[0,0],[0,1]]) # C10
    )
    
    # M3
    add_2x2_strassen_mapped(
        get_a_map([[1,0],[0,0]]), 
        get_b_map([[0,1],[0,-1]]), 
        get_c_map([[0,0],[0,-1]]) # -C11
    )
    
    # M4
    add_2x2_strassen_mapped(
        get_a_map([[0,0],[0,1]]), 
        get_b_map([[0,0],[1,-1]]), 
        get_c_map([[0,0],[0,1]]) # C10
    )
    
    # M5
    add_2x2_strassen_mapped(
        get_a_map([[1,1],[0,0]]), 
        get_b_map([[0,0],[0,1]]), 
        get_c_map([[-1,1],[0,0]]) # -C00 + C01
    )
    
    # M6
    add_2x2_strassen_mapped(
        get_a_map([[0,0],[1,-1]]), 
        get_b_map([[1,1],[0,0]]), 
        get_c_map([[1,0],[0,0]]) # C00
    )
    
    # M7
    add_2x2_strassen_mapped(
        get_a_map([[0,1],[0,-1]]), 
        get_b_map([[0,0],[1,1]]), 
        get_c_map([[0,1],[0,1]]) # C01 + C11
    )

    # Part 2: 4x4x1 schoolbook (rank 16)
    if time.time() < deadline - 2.0:
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 5 + 4][col] = 1
                W[4 * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
