import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is sum of ranks of sub-blocks
    # 4x4x4: 49
    # Border terms: 4+4+1+4+4+4+4 = 25
    # Total: 74
    
    rank = 74
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # 1. 4x4x4 Block (indices 0..3)
    # Use Kronecker product of Strassen 2x2
    # Strassen 2x2 coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We represent the 2x2 Strassen as a list of (U_list, V_list, W_list)
    # Each is a list of (index, coeff)
    
    def strassen_2x2():
        # Returns list of 7 terms
        # Each term is (U_entries, V_entries, W_entries)
        # U_entries: list of (i, j, coeff)
        # Indices for 2x2: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        
        terms = []
        
        # M1
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (1,1):1
        terms.append([
            [(0, 1), (3, 1)], # U
            [(0, 1), (3, 1)], # V
            [(0, 1), (3, 1)]  # W
        ])
        
        # M2
        # U: (1,0):1, (1,1):1
        # V: (0,0):1
        # W: (0,1):1? No, W index is (k, i).
        # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
        # Standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map W indices (k, i) to (0,0), (0,1), (1,0), (1,1)
        # W for C11: (0,0) and (1,1) are from M1?
        # M1 contributes to C11, C22.
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22.
        # W: C11, C22.
        
        # M2 = (A21+A22)B11
        # U: A21, A22. V: B11.
        # W: C21, C22.
        
        # M3 = A11(B12-B22)
        # U: A11. V: B12, -B22.
        # W: C11, C12.
        
        # M4 = A22(B21-B11)
        # U: A22. V: B21, -B11.
        # W: C11, C21.
        
        # M5 = (A11+A12)B22
        # U: A11, A12. V: B22.
        # W: C12, C22.
        
        # M6 = (A21-A11)(B11+B12)
        # U: A21, -A11. V: B11, B12.
        # W: C21, C22.
        
        # M7 = (A12-A22)(B21+B22)
        # U: A12, -A22. V: B21, B22.
        # W: C11, C12.
        
        # Let's re-verify W assignments.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 1 (M1):
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1 (C11), (1,1):1 (C22)
        terms[0] = [
            [(0, 1), (3, 1)],
            [(0, 1), (3, 1)],
            [(0, 1), (3, 1)]
        ]
        
        # Term 2 (M2):
        # U: (1,0):1, (1,1):1
        # V: (0,0):1
        # W: (1,0):1 (C21), (1,1):-1 (C22)
        terms.append([
            [(2, 1), (3, 1)],
            [(0, 1)],
            [(2, 1), (3, -1)]
        ])
        
        # Term 3 (M3):
        # U: (0,0):1
        # V: (0,1):1, (1,1):-1
        # W: (0,1):1 (C12), (1,1):1 (C22) -- Wait, M3 is in C12 and C22?
        # C12 = M3 + M5. C22 = M1 - M2 + M3 + M6.
        # Yes.
        # W indices: C12 is (0,1), C22 is (1,1).
        terms.append([
            [(0, 1)],
            [(1, 1), (3, -1)],
            [(1, 1), (3, 1)]
        ])
        
        # Term 4 (M4):
        # U: (1,1):1
        # V: (1,0):1, (0,0):-1
        # W: (0,0):1 (C11), (1,0):1 (C21)
        terms.append([
            [(3, 1)],
            [(2, 1), (0, -1)],
            [(0, 1), (2, 1)]
        ])
        
        # Term 5 (M5):
        # U: (0,0):1, (0,1):1
        # V: (1,1):1
        # W: (0,1):1 (C12), (1,1):-1 (C22) -- Wait, C22 has -M2, +M1, +M3, +M6. M5 is not in C22.
        # C12 = M3 + M5.
        # C22 = M1 - M2 + M3 + M6.
        # So M5 is only in C12?
        # Let's check C22 again.
        # C22 = A21B12 + A22B22.
        # M1 = A11B11 + A11B22 + A22B11 + A22B22.
        # M2 = A21B11 + A22B11.
        # M3 = A11B12 - A11B22.
        # M6 = A21B11 + A21B12 - A11B11 - A11B12.
        # M1 - M2 + M3 + M6 = (A11B11 + A11B22 + A22B11 + A22B22) - (A21B11 + A22B11) + (A11B12 - A11B22) + (A21B11 + A21B12 - A11B11 - A11B12)
        # = A22B22 + A21B12. Correct.
        # So M5 is NOT in C22.
        # M5 = (A11+A12)B22 = A11B22 + A12B22.
        # C12 = A11B12 + A12B22.
        # M3 + M5 = (A11B12 - A11B22) + (A11B22 + A12B22) = A11B12 + A12B22. Correct.
        # So M5 is only in C12.
        terms.append([
            [(0, 1), (1, 1)],
            [(3, 1)],
            [(1, 1)]
        ])
        
        # Term 6 (M6):
        # U: (1,0):1, (0,0):-1
        # V: (0,0):1, (0,1):1
        # W: (1,0):1 (C21), (1,1):1 (C22)
        terms.append([
            [(2, 1), (0, -1)],
            [(0, 1), (1, 1)],
            [(2, 1), (3, 1)]
        ])
        
        # Term 7 (M7):
        # U: (0,1):1, (1,1):-1
        # V: (1,0):1, (1,1):1
        # W: (0,0):1 (C11), (0,1):1 (C12)
        terms.append([
            [(1, 1), (3, -1)],
            [(2, 1), (3, 1)],
            [(0, 1), (1, 1)]
        ])
        
        return terms

    s2_terms = strassen_2x2()
    
    # Kronecker Product
    # For each term in outer (7) and inner (7), we create a term for 4x4.
    # Indices:
    # 4x4 index (I, J) where I, J in 0..3.
    # I = 2*i1 + i2, J = 2*j1 + j2.
    # Outer term uses (i1, j1) for A, (j1, k1) for B, (k1, i1) for C.
    # Inner term uses (i2, j2) for A, (j2, k2) for B, (k2, i2) for C.
    # Combined index:
    # A: (2*i1 + i2, 2*j1 + j2)
    # B: (2*j1 + j2, 2*k1 + k2)
    # C: (2*k1 + k2, 2*i1 + i2)
    
    for outer in s2_terms:
        u_outer, v_outer, w_outer = outer
        for inner in s2_terms:
            u_inner, v_inner, w_inner = inner
            
            # Combine entries
            u_entries = []
            v_entries = []
            w_entries = []
            
            coeff_product = 1.0
            # We need to multiply coefficients
            # Since each entry has a coeff, we multiply them.
            
            for (i1, c1) in u_outer:
                for (i2, c2) in u_inner:
                    # i1 is (r1, c1_idx), i2 is (r2, c2_idx)
                    # Wait, my u_outer entries are (index, coeff).
                    # index is 0..3 for 2x2.
                    # We need to map 2x2 index to (r, c).
                    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    r1, c1_idx = divmod(i1, 2)
                    r2, c2_idx = divmod(i2, 2)
                    I = 2*r1 + r2
                    J = 2*c1_idx + c2_idx
                    u_entries.append(((I, J), c1 * c2))
                    
            for (j1, c1) in v_outer:
                for (j2, c2) in v_inner:
                    r1, c1_idx = divmod(j1, 2)
                    r2, c2_idx = divmod(j2, 2)
                    J = 2*r1 + r2
                    K = 2*c1_idx + c2_idx
                    v_entries.append(((J, K), c1 * c2))
                    
            for (k1, c1) in w_outer:
                for (k2, c2) in w_inner:
                    r1, c1_idx = divmod(k1, 2)
                    r2, c2_idx = divmod(k2, 2)
                    K = 2*r1 + r2
                    I = 2*c1_idx + r2
                    w_entries.append(((K, I), c1 * c2))
            
            # Add to global U, V, W
            # We need to merge entries with same indices?
            # In Kronecker product, indices are unique for each pair.
            # So no merging needed.
            
            for (idx, val) in u_entries:
                U[idx[0] * 5 + idx[1]][col] += val
            for (idx, val) in v_entries:
                V[idx[0] * 5 + idx[1]][col] += val
            for (idx, val) in w_entries:
                W[idx[0] * 5 + idx[1]][col] += val
            col += 1

    # 2. Border Terms
    # Col 5 (index 4), Rows 0..3
    # C_{i,4} = sum_{k=0}^3 A_{i,k} B_{k,4}
    for k in range(4):
        for i in range(4):
            U[i * 5 + k][col] = 1
        V[k * 5 + 4][col] = 1
        for i in range(4):
            W[4 * 5 + i][col] = 1
        col += 1

    # Row 5 (index 4), Cols 0..3
    # C_{4,j} = sum_{k=0}^3 A_{4,k} B_{k,j}
    for k in range(4):
        U[4 * 5 + k][col] = 1
        for j in range(4):
            V[k * 5 + j][col] = 1
        W[k * 5 + 4][col] = 1
        col += 1

    # C_{4,4} = A_{4,4} B_{4,4} + sum_{k=0}^3 A_{4,k} B_{k,4}
    # Term 1: A_{4,4} B_{4,4}
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1
    
    # Term 2: sum_{k=0}^3 A_{4,k} B_{k,4}
    # This is similar to Col 5 but for Row 5?
    # C_{4,4} gets contribution from A_{4,k} B_{k,4}.
    for k in range(4):
        U[4 * 5 + k][col] = 1
        V[k * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1

    # C_{0..3, 4} also gets contribution from A_{0..3, 4} B_{4, 0..3}?
    # No, C_{i,4} = sum_{k=0}^4 A_{i,k} B_{k,4}.
    # We handled k=0..3 above.
    # We need k=4: A_{i,4} B_{4,4}.
    for i in range(4):
        U[i * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    for i in range(4):
        W[4 * 5 + i][col] = 1
    col += 1

    # C_{4, 0..3} also gets contribution from A_{4, 4} B_{4, 0..3}?
    # C_{4,j} = sum_{k=0}^4 A_{4,k} B_{k,j}.
    # We handled k=0..3 above.
    # We need k=4: A_{4,4} B_{4,j}.
    U[4 * 5 + 4][col] = 1
    for j in range(4):
        V[4 * 5 + j][col] = 1
    for j in range(4):
        W[j * 5 + 4][col] = 1
    col += 1

    # Check col
    # 49 + 4 + 4 + 1 + 4 + 4 + 4 = 70?
    # My count:
    # 4x4: 49
    # Col 5 (k=0..3): 4
    # Row 5 (k=0..3): 4
    # C44 (A44B44): 1
    # C44 (sum k=0..3): 4
    # Col 5 (k=4): 4
    # Row 5 (k=4): 4
    # Total: 49 + 4 + 4 + 1 + 4 + 4 + 4 = 70.
    
    # Wait, my rank estimate was 74.
    # Let's re-verify the border terms.
    # C_{i,4} (i=0..3):
    #   k=0..3: 4 terms.
    #   k=4: 1 term?
    #   A_{i,4} B_{4,4}.
    #   U: (i,4). V: (4,4). W: (4,i).
    #   This is 1 term for all i?
    #   No, U must be a vector.
    #   If we use 1 term, U has 1s at (0,4)..(3,4). V has 1 at (4,4). W has 1s at (4,0)..(4,3).
    #   Then C_{i,4} gets A_{i,4} * 1 * 1 = A_{i,4}.
    #   But we need A_{i,4} B_{4,4}.
    #   So V must be B_{4,4}.
    #   So yes, 1 term suffices for the k=4 part of C_{0..3, 4}.
    #   Similarly for C_{4, 0..3}, 1 term suffices for k=4.
    
    # So:
    # Col 5 (k=0..3): 4 terms.
    # Col 5 (k=4): 1 term.
    # Row 5 (k=0..3): 4 terms.
    # Row 5 (k=4): 1 term.
    # C44 (A44B44): 1 term.
    # C44 (sum k=0..3): 4 terms.
    
    # Total Border: 4 + 1 + 4 + 1 + 1 + 4 = 15.
    # Total Rank: 49 + 15 = 64.
    
    # Let's adjust the code to use 1 term for the k=4 parts.
    
    # Re-do border terms with correct count
    
    # Reset col to 49
    col = 49
    
    # Col 5 (k=0..3)
    for k in range(4):
        for i in range(4):
            U[i * 5 + k][col] = 1
        V[k * 5 + 4][col] = 1
        for i in range(4):
            W[4 * 5 + i][col] = 1
        col += 1
        
    # Col 5 (k=4)
    for i in range(4):
        U[i * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    for i in range(4):
        W[4 * 5 + i][col] = 1
    col += 1

    # Row 5 (k=0..3)
    for k in range(4):
        U[4 * 5 + k][col] = 1
        for j in range(4):
            V[k * 5 + j][col] = 1
        W[k * 5 + 4][col] = 1
        col += 1

    # Row 5 (k=4)
    U[4 * 5 + 4][col] = 1
    for j in range(4):
        V[4 * 5 + j][col] = 1
    for j in range(4):
        W[j * 5 + 4][col] = 1
    col += 1

    # C44 (A44B44)
    U[4 * 5 + 4][col] = 1
    V[4 * 5 + 4][col] = 1
    W[4 * 5 + 4][col] = 1
    col += 1

    # C44 (sum k=0..3)
    for k in range(4):
        U[4 * 5 + k][col] = 1
        V[k * 5 + 4][col] = 1
        W[4 * 5 + 4][col] = 1
        col += 1

    # Total rank used: 49 + 4 + 1 + 4 + 1 + 1 + 4 = 64.
    
    # Trim U, V, W to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
