import numpy as np

def construct():
    n, m, p = 3, 4, 7
    # We will build the decomposition using blocks.
    # Total rank estimate: 78
    
    # We use lists to accumulate U, V, W rows/columns
    # U is (n*m, rank)
    # V is (m*p, rank)
    # W is (p*n, rank)
    
    # Since rank is dynamic, we'll collect terms as tuples of (u_vec, v_vec, w_vec)
    # where u_vec is length n*m, v_vec is length m*p, w_vec is length p*n
    # Then we stack them.
    
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # Create sparse vectors
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        
        u_vec[u_idx] += u_val
        v_vec[v_idx] += v_val
        w_vec[w_idx] += w_val
        
        terms.append((u_vec, v_vec, w_vec))

    # Helper to add a full rank-1 tensor for a sub-problem
    # This is inefficient, but rank is small (78).
    # Better: just accumulate into global arrays?
    # No, we don't know rank ahead of time exactly? We do: 78.
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a decomposition of M_{3,2,2}
    # A is 3x2, B is 2x2, C is 3x2
    # A rows: i in 0..2, cols j in 0..1
    # B rows: j in 0..1, cols k in 0..1
    # C rows: i in 0..2, cols k in 0..1
    # Indices in global:
    # A offset: (i_off, j_off)
    # B offset: (j_off, k_off)
    # C offset: (k_off, i_off)
    
    def add_M322(i_off, j_off, k_off):
        nonlocal col
        # Split A (3x2) into A1 (2x2) and A2 (1x2)
        # A1 rows: i_off, i_off+1. Cols: j_off, j_off+1
        # A2 row: i_off+2. Cols: j_off, j_off+1
        
        # Part 1: M_{2,2,2} for A1 * B
        # Strassen
        # P1 = (A11+A22)(B11+B22)
        # Indices:
        # A11: (i_off, j_off), A22: (i_off+1, j_off+1)
        # B11: (j_off, k_off), B22: (j_off+1, k_off+1)
        
        # We need to map global indices to local 0..1 for A1, B, C1
        # C1 rows: i_off, i_off+1. Cols: k_off, k_off+1
        
        # Strassen coefficients
        # We will add 7 terms
        
        # 1. P1
        # U: A11 + A22
        # V: B11 + B22
        # W: C11 + C12 + C21 + C22? No, W is distributed.
        # W11 += 1, W12 += 1, W21 += 1, W22 += 1 ?
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Term 1 (P1)
        U_idx_1 = [(i_off * m + j_off, 1), ((i_off+1) * m + (j_off+1), 1)]
        V_idx_1 = [(j_off * p + k_off, 1), ((j_off+1) * p + (k_off+1), 1)]
        W_idx_1 = [(k_off * n + i_off, 1), ((k_off+1) * n + i_off, 1), (k_off * n + (i_off+1), 1), ((k_off+1) * n + (i_off+1), 1)]
        
        # Helper to apply a term
        def apply_term(u_list, v_list, w_list):
            nonlocal col
            for idx, val in u_list:
                U[idx][col] += val
            for idx, val in v_list:
                V[idx][col] += val
            for idx, val in w_list:
                W[idx][col] += val
            col += 1

        # P1
        apply_term(
            [(i_off*m + j_off, 1), ((i_off+1)*m + j_off+1, 1)],
            [(j_off*p + k_off, 1), ((j_off+1)*p + k_off+1, 1)],
            [(k_off*n + i_off, 1), ((k_off+1)*n + i_off, 1), (k_off*n + i_off+1, 1), ((k_off+1)*n + i_off+1, 1)]
        )
        
        # P2 = (A21+A22)B11
        # A21: (i_off+1, j_off), A22: (i_off+1, j_off+1)
        # B11: (j_off, k_off)
        # W: C21 += 1, C22 -= 1
        apply_term(
            [(i_off+1)*m + j_off, 1, (i_off+1)*m + j_off+1, 1],
            [(j_off*p + k_off, 1)],
            [(k_off*n + i_off+1, 1), ((k_off+1)*n + i_off+1, -1)]
        )
        
        # P3 = A11(B12-B22)
        # A11: (i_off, j_off)
        # B12: (j_off, k_off+1), B22: (j_off+1, k_off+1)
        # W: C12 += 1, C22 += 1
        apply_term(
            [(i_off*m + j_off, 1)],
            [(j_off*p + k_off+1, 1), ((j_off+1)*p + k_off+1, -1)],
            [((k_off+1)*n + i_off, 1), ((k_off+1)*n + i_off+1, 1)]
        )
        
        # P4 = A22(B21-B11)
        # A22: (i_off+1, j_off+1)
        # B21: (j_off+1, k_off), B11: (j_off, k_off)
        # W: C11 += 1, C21 += 1
        apply_term(
            [(i_off+1)*m + j_off+1, 1],
            [((j_off+1)*p + k_off, 1), (j_off*p + k_off, -1)],
            [(k_off*n + i_off, 1), (k_off*n + i_off+1, 1)]
        )
        
        # P5 = (A11+A12)B22
        # A11: (i_off, j_off), A12: (i_off, j_off+1)
        # B22: (j_off+1, k_off+1)
        # W: C11 -= 1, C12 += 1
        apply_term(
            [(i_off*m + j_off, 1), (i_off*m + j_off+1, 1)],
            [((j_off+1)*p + k_off+1, 1)],
            [(k_off*n + i_off, -1), ((k_off+1)*n + i_off, 1)]
        )
        
        # P6 = (A21-A11)(B11+B12)
        # A21: (i_off+1, j_off), A11: (i_off, j_off)
        # B11: (j_off, k_off), B12: (j_off, k_off+1)
        # W: C22 += 1
        apply_term(
            [(i_off+1)*m + j_off, 1, (i_off*m + j_off, -1)],
            [(j_off*p + k_off, 1), (j_off*p + k_off+1, 1)],
            [((k_off+1)*n + i_off+1, 1)]
        )
        
        # P7 = (A12-A22)(B21+B22)
        # A12: (i_off, j_off+1), A22: (i_off+1, j_off+1)
        # B21: (j_off+1, k_off), B22: (j_off+1, k_off+1)
        # W: C11 += 1
        apply_term(
            [(i_off*m + j_off+1, 1), (i_off+1)*m + j_off+1, -1],
            [((j_off+1)*p + k_off, 1), ((j_off+1)*p + k_off+1, 1)],
            [(k_off*n + i_off, 1)]
        )
        
        # Part 2: M_{1,2,2} for A2 * B
        # A2: (i_off+2, j_off), (i_off+2, j_off+1)
        # B: same as above
        # C2: (i_off+2, k_off), (i_off+2, k_off+1)
        
        # Schoolbook 4 terms
        # C21 = A21 B11 + A22 B21
        # C22 = A21 B12 + A22 B22
        
        # T1: A21 B11 -> C21
        apply_term(
            [(i_off+2)*m + j_off, 1],
            [(j_off*p + k_off, 1)],
            [(k_off*n + i_off+2, 1)]
        )
        
        # T2: A22 B21 -> C21
        apply_term(
            [(i_off+2)*m + j_off+1, 1],
            [((j_off+1)*p + k_off, 1)],
            [(k_off*n + i_off+2, 1)]
        )
        
        # T3: A21 B12 -> C22
        apply_term(
            [(i_off+2)*m + j_off, 1],
            [(j_off*p + k_off+1, 1)],
            [((k_off+1)*n + i_off+2, 1)]
        )
        
        # T4: A22 B22 -> C22
        apply_term(
            [(i_off+2)*m + j_off+1, 1],
            [((j_off+1)*p + k_off+1, 1)],
            [((k_off+1)*n + i_off+2, 1)]
        )

    # Function to add a decomposition of M_{3,2,1}
    # A is 3x2, B is 2x1, C is 3x1
    def add_M321(i_off, j_off, k_off):
        nonlocal col
        # Split A into A1 (2x2) and A2 (1x2)
        # A1 * B (2x2 x 2x1) -> M_{2,2,1} Rank 4
        # A2 * B (1x2 x 2x1) -> M_{1,2,1} Rank 2
        
        # M_{2,2,1} Schoolbook
        # C11 = A11 B11 + A12 B21
        # C21 = A21 B11 + A22 B21
        
        # T1: A11 B11 -> C11
        apply_term(
            [(i_off*m + j_off, 1)],
            [(j_off*p + k_off, 1)],
            [(k_off*n + i_off, 1)]
        )
        
        # T2: A12 B21 -> C11
        apply_term(
            [(i_off*m + j_off+1, 1)],
            [((j_off+1)*p + k_off, 1)],
            [(k_off*n + i_off, 1)]
        )
        
        # T3: A21 B11 -> C21
        apply_term(
            [(i_off+1)*m + j_off, 1],
            [(j_off*p + k_off, 1)],
            [(k_off*n + i_off+1, 1)]
        )
        
        # T4: A22 B21 -> C21
        apply_term(
            [(i_off+1)*m + j_off+1, 1],
            [((j_off+1)*p + k_off, 1)],
            [(k_off*n + i_off+1, 1)]
        )
        
        # M_{1,2,1} Schoolbook
        # C31 = A31 B11 + A32 B21
        
        # T5: A31 B11 -> C31
        apply_term(
            [(i_off+2)*m + j_off, 1],
            [(j_off*p + k_off, 1)],
            [(k_off*n + i_off+2, 1)]
        )
        
        # T6: A32 B21 -> C31
        apply_term(
            [(i_off+2)*m + j_off+1, 1],
            [((j_off+1)*p + k_off, 1)],
            [(k_off*n + i_off+2, 1)]
        )

    # Main loop
    # Columns of B: 0..6
    # Pairs: (0,1), (2,3), (4,5)
    # Single: 6
    
    # For each pair (k, k+1)
    for k in [0, 2, 4]:
        # B_T is rows 0,1 of B. Cols k, k+1.
        # B_B is rows 2,3 of B. Cols k, k+1.
        # A_L is rows 0,1,2 of A? No, A is 3x4.
        # We split A into A_L (rows 0,1,2, cols 0,1) and A_R (rows 0,1,2, cols 2,3).
        # Wait, my previous analysis:
        # A_L: cols 0,1. A_R: cols 2,3.
        # B_T: rows 0,1. B_B: rows 2,3.
        
        # Term 1: A_L * B_T
        # A_L is 3x2 (cols 0,1 of A).
        # B_T is 2x2 (rows 0,1 of B, cols k, k+1).
        # Output C part: rows 0,1,2 of C, cols k, k+1.
        # i_off = 0 (A rows 0,1,2 map to C rows 0,1,2)
        # j_off = 0 (A cols 0,1 map to B rows 0,1)
        # k_off = k (B cols k, k+1 map to C cols k, k+1)
        
        add_M322(0, 0, k)
        
        # Term 2: A_R * B_B
        # A_R is 3x2 (cols 2,3 of A).
        # B_B is 2x2 (rows 2,3 of B, cols k, k+1).
        # Output C part: rows 0,1,2 of C, cols k, k+1.
        # i_off = 0
        # j_off = 2 (A cols 2,3 map to B rows 2,3)
        # k_off = k
        
        add_M322(0, 2, k)

    # Single column k=6
    # B_T is 2x1 (rows 0,1, col 6).
    # B_B is 2x1 (rows 2,3, col 6).
    
    # Term 1: A_L * B_T
    add_M321(0, 0, 6)
    
    # Term 2: A_R * B_B
    add_M321(0, 2, 6)
    
    # Check col count
    # 3 pairs * (11 + 11) + (6 + 6) = 66 + 12 = 78.
    # col should be 78.
    
    return {"U": U, "V": V, "W": W}
