"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper: create a column for a rank-1 tensor term
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For a term (u_idx, v_idx, w_idx, coeff):
    # U[u_idx][col] += coeff * u_val
    # V[v_idx][col] += coeff * v_val
    # W[w_idx][col] += coeff * w_val
    # But typically we absorb the coefficient into one of the matrices.
    # Let's define a function to add a term where U, V, W entries are given.
    
    U = [[0] * 1 for _ in range(n * m)]
    V = [[0] * 1 for _ in range(m * p)]
    W = [[0] * 1 for _ in range(p * n)]
    
    rank = 0
    col_id = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank, col_id
        U[u_idx].append(coeff)
        V[v_idx].append(1)
        W[w_idx].append(1)
        # Pad other rows
        for i in range(n * m):
            if i != u_idx:
                U[i].append(0)
        for i in range(m * p):
            if i != v_idx:
                V[i].append(0)
        for i in range(p * n):
            if i != w_idx:
                W[i].append(0)
        rank += 1
        col_id += 1

    # Helper: Schoolbook decomposition for n x m x p
    # We return the list of (u_idx, v_idx, w_idx) terms
    def schoolbook_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Global indices
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append((u_idx, v_idx, w_idx))
        return terms

    # Helper: Strassen-like decomposition for 3x2x2? Or use known bounds.
    # For 3x5x5, splitting m=5 into 3+2 and p=5 into 3+2:
    # C = A * B
    # A is 3x5, B is 5x5
    # Let A = [A1 | A2] where A1 is 3x3, A2 is 3x2
    # Let B = [[B11, B12], [B21, B22]] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2
    # C = A1*B11 + A2*B21 (left 3 cols)
    #    + A1*B12 + A2*B22 (right 2 cols)
    
    # Subproblems:
    # 1. 3x3x3 (A1*B11): Rank 23 is known optimal? No, 3x3x3 is 23? 
    #    Actually, rank of 3x3x3 is 23? 
    #    Wait, rank of 3x3 matrix multiplication is 23? No, it's 23? 
    #    Let's check: R(3,3) = 23? 
    #    Actually, R(3,3) is known to be 23? 
    #    No, R(3,3) is 23? 
    #    Wait, R(2,2)=7. R(3,3) is at least 19? 
    #    Actually, the best known upper bound for 3x3 is 23? 
    #    Let's stick to schoolbook for 3x3 to be safe, rank 27.
    #    But maybe we can do better.
    #    Let's just use schoolbook for all subblocks to guarantee correctness.
    #    Schoolbook ranks:
    #    3x3x3: 27
    #    3x2x3: 18
    #    3x3x2: 18
    #    3x2x2: 12
    #    Total: 27+18+18+12 = 75. Same as schoolbook.
    
    # To improve, we need a better decomposition for one of the blocks.
    # 3x2x2: Can we do better than 12?
    # 3x2x2 is equivalent to 2x3x2? 
    # R(2,2,2) is 7.
    # R(3,2,2) ?
    # Let's try to find a rank 11 decomposition for 3x2x2.
    # If we can find one, total becomes 27+18+18+11 = 74.
    
    # Let's look for a rank 11 decomposition of 3x2x2.
    # Tensor M_{3,2,2}.
    # This is a small problem. We can brute force search for a rank 11 decomposition.
    # Or use a known algorithm.
    # Actually, let's just implement the schoolbook for all blocks first to ensure it works, 
    # then optimize 3x2x2 if possible.
    
    # Block 1: A1 (3x3) * B11 (3x3) -> C1 (3x3)
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # Global i: 0..2
    # Global j: 0..2
    # Global k: 0..2
    # U_idx = i*5 + j
    # V_idx = j*5 + k
    # W_idx = k*3 + i
    
    # Block 2: A2 (3x2) * B21 (2x3) -> C2 (3x3)
    # A2 cols: 3..4. B21 rows: 3..4.
    # Global i: 0..2
    # Global j: 3..4 (local j2: 0..1, global j = 3+j2)
    # Global k: 0..2 (local k2: 0..2)
    # U_idx = i*5 + (3+j2)
    # V_idx = (3+j2)*5 + k2
    # W_idx = k2*3 + i
    
    # Block 3: A1 (3x3) * B12 (3x2) -> C3 (3x2)
    # Global i: 0..2
    # Global j: 0..2
    # Global k: 3..4 (local k2: 0..1, global k = 3+k2)
    # U_idx = i*5 + j
    # V_idx = j*5 + (3+k2)
    # W_idx = (3+k2)*3 + i
    
    # Block 4: A2 (3x2) * B22 (2x2) -> C4 (3x2)
    # Global i: 0..2
    # Global j: 3..4
    # Global k: 3..4
    # U_idx = i*5 + (3+j2)
    # V_idx = (3+j2)*5 + (3+k2)
    # W_idx = (3+k2)*3 + i

    # Initialize U, V, W as empty lists of lists
    U = []
    V = []
    W = []
    
    # Add terms from each block
    # Block 1: 3x3x3 Schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 3 + i
                # Append column
                U.append([0] * (n*m))
                V.append([0] * (m*p))
                W.append([0] * (p*n))
                U[-1][u_idx] = 1
                V[-1][v_idx] = 1
                W[-1][w_idx] = 1

    # Block 2: 3x2x3 Schoolbook (A2 * B21)
    for i in range(3):
        for j2 in range(2):
            for k in range(3):
                j = 3 + j2
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 3 + i
                U.append([0] * (n*m))
                V.append([0] * (m*p))
                W.append([0] * (p*n))
                U[-1][u_idx] = 1
                V[-1][v_idx] = 1
                W[-1][w_idx] = 1

    # Block 3: 3x3x2 Schoolbook (A1 * B12)
    for i in range(3):
        for j in range(3):
            for k2 in range(2):
                k = 3 + k2
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 3 + i
                U.append([0] * (n*m))
                V.append([0] * (m*p))
                W.append([0] * (p*n))
                U[-1][u_idx] = 1
                V[-1][v_idx] = 1
                W[-1][w_idx] = 1

    # Block 4: 3x2x2 Schoolbook (A2 * B22)
    for i in range(3):
        for j2 in range(2):
            for k2 in range(2):
                j = 3 + j2
                k = 3 + k2
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 3 + i
                U.append([0] * (n*m))
                V.append([0] * (m*p))
                W.append([0] * (p*n))
                U[-1][u_idx] = 1
                V[-1][v_idx] = 1
                W[-1][w_idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
