"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Strategy: 
    1. Try Rank 46 using Laderman's rank-23 algorithm for 3x3x3, applied to two 3x3x3 blocks.
       This requires a verified set of coefficients. If the hardcoded coefficients are incorrect,
       verification will fail and we fall back.
    2. Fallback to Rank 51 using Strassen for 2x2 blocks and schoolbook for the rest.
    3. Final fallback to Rank 54 (Schoolbook).
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # --- Rank 23 Decomposition for 3x3x3 (Laderman) ---
    # Reference: Laderman (1976) "The rank of 3x3 matrix multiplication"
    # This is a hardcoded set of coefficients. 
    # U: 9x23, V: 9x23, W: 9x23
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    def get_laderman_3x3x3():
        # The 23 terms of Laderman's algorithm
        # Each term is a triple (u, v, w) where u, v, w are 9-vectors (flattened 3x3 matrices)
        # We define them as lists of 9 coefficients.
        
        # Helper to create a 9-vector with 1 at (r, c)
        def e(r, c):
            v = [0]*9
            v[r*3+c] = 1
            return v
            
        # Helper to add/subtract vectors
        def add(v1, v2, sign=1):
            return [x + sign*y for x, y in zip(v1, v2)]
            
        # The 23 products
        # Note: There are multiple equivalent formulations. This one is a common standard.
        # If this specific set is wrong, verification will catch it.
        
        terms = []
        
        # 1. a00 b00
        terms.append((e(0,0), e(0,0), e(0,0)))
        
        # 2. a01 (b10 + b20)
        terms.append((e(0,1), add(e(1,0), e(2,0)), add(e(1,0), e(2,0))))
        
        # 3. (a00 + a10) b11
        terms.append((add(e(0,0), e(1,0)), e(1,1), add(e(0,1), e(1,1))))
        
        # 4. a12 (b01 + b21)
        terms.append((e(1,2), add(e(0,1), e(2,1)), add(e(1,0), e(2,0))))
        
        # 5. (a01 + a11) (b00 - b10)
        terms.append((add(e(0,1), e(1,1)), add(e(0,0), e(1,0), -1), add(e(0,0), e(1,0), -1)))
        
        # 6. (a02 + a12) (b00 - b10)
        terms.append((add(e(0,2), e(1,2)), add(e(0,0), e(1,0), -1), add(e(0,0), e(1,0), -1)))
        
        # 7. (a10 + a20) (b01 - b11)
        terms.append((add(e(1,0), e(2,0)), add(e(0,1), e(1,1), -1), add(e(0,0), e(1,0), -1)))
        
        # 8. (a11 + a21) (b00 - b10)
        terms.append((add(e(1,1), e(2,1)), add(e(0,0), e(1,0), -1), add(e(0,1), e(1,1), -1)))
        
        # 9. (a12 + a22) (b00 - b10)
        terms.append((add(e(1,2), e(2,2)), add(e(0,0), e(1,0), -1), add(e(0,2), e(1,2), -1)))
        
        # 10. a20 (b11 + b21)
        terms.append((e(2,0), add(e(1,1), e(2,1)), add(e(1,0), e(2,0))))
        
        # 11. (a00 + a20) (b01 - b21)
        terms.append((add(e(0,0), e(2,0)), add(e(0,1), e(2,1), -1), add(e(0,0), e(2,0), -1)))
        
        # 12. (a01 + a21) (b00 - b20)
        terms.append((add(e(0,1), e(2,1)), add(e(0,0), e(2,0), -1), add(e(0,1), e(2,1), -1)))
        
        # 13. (a02 + a22) (b00 - b20)
        terms.append((add(e(0,2), e(2,2)), add(e(0,0), e(2,0), -1), add(e(0,2), e(2,2), -1)))
        
        # 14. a22 (b10 + b20)
        terms.append((e(2,2), add(e(1,0), e(2,0)), add(e(2,0), e(2,1))))
        
        # 15. (a01 - a02) (b12 - b22)
        terms.append((add(e(0,1), e(0,2), -1), add(e(1,2), e(2,2), -1), add(e(0,1), e(0,2), -1)))
        
        # 16. (a10 - a11) (b02 - b22)
        terms.append((add(e(1,0), e(1,1), -1), add(e(0,2), e(2,2), -1), add(e(1,0), e(1,1), -1)))
        
        # 17. (a20 - a21) (b02 - b12)
        terms.append((add(e(2,0), e(2,1), -1), add(e(0,2), e(1,2), -1), add(e(2,0), e(2,1), -1)))
        
        # 18. (a12 - a22) (b01 - b11)
        terms.append((add(e(1,2), e(2,2), -1), add(e(0,1), e(1,1), -1), add(e(1,2), e(2,2), -1)))
        
        # 19. (a02 - a12) (b10 - b20)
        terms.append((add(e(0,2), e(1,2), -1), add(e(1,0), e(2,0), -1), add(e(0,2), e(1,2), -1)))
        
        # 20. (a11 - a21) (b00 - b10)
        terms.append((add(e(1,1), e(2,1), -1), add(e(0,0), e(1,0), -1), add(e(1,1), e(2,1), -1)))
        
        # 21. (a00 - a10) (b11 - b21)
        terms.append((add(e(0,0), e(1,0), -1), add(e(1,1), e(2,1), -1), add(e(0,0), e(1,0), -1)))
        
        # 22. (a01 - a11) (b00 - b20)
        terms.append((add(e(0,1), e(1,1), -1), add(e(0,0), e(2,0), -1), add(e(0,1), e(1,1), -1)))
        
        # 23. (a02 - a12) (b00 - b20)
        terms.append((add(e(0,2), e(1,2), -1), add(e(0,0), e(2,0), -1), add(e(0,2), e(1,2), -1)))
        
        # Construct U, V, W matrices
        rank = len(terms)
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(9)]
        W = [[0]*rank for _ in range(9)]
        
        for c, (u, v, w) in enumerate(terms):
            for idx in range(9):
                U[idx][c] = u[idx]
                V[idx][c] = v[idx]
                W[idx][c] = w[idx]
                
        return U, V, W, rank

    # --- Fallback: Rank 51 Construction ---
    def construct_rank_51():
        n, m, p = 3, 3, 6
        total_rank = 51
        U_full = [[0.0] * total_rank for _ in range(3 * 3)]
        V_full = [[0.0] * total_rank for _ in range(3 * 6)]
        W_full = [[0.0] * total_rank for _ in range(6 * 3)]
        
        # Strassen 2x2x2 helper
        def strassen_2x2x2_decomp():
            U = [
                [1, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, -1, 0],
                [1, 1, 0, 1, 0, 0, -1],
            ]
            V = [
                [1, 1, 0, -1, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 1],
                [0, 0, -1, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1],
            ]
            W = [
                [1, 0, 0, 0, -1, 0, 1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 1, 0],
            ]
            return U, V, W, 7
        
        strassen_base = 0
        for block in range(3):
            U_s, V_s, W_s, r_s = strassen_2x2x2_decomp()
            a_map = {0: 0, 1: 1, 2: 3, 3: 4}
            b_map = {
                0: 2 * block + 0,
                1: 2 * block + 1,
                2: 6 + 2 * block + 0,
                3: 6 + 2 * block + 1,
            }
            w_map = {
                0: (2 * block + 0) * 3 + 0,
                1: (2 * block + 1) * 3 + 0,
                2: (2 * block + 0) * 3 + 1,
                3: (2 * block + 1) * 3 + 1,
            }
            
            for c in range(r_s):
                col = strassen_base + c
                for s_row in range(4):
                    full_a_row = a_map[s_row]
                    U_full[full_a_row][col] += U_s[s_row][c]
                for s_row in range(4):
                    full_b_row = b_map[s_row]
                    V_full[full_b_row][col] += V_s[s_row][c]
                for s_row in range(4):
                    full_c_row = w_map[s_row]
                    W_full[full_c_row][col] += W_s[s_row][c]
        
        strassen_base = 21
        
        # A21 * B16 (Rows 0,1 of A, Col 2 of A * Row 2 of B)
        # U index: i*3+2
        # V index: 2*6+k
        # W index: k*3+i
        col = strassen_base
        for i in range(2):
            for k in range(6):
                U_full[i * 3 + 2][col] = 1.0
                V_full[2 * 6 + k][col] = 1.0
                W_full[k * 3 + i][col] = 1.0
                col += 1
        strassen_base += 12
        
        # A12 * B26 (Row 2 of A, Cols 0,1 of A * Rows 0,1 of B)
        # U index: 2*3+j
        # V index: j*6+k
        # W index: k*3+2
        col = strassen_base
        for j in range(2):
            for k in range(6):
                U_full[2 * 3 + j][col] = 1.0
                V_full[j * 6 + k][col] = 1.0
                W_full[k * 3 + 2][col] = 1.0
                col += 1
        strassen_base += 12
        
        # A11 * B16 (Row 2, Col 2 of A * Row 2 of B)
        # U index: 2*3+2
        # V index: 2*6+k
        # W index: k*3+2
        col = strassen_base
        for k in range(6):
            U_full[2 * 3 + 2][col] = 1.0
            V_full[2 * 6 + k][col] = 1.0
            W_full[k * 3 + 2][col] = 1.0
            col += 1
            
        return U_full, V_full, W_full

    # --- Verification ---
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0]) if U else 0
        # Check that for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    # --- Attempt Rank 46 ---
    # We split the 3x3x6 multiplication into two 3x3x3 multiplications.
    # C[:, 0:3] = A * B[:, 0:3]
    # C[:, 3:6] = A * B[:, 3:6]
    
    U_l, V_l, W_l, rank_l = get_laderman_3x3x3()
    
    total_rank = 2 * rank_l
    U_full = [[0] * total_rank for _ in range(9)]
    V_full = [[0] * total_rank for _ in range(18)]
    W_full = [[0] * total_rank for _ in range(18)]
    
    # First block: k=0,1,2
    for i in range(9):
        for c in range(rank_l):
            U_full[i][c] = U_l[i][c]
            # V index for first block: j*3 + k (k in 0..2) -> global index j*6 + k
            # But V_l is 9x23, indexed by j*3+k.
            # We need to map local (j,k) to global (j, k).
            # Local row r in V_l corresponds to j=r//3, k=r%3.
            # Global row in V_full is j*6 + k.
            j = i // 3
            k = i % 3
            # Wait, V_l row index is j*3+k.
            # We are iterating i over rows of U_l.
            # Let's iterate properly over j and k for V and W.
            
            # Actually, it's easier to just copy the matrices and shift indices.
            # U is same for both blocks.
            # V: local row (j*3+k) -> global row (j*6+k) for first block, (j*6+k+3) for second.
            # W: local row (k*3+i) -> global row (k*3+i) for first block, ((k+3)*3+i) for second.
            
    # Let's redo the copying logic carefully.
    for c in range(rank_l):
        for j in range(3):
            for k in range(3):
                # U is shared
                # V
                # Local V index: j*3+k
                # Global V index (block 1): j*6 + k
                V_full[j*6 + k][c] = V_l[j*3 + k][c]
                # Global V index (block 2): j*6 + k + 3
                V_full[j*6 + k + 3][rank_l + c] = V_l[j*3 + k][c]
                
                # W
                # Local W index: k*3+i
                # Global W index (block 1): k*3 + i
                # Global W index (block 2): (k+3)*3 + i = k*3 + 9 + i
                for i in range(3):
                    W_full[k*3 + i][c] = W_l[k*3 + i][c]
                    W_full[k*3 + 9 + i][rank_l + c] = W_l[k*3 + i][c]
                    
        # U
        for i in range(3):
            for j in range(3):
                U_full[i*3 + j][c] = U_l[i*3 + j][c]
                U_full[i*3 + j][rank_l + c] = U_l[i*3 + j][c]

    if verify_decomposition(U_full, V_full, W_full, 3, 3, 6):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # --- Fallback Rank 51 ---
    if time.time() < deadline:
        U51, V51, W51 = construct_rank_51()
        if verify_decomposition(U51, V51, W51, 3, 3, 6):
            return {"U": U51, "V": V51, "W": W51}
            
    # --- Final Fallback Rank 54 ---
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
