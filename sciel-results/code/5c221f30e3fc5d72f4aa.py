import time
import numpy as np

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110

    # Trivial rank 44 decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    # Try to find a rank 39 decomposition using Strassen
    # We tile 5 blocks of 2x2x2 and 1 block of 2x2x1
    # Rank 39 = 5 * 7 + 4
    
    # Define candidate Strassen factors for 2x2x2
    # U is 4x7, V is 4x7, W is 4x7
    # Rows of U: 00, 01, 10, 11 (i*m+j)
    # Rows of V: 00, 01, 10, 11 (j*p+k) -> for 2x2x2, p=2, so 00, 01, 10, 11
    # Rows of W: 00, 01, 10, 11 (k*n+i) -> for 2x2x2, n=2, so 00, 01, 10, 11
    
    # We will test a few known Strassen variants
    candidates = []
    
    # Variant 1: Standard Strassen from Wikipedia (1-indexed converted to 0-indexed)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = (A00+A01)B11
    # M4 = (A11-A00)(B00+B01)
    # M5 = (A00+A11)(B10+B11)
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    
    # U (4x7)
    U1 = [
        [1, 0, 1, -1, 1, 0, -1], # A00
        [0, 0, 1, 0, 0, 1, 0],   # A01
        [0, 1, 0, 0, 0, 0, 1],   # A10
        [1, 1, 0, 1, 1, -1, 0]   # A11
    ]
    # V (4x7)
    V1 = [
        [1, 1, 0, 1, 0, 0, 1],   # B00
        [0, 0, 0, 1, 0, 0, 1],   # B01
        [0, 0, 1, 0, 1, 1, 0],   # B10
        [1, 0, -1, 0, 1, 1, 0]   # B11
    ]
    # W (4x7) - We need to solve for W or use known formulas
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 - M1 + M6 + M7
    # C11 = M1 - M3 + M4 + M5
    
    W1 = [
        [1, 0, 0, -1, 1, 0, -1], # C00
        [0, 0, 1, 1, 0, 0, 0],   # C01
        [0, 1, 0, 0, -1, 1, 1],  # C10
        [1, 0, -1, 1, 1, 0, 0]   # C11
    ]
    candidates.append((U1, V1, W1))

    # Variant 2: Another common formulation
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    W2 = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 0, 0, 1, 1]   # C11
    ]
    candidates.append((U2, V2, W2))

    def verify_strassen(U, V, W):
        # Test with random integer matrices
        for _ in range(10):
            A = np.random.randint(-5, 5, (2, 2))
            B = np.random.randint(-5, 5, (2, 2))
            C_expected = A @ B
            
            # Compute C using decomposition
            # C_ik = sum_r U_{i,m+j} V_{j,p+k} W_{k,n+i} ? No.
            # The decomposition is C = sum_r U[:,r] V[:,r] W[:,r]^T ?
            # Actually, the tensor is T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
            # a = i*m+j, b = j*p+k, c = k*n+i
            # For 2x2x2:
            # a = 2i+j, b = 2j+k, c = 2k+i
            C_calc = np.zeros((2, 2))
            for i in range(2):
                for k in range(2):
                    val = 0
                    for r in range(7):
                        # Sum over j
                        for j in range(2):
                            a = i * 2 + j
                            b = j * 2 + k
                            c = k * 2 + i
                            val += U[a][r] * V[b][r] * W[c][r]
                    C_calc[i][k] = val
            if not np.allclose(C_calc, C_expected):
                return False
        return True

    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial

    for U_sub, V_sub, W_sub in candidates:
        if verify_strassen(U_sub, V_sub, W_sub):
            # Construct full 39 rank matrix
            rank = 39
            U_full = [[0] * rank for _ in range(n * m)]
            V_full = [[0] * rank for _ in range(m * p)]
            W_full = [[0] * rank for _ in range(p * n)]
            
            # 5 blocks of 2x2x2 (cols 0-1, 2-3, 4-5, 6-7, 8-9)
            # 1 block of 2x2x1 (col 10)
            
            base_r = 0
            for block in range(5):
                k_start = block * 2
                # U is same for all blocks (depends only on i,j)
                for i in range(2):
                    for j in range(2):
                        a = i * m + j
                        for r in range(7):
                            U_full[a][base_r + r] = U_sub[a][r]
                
                # V depends on j and k
                for j in range(2):
                    for k in range(2):
                        b = j * p + (k_start + k)
                        for r in range(7):
                            V_full[b][base_r + r] = V_sub[j * 2 + k][r]
                
                # W depends on k and i
                for k in range(2):
                    for i in range(2):
                        c = (k_start + k) * n + i
                        for r in range(7):
                            W_full[c][base_r + r] = W_sub[k * 2 + i][r]
                
                base_r += 7
            
            # Last block: col 10, trivial rank 4
            # C_i,10 = sum_j A_i,j B_j,10
            # Terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
            for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                r = base_r + idx
                a = i * m + j
                b = j * p + 10
                c = 10 * n + i
                U_full[a][r] = 1
                V_full[b][r] = 1
                W_full[c][r] = 1
            
            best_rank = 39
            best_U, best_V, best_W = U_full, V_full, W_full
            break

    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
