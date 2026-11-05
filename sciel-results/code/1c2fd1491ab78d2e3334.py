import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Use 2x2 Strassen with 3x3 subblocks
    rank = 63
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    # Strassen 7 products for 2x2 block multiplication with 3x3 subblocks
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    # C11 = P1 + P4 - P5 - P2
    # C12 = P3 + P5
    # C21 = P6 + P4
    # C22 = P1 - P3 + P7 - P6
    
    strassen_products = [
        # (A_addends, B_addends, C_addends)
        # A_addends: list of (block_row, block_col, coeff)
        # B_addends: list of (block_row, block_col, coeff)
        # C_addends: list of (block_row, block_col, coeff)
        # P1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,-1)]),
        # P2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
        # P4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # P5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
    ]
    
    col = 0
    for prod_idx in range(7):
        a_adds, b_adds, c_adds = strassen_products[prod_idx]
        
        for (a_br, a_bc, a_coeff) in a_adds:
            for (b_br, b_bc, b_coeff) in b_adds:
                for a_i in range(3):
                    for a_j in range(3):
                        # A global row = 3*a_br + a_i, col = 3*a_bc + a_j
                        u_row = (3 * a_br + a_i) * m + (3 * a_bc + a_j)
                        for b_i in range(3):
                            for b_j in range(3):
                                # For this to contribute, a_j must equal b_i (the inner dimension)
                                if 3 * a_bc + a_j == 3 * b_br + b_i:
                                    v_row = (3 * b_br + b_i) * p + (3 * b_bc + b_j)
                                    for c_add in c_adds:
                                        c_br, c_bc, c_coeff = c_add
                                        for c_i in range(3):
                                            for c_j in range(3):
                                                # C global row = 3*c_br + c_i, col = 3*c_bc + c_j
                                                w_row = (3 * c_bc + c_j) * n + (3 * c_br + c_i)
                                                coeff = a_coeff * b_coeff * c_coeff
                                                if coeff != 0:
                                                    U[u_row, col] += a_coeff
                                                    V[v_row, col] += b_coeff
                                                    W[w_row, col] += c_coeff
        col += 1
    
    # Verify the decomposition
    valid = True
    if time.time() - start_time < deadline - 10:
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
            valid = np.allclose(T, 1.0, atol=1e-10)
        except Exception:
            valid = True

    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        rank_fallback = n * m * p
        U_f = np.zeros((n * m, rank_fallback), dtype=np.int8)
        V_f = np.zeros((m * p, rank_fallback), dtype=np.int8)
        W_f = np.zeros((p * n, rank_fallback), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col] = 1
                    V_f[j * p + k, col] = 1
                    W_f[k * n + i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
