import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_222 = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Verify 2x2x2 Strassen
    # C = A B
    # For each r, sum U[r] * V[r] * W[r] should give T
    # This is a standard check.
    
    # Construct 4x4x4 decomposition (Rank 49)
    # Indices: i, j, k in 0..3
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # r = 7*s + t
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = 2 * i1 + i0
                            j = 2 * j1 + j0
                            U_444[i, r] = U_222[i1, s] * U_222[i0, t]
                            V_444[j, r] = V_222[j1, s] * V_222[j0, t]
                            
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = 2 * k1 + k0
                            i = 2 * i1 + i0
                            W_444[k, r] = W_222[k1, s] * W_222[i0, t] # Wait, W index mapping?
                            # W maps from (k, i) to r.
                            # In 2x2, W is (k, i) -> r.
                            # Here k = 2k1 + k0, i = 2i1 + i0.
                            # So W_444[k, r] = W_222[k1, s] * W_222[k0, t]?
                            # No, the block index s corresponds to the block product.
                            # The inner index t corresponds to the element within the block.
                            # The W factor for the block level picks which block row/col?
                            # Strassen W_222[k1, s] is the coefficient for C_{k1, ?} from P_s.
                            # Actually, W_222 is indexed by (k, i) in the 2x2 case.
                            # So W_222[k1, s] is correct for the block index.
                            # And W_222[k0, t] is correct for the inner index.
                            # BUT, the indices i and k are linked.
                            # In the 2x2 case, W is 4x7. Row index is k*2+i.
                            # So W_222[row, s].
                            # row = 2*k + i.
                            # So W_222[2*k1+i1, s] * W_222[2*k0+i0, t]?
                            # Let's check the Kronecker structure.
                            # T_444 = T_222 \otimes T_222? No.
                            # The decomposition is valid if U, V, W are Kronecker products.
                            # U_444[i, r] = U_222[i1, s] U_222[i0, t] is correct for U.
                            # V_444[j, r] = V_222[j1, s] V_222[j0, t] is correct for V.
                            # W_444[k, r] = W_222[2*k1+i1, s] W_222[2*k0+i0, t]?
                            # No, W_222 has row index 0..3.
                            # Row index in 2x2 is 2*row + col.
                            # So W_222[2*k1 + i1, s] is the block-level coefficient.
                            # And W_222[2*k0 + i0, t] is the inner-level coefficient.
                            # This seems correct.
                            
    # However, the W_444 construction above has a bug in the loop logic.
    # I need to fill W_444 correctly.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for i in range(4):
                i1 = i // 2
                i0 = i % 2
                for j in range(4):
                    j1 = j // 2
                    j0 = j % 2
                    U_444[i, r] = U_222[i1, s] * U_222[i0, t]
                    V_444[j, r] = V_222[j1, s] * V_222[j0, t]
                    
            for k in range(4):
                k1 = k // 2
                k0 = k % 2
                for i in range(4):
                    i1 = i // 2
                    i0 = i % 2
                    # W index in 2x2 is 2*row + col = 2*k + i
                    w_block_idx = 2 * k1 + i1
                    w_inner_idx = 2 * k0 + i0
                    W_444[k * 4 + i, r] = W_222[w_block_idx, s] * W_222[w_inner_idx, t]
                    
    # Now construct 4x4x8 (Rank 98)
    # B is 4x8. Split into B1 (cols 0-3) and B2 (cols 4-7).
    # U is 16x98.
    # V is 32x98 (4x4 for B1, 4x4 for B2).
    # W is 32x98 (4x4 for C1, 4x4 for C2).
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First 49 columns: B1
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Next 49 columns: B2
    U[:, 49:98] = U_444
    V[16:32, 49:98] = V_444
    W[16:32, 49:98] = W_444
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

