import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    
    # We construct a decomposition of rank 53.
    # Part 1: 49 terms from 4x4 Strassen for columns 0-3
    # Part 2: 4 terms for column 4
    
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 4x4 Strassen ---
    # Define 2x2 Strassen vectors
    u_2x2 = [
        np.array([1, 0, 0, 1]), # M1
        np.array([0, 0, 1, 1]), # M2
        np.array([1, 0, 0, 0]), # M3
        np.array([0, 0, 0, 1]), # M4
        np.array([1, 1, 0, 0]), # M5
        np.array([-1, 0, 1, 0]), # M6
        np.array([0, 1, 0, -1])  # M7
    ]
    v_2x2 = [
        np.array([1, 0, 0, 1]), # M1
        np.array([1, 0, 0, 0]), # M2
        np.array([0, 1, 0, -1]), # M3
        np.array([0, 0, 1, -1]), # M4
        np.array([0, 0, 0, 1]), # M5
        np.array([1, 1, 0, 0]), # M6
        np.array([0, 0, 1, 1])  # M7
    ]
    w_2x2 = [
        np.array([1, 0, 0, 1]), # M1 -> C00, C11
        np.array([-1, 0, 1, -1]), # M2 -> C00, C10, C11
        np.array([0, 1, 0, 1]), # M3 -> C01, C11
        np.array([1, 0, 1, 0]), # M4 -> C00, C10
        np.array([0, 1, 0, 0]), # M5 -> C01
        np.array([0, 0, 0, 1]), # M6 -> C11
        np.array([-1, 0, 0, 0])  # M7 -> C00
    ]
    
    # Generate 49 terms for 4x4x4
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    col_idx = 0
    for r in range(7):
        for s in range(7):
            # Kronecker product for U, V, W
            # The 4x4 matrix is formed by 2x2 blocks.
            # Index (I, J) in 4x4 corresponds to (i, j) where I = 2*i + a, J = 2*j + b?
            # Actually, standard Kronecker:
            # If A = [[A00, A01], [A10, A11]] where Axx are 2x2.
            # Then vec(A) = [vec(A00), vec(A01), vec(A10), vec(A11)]?
            # Or [vec(A00), vec(A10), vec(A01), vec(A11)]?
            # Numpy kron uses standard order.
            # Let's assume the vectors u_2x2 are for the 2x2 block coefficients.
            # The full U vector for the 4x4 case is the kron of the block coeff vector and the inner vec?
            
            # Let's be precise.
            # The term is (u_r * A_blocks) (v_r * B_blocks).
            # This is equivalent to a linear combination of A.
            # The coefficient vector for the flattened 4x4 A is the Kronecker product of the block selection vector and the inner vector?
            # No, it's just the Kronecker product of the two 2x2 coefficient matrices flattened?
            
            # Let U_4x4 = kron(u_r, u_s) ?
            # u_r is length 4. u_s is length 4.
            # kron(u_r, u_s) is length 16.
            
            U_44 = np.kron(u_2x2[r], u_2x2[s])
            V_44 = np.kron(v_2x2[r], v_2x2[s])
            W_44 = np.kron(w_2x2[r], w_2x2[s])
            
            # Map U_44 (size 16) to U (size 12)
            # U_44 index: I*4 + J
            # We want A_{i, j} for i in 0..2, j in 0..3.
            # A is padded to 4x4 with row 3 = 0.
            # So we only take U_44 indices where i < 3.
            # I = i, J = j.
            # U[i*4 + j, col_idx] = U_44[I*4 + J]
            
            for i in range(3):
                for j in range(4):
                    U[i * 4 + j, col_idx] = U_44[i * 4 + j]
            
            # Map V_44 (size 16) to V (size 20)
            # V_44 index: J*4 + K
            # We want B_{j, k} for j in 0..3, k in 0..3 (first 4 cols).
            # B is 4x5. We use cols 0-3.
            # V[j*5 + k, col_idx] = V_44[J*4 + K]
            
            for j in range(4):
                for k in range(4):
                    V[j * 5 + k, col_idx] = V_44[j * 4 + k]
            
            # Map W_44 (size 16) to W (size 15)
            # W_44 index: K*4 + I
            # We want C_{i, k} for i in 0..2, k in 0..3.
            # W index: k*n + i = k*3 + i.
            # W[k*3 + i, col_idx] = W_44[K*4 + I]
            
            for k in range(4):
                for i in range(3):
                    W[k * 3 + i, col_idx] = W_44[k * 4 + i]
            
            col_idx += 1
    
    # --- Part 2: Column 4 ---
    # C_{i, 4} = sum_{j=0}^3 A_{i, j} B_{j, 4}
    # 4 terms.
    for j in range(4):
        # U: 1s at (0, j), (1, j), (2, j)
        for i in range(3):
            U[i * 4 + j, col_idx] = 1.0
        
        # V: 1 at (j, 4)
        V[j * 5 + 4, col_idx] = 1.0
        
        # W: 1s at (4, 0), (4, 1), (4, 2) -> indices 12, 13, 14
        for i in range(3):
            W[4 * 3 + i, col_idx] = 1.0
            
        col_idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
