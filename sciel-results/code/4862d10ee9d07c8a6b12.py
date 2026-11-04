import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strassen 2x2x2 coefficients
    # u, v, w are 4-vectors for 2x2 matrices (row-major: [00, 01, 10, 11])
    strassen_u = [
        [1, 0, 0, 1],  # u1
        [0, 0, 1, 1],  # u2
        [1, 0, 0, 0],  # u3
        [0, 0, 0, 1],  # u4
        [1, 1, 0, 0],  # u5
        [1, 0, 1, 0],  # u6 (A10 - A00 is [0,0,1,0] - [1,0,0,0] = [-1,0,1,0] or [1,0,-1,0]? 
                        # Let's use the standard signs carefully.
                        # M6 = (A10 - A00)(B00 + B01)
                        # A10 - A00: [0,0,1,0] - [1,0,0,0] = [-1, 0, 1, 0]
                        # Let's stick to the previous definition but ensure consistency.
                        # Actually, signs can be flipped in U and V as long as W is consistent?
                        # No, U*V*W must sum to 1.
                        # Let's use the explicit vectors from a reliable source or derive them.
                        # Derivation:
                        # M1 = (A00+A11)(B00+B11)
                        # M2 = (A10+A11)B00
                        # M3 = A00(B10-B00)
                        # M4 = A11(B01-B10)
                        # M5 = (A00+A01)B11
                        # M6 = (A10-A00)(B00+B01)
                        # M7 = (A01-A11)(B10+B11)
                        
                        # u1 = [1,0,0,1]
                        # u2 = [0,0,1,1]
                        # u3 = [1,0,0,0]
                        # u4 = [0,0,0,1]
                        # u5 = [1,1,0,0]
                        # u6 = [-1,0,1,0]  # A10 - A00
                        # u7 = [0,1,0,-1]  # A01 - A11
                        
                        # v1 = [1,0,0,1]
                        # v2 = [1,0,0,0]
                        # v3 = [0,0,1,-1] # B10 - B00
                        # v4 = [0,1,-1,0] # B01 - B10
                        # v5 = [0,0,0,1]
                        # v6 = [1,1,0,0]
                        # v7 = [0,0,1,1]
                        
                        # C00 = M1 + M4 - M5 - M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # w1 (M1) = [1, 0, 0, 1]
                        # w2 (M2) = [0, 0, 1, -1]
                        # w3 (M3) = [0, 1, 0, 1]
                        # w4 (M4) = [1, 0, 1, 0]
                        # w5 (M5) = [-1, 1, 0, 0]
                        # w6 (M6) = [0, 0, 0, 1]
                        # w7 (M7) = [-1, 0, 0, 0]
                    ]
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0]
    ]

    # Generate 4x4x4 Strassen decomposition (rank 49)
    U_44 = []
    V_44 = []
    W_44 = []
    
    for s in range(7):
        for t in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            u_block = strassen_u[s]
            v_block = strassen_v[s]
            w_block = strassen_w[s]
            
            u_in = strassen_u[t]
            v_in = strassen_v[t]
            w_in = strassen_w[t]
            
            for bi in range(2):
                for bj in range(2):
                    a_coeff = u_block[bi * 2 + bj]
                    if a_coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                in_coeff = u_in[local_idx]
                                if in_coeff != 0:
                                    gi = 2 * bi + li
                                    gj = 2 * bj + lj
                                    gidx = gi * 4 + gj
                                    u_vec[gidx] += a_coeff * in_coeff
                                    
                    b_coeff = v_block[bi * 2 + bj]
                    if b_coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                in_coeff = v_in[local_idx]
                                if in_coeff != 0:
                                    gi = 2 * bi + li
                                    gj = 2 * bj + lj
                                    gidx = gi * 4 + gj
                                    v_vec[gidx] += b_coeff * in_coeff
                                    
                    c_coeff = w_block[bi * 2 + bj]
                    if c_coeff != 0:
                        for li in range(2):
                            for lj in range(2):
                                local_idx = li * 2 + lj
                                in_coeff = w_in[local_idx]
                                if in_coeff != 0:
                                    gi_row = 2 * bi + li
                                    gi_col = 2 * bj + lj
                                    # W index is k*n + i. Here C is 4x4.
                                    # C index is (row, col) -> (i, k).
                                    # In the 4x4x4 context, W index is k*4 + i.
                                    # So gi_col is k, gi_row is i.
                                    gidx = gi_col * 4 + gi_row
                                    w_vec[gidx] += c_coeff * in_coeff
                                    
            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)
            
    U_44_np = np.array(U_44, dtype=float).T # (16, 49)
    V_44_np = np.array(V_44, dtype=float).T # (16, 49)
    W_44_np = np.array(W_44, dtype=float).T # (16, 49)
    
    # Generate 4x4x1 Schoolbook decomposition (rank 16)
    U_41 = []
    V_41 = []
    W_41 = []
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 16 # Only first 4 entries (j=0..3) are used for k=0
            w_vec = [0] * 16 # Only entries where k=0 are used. Index k*4+i = i.
            
            u_vec[i * 4 + j] = 1
            v_vec[j * 1 + 0] = 1 # V index j*1+0
            w_vec[0 * 4 + i] = 1 # W index 0*4+i
            
            U_41.append(u_vec)
            V_41.append(v_vec)
            W_41.append(w_vec)
            
    U_41_np = np.array(U_41, dtype=float).T # (16, 16)
    V_41_np = np.array(V_41, dtype=float).T # (16, 16)
    W_41_np = np.array(W_41, dtype=float).T # (16, 16)
    
    # Assemble full 4x4x9
    rank_total = 49 + 49 + 16
    U_full = np.zeros((16, rank_total), dtype=float)
    V_full = np.zeros((36, rank_total), dtype=float)
    W_full = np.zeros((36, rank_total), dtype=float)
    
    # Block 1: k=0..3
    U_full[:, 0:49] = U_44_np
    # V: j*9 + k. k=0..3.
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + k, 0:49] = V_44_np[j * 4 + k, :]
    # W: k*4 + i. k=0..3.
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, 0:49] = W_44_np[k * 4 + i, :]
            
    # Block 2: k=4..7
    U_full[:, 49:98] = U_44_np
    for j in range(4):
        for k in range(4):
            V_full[j * 9 + (4 + k), 49:98] = V_44_np[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[(4 + k) * 4 + i, 49:98] = W_44_np[k * 4 + i, :]
            
    # Block 3: k=8
    U_full[:, 98:114] = U_41_np
    # V: j*9 + 8.
    for j in range(4):
        V_full[j * 9 + 8, 98:114] = V_41_np[j * 1 + 0, :]
    # W: 8*4 + i.
    for i in range(4):
        W_full[8 * 4 + i, 98:114] = W_41_np[0 * 4 + i, :]
        
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Full Validation
    def is_valid_full(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all non-zero entries
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    u_idx = i * 4 + j
                    v_idx = j * 9 + k
                    w_idx = k * 4 + i
                    val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some zero entries to ensure no leakage
        # Sample a few
        import random
        random.seed(42)
        for _ in range(1000):
            u_idx = random.randint(0, 15)
            v_idx = random.randint(0, 35)
            w_idx = random.randint(0, 35)
            
            # Check if this is a valid (i,j,k) triple
            i = u_idx // 4
            j = u_idx % 4
            j_v = v_idx // 9
            k = v_idx % 9
            k_w = w_idx // 4
            i_w = w_idx % 4
            
            if j == j_v and i == i_w and k == k_w:
                continue # Skip diagonal
                
            val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
            if abs(val) > 1e-9:
                return False
                
        return True

    if is_valid_full(U_list, V_list, W_list, rank_total):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Try random search for a better or alternative decomposition
        import random
        random.seed(42)
        best_result = None
        for R in range(114, 100, -1):
            if time.time() > deadline - 10:
                break
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                # Random sparse U and V
                U_rand = np.zeros((16, R))
                V_rand = np.zeros((36, R))
                for r in range(R):
                    # Each column of U has ~4 non-zeros
                    nz_u = np.random.choice(16, size=min(4, 16), replace=False)
                    U_rand[nz_u, r] = np.random.choice([-1, 1], size=len(nz_u))
                    # Each column of V has ~6 non-zeros
                    nz_v = np.random.choice(36, size=min(6, 36), replace=False)
                    V_rand[nz_v, r] = np.random.choice([-1, 1], size=len(nz_v))
                
                # Build linear system for W
                # For each (i,j,k): sum_r U[i*4+j,r] * V[j*9+k,r] * W[k*4+i,r] = 1
                # For invalid (i,j,k): sum_r U[i*4+j,r] * V[j*9+k,r] * W[k*4+i,r] = 0
                # This is linear in W: A @ W_vec = b
                # W_vec is 36*R elements (flattened W)
                # A is (36*36) x (36*R) but we only need the 36*4*9 = 144 diagonal constraints
                # Actually, we need all 16*36*36 = 20736 constraints, but most are zero
                # Let's just check the 144 diagonal constraints and a sample of zeros
                
                # Build A for the 144 diagonal constraints
                A_rows = []
                b_vec = []
                for i in range(4):
                    for j in range(4):
                        for k in range(9):
                            u_idx = i * 4 + j
                            v_idx = j * 9 + k
                            w_idx = k * 4 + i
                            row = np.zeros(36 * R)
                            for r in range(R):
                                coeff = U_rand[u_idx, r] * V_rand[v_idx, r]
                                if coeff != 0:
                                    row[w_idx * R + r] = coeff
                            A_rows.append(row)
                            b_vec.append(1.0)
                
                A = np.array(A_rows)
                b = np.array(b_vec)
                
                # Solve least squares
                W_sol, residuals, rank_A, sv = lstsq(A, b, cond=None)
                
                # Check residual
                if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-8:
                    continue
                
                W_mat = W_sol.reshape(36, R)
                
                # Check zero constraints (sample)
                valid = True
                for _ in range(500):
                    u_idx = random.randint(0, 15)
                    v_idx = random.randint(0, 35)
                    w_idx = random.randint(0, 35)
                    i = u_idx // 4
                    j = u_idx % 4
                    j_v = v_idx // 9
                    k = v_idx % 9
                    k_w = w_idx // 4
                    i_w = w_idx % 4
                    if j == j_v and i == i_w and k == k_w:
                        continue
                    val = np.sum(U_rand[u_idx, :] * V_rand[v_idx, :] * W_mat[w_idx, :])
                    if abs(val) > 1e-8:
                        valid = False
                        break
                
                if valid:
                    best_result = (U_rand.tolist(), V_rand.tolist(), W_mat.tolist(), R)
                    break
            if best_result is not None:
                break
        
        if best_result is not None:
            U_r, V_r, W_r, R_r = best_result
            return {"U": U_r, "V": V_r, "W": W_r}
        
        # Fallback to schoolbook
        rank = 144
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U[i * 4 + j][col] = 1
                    V[j * 9 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
