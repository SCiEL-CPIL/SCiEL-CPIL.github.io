import numpy as np

def construct():
    n, m, p = 2, 4, 4
    
    # We construct the rank 28 solution using 4 Strassen blocks.
    # The blocks correspond to (j_block, k_block) where j_block in {0, 1} and k_block in {0, 1}.
    # j_block 0 corresponds to j in {0, 1}, j_block 1 to j in {2, 3}.
    # k_block 0 corresponds to k in {0, 1}, k_block 1 to k in {2, 3}.
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Indices in local 2x2:
    # A: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # B: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    # C: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
    
    # Strassen coefficients (U, V, W) for local indices
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    def add_block(j_block, k_block, start_col):
        # j_block: 0 or 1 (offset in j)
        # k_block: 0 or 1 (offset in k)
        
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            # U indices: i (0,1) and local_j (0,1)
            # global_j = j_block * 2 + local_j
            # global_u_idx = i * 4 + global_j
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                i = loc_idx // 2
                local_j = loc_idx % 2
                global_j = j_block * 2 + local_j
                global_u_idx = i * 4 + global_j
                u_global[global_u_idx] = u_global.get(global_u_idx, 0) + coeff
            
            # V indices: local_j (0,1) and local_k (0,1)
            # global_j = j_block * 2 + local_j
            # global_k = k_block * 2 + local_k
            # global_v_idx = global_j * 4 + global_k
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                local_j = loc_idx // 2
                local_k = loc_idx % 2
                global_j = j_block * 2 + local_j
                global_k = k_block * 2 + local_k
                global_v_idx = global_j * 4 + global_k
                v_global[global_v_idx] = v_global.get(global_v_idx, 0) + coeff
            
            # W indices: local_k (0,1) and i (0,1)
            # global_k = k_block * 2 + local_k
            # global_w_idx = global_k * 2 + i
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                local_k = loc_idx // 2
                i = loc_idx % 2
                global_k = k_block * 2 + local_k
                global_w_idx = global_k * 2 + i
                w_global[global_w_idx] = w_global.get(global_w_idx, 0) + coeff
            
            for idx, coeff in u_global.items():
                U[idx, col] += coeff
            for idx, coeff in v_global.items():
                V[idx, col] += coeff
            for idx, coeff in w_global.items():
                W[idx, col] += coeff

    # Add 4 blocks
    add_block(0, 0, 0)
    add_block(1, 0, 7)
    add_block(0, 1, 14)
    add_block(1, 1, 21)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 4, 4
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Verify
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
