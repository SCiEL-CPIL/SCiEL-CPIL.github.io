import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 33 solution
    def get_rank_33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Terms for M_{2,3,2} decomposition (Rank 11: 7 Strassen + 4 Naive)
        # We use the same 11 terms for each of the 3 blocks.
        
        # Strassen terms for 2x2 part of A (cols 0,1) and 2x2 part of B (cols 0,1 of block)
        # A_2x2 = [[a0, a1], [a3, a4]]
        # B_2x2 = [[b0, b1], [b2, b3]] (local indices)
        
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B22
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B11
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Note: Indices for A: 0,1 (row 0), 3,4 (row 1).
        # Indices for B (local 2x2): 0,1 (col 0), 2,3 (col 1).
        
        strassen_terms = [
            # M1
            {'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
            # M2
            {'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
            # M3
            {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
            # M4
            {'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
            # M5
            {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
            # M6
            {'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
            # M7
            {'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}},
        ]
        
        # Naive terms for the 3rd column of A (index 2, 5) and 3rd row of B (index 4, 5)
        # C0 += a2*b4, C1 += a2*b5, C2 += a5*b4, C3 += a5*b5
        naive_terms = [
            {'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}},
            {'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}},
            {'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}},
            {'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}},
        ]
        
        terms = strassen_terms + naive_terms
        
        for block in range(3):
            for t_idx, term in enumerate(terms):
                col = block * 11 + t_idx
                for idx, val in term['u'].items():
                    U[idx][col] = val
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx][col] = val
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx][col] = val
        return {"U": U, "V": V, "W": W}

    # Try to find Rank 30 (3 * 10)
    # Search for rank 10 decomposition of M_{2,3,2}
    
    def search_rank_10_m232():
        # Target tensors for M_{2,3,2}
        # A (2x3) flat: a0,a1,a2,a3,a4,a5
        # B (3x2) flat: b0,b1,b2,b3,b4,b5
        # C (2x2) flat: c0,c1,c2,c3
        
        targets = []
        for ik in range(4):
            M = np.zeros((6, 6))
            if ik == 0:
                M[0,0] = 1; M[1,2] = 1; M[2,4] = 1
            elif ik == 1:
                M[0,1] = 1; M[1,3] = 1; M[2,5] = 1
            elif ik == 2:
                M[3,0] = 1; M[4,2] = 1; M[5,4] = 1
            elif ik == 3:
                M[3,1] = 1; M[4,3] = 1; M[5,5] = 1
            targets.append(M)
            
        np.random.seed(42)
        
        # Local search
        # Initialize U, V with random sparse matrices
        # Minimize sum of squared residuals
        
        best_residual = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for trial in range(100):
            if time.time() > deadline - 15:
                break
                
            U = np.zeros((10, 6), dtype=int)
            V = np.zeros((10, 6), dtype=int)
            for r in range(10):
                nnz = np.random.choice([2, 3])
                cols = np.random.choice(6, size=nnz, replace=False)
                for c in cols:
                    U[r, c] = np.random.choice([-1, 1])
                    V[r, c] = np.random.choice([-1, 1])
            
            # Solve for W
            M_r = np.einsum('ri,rj->rij', U.astype(float), V.astype(float))
            M_r_flat = M_r.reshape(10, 36)
            A_mat = M_r_flat.T
            
            W_vals = []
            valid = True
            for ik in range(4):
                b_vec = targets[ik].flatten()
                W_real, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                if np.linalg.norm(A_mat @ W_real - b_vec) > 1e-8:
                    valid = False
                    break
                W_half = np.round(W_real * 2) / 2
                if not np.allclose(W_real, W_half, atol=1e-8):
                    valid = False
                    break
                W_vals.append(W_half)
                
            if valid:
                return U.tolist(), V.tolist(), [w.tolist() for w in W_vals]
                
            # If not valid, we could do a local search here, but for simplicity, we just try more random inits
            # A proper local search would be better, but time is tight.
            
        return None

    sol_10 = search_rank_10_m232()
    
    if sol_10:
        U_10, V_10, W_10 = sol_10
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            for r in range(10):
                col = block * 10 + r
                for i in range(6):
                    U[i][col] = U_10[r][i]
                for j in range(3):
                    for k in range(2):
                        l_idx = j * 2 + k
                        g_idx = j * p + (block * 2 + k)
                        V[g_idx][col] = V_10[r][l_idx]
                for i in range(2):
                    for k in range(2):
                        l_idx = i * 2 + k
                        g_idx = (block * 2 + k) * n + i
                        W[g_idx][col] = W_10[l_idx][r]
                        
        # Verify
        if time.time() < deadline - 5:
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U_arr[:, r], A_flat)
                v_val = np.dot(V_arr[:, r], B_flat)
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += W_arr[k * n + i, r] * u_val * v_val
                        
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U, "V": V, "W": W}
                
    return get_rank_33()
