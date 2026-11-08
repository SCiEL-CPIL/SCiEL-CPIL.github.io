import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 33 solution (current best)
    def get_rank_33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        terms = []
        # Strassen for 2x2
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
        # Naive for 2x1x2
        terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
        terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
        terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
        terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
        
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
    # U: 10 x 6, V: 10 x 6
    # Target: M_{2,3,2}
    
    def search_rank_10_m232():
        # Target tensors for M_{2,3,2}
        # A (2x3) flat: a0,a1,a2,a3,a4,a5
        # B (3x2) flat: b0,b1,b2,b3,b4,b5
        # C (2x2) flat: c0,c1,c2,c3 (c0=00, c1=01, c2=10, c3=11)
        
        # Target matrices M_ik (6x6)
        # c0 = a0*b0 + a1*b2 + a2*b4
        # c1 = a0*b1 + a1*b3 + a2*b5
        # c2 = a3*b0 + a4*b2 + a5*b4
        # c3 = a3*b1 + a4*b3 + a5*b5
        
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
            
        # Random search
        np.random.seed(123)
        for _ in range(5000):
            if time.time() > deadline - 10:
                return None
                
            U = np.random.choice([-1, 0, 1], size=(10, 6))
            V = np.random.choice([-1, 0, 1], size=(10, 6))
            
            # Compute M_r = u_r v_r^T (outer product of coeffs)
            # M_r[i, j] = U[r, i] * V[r, j]
            M_r = np.einsum('ri,rj->rij', U, V) # 10 x 6 x 6
            
            # Solve for W
            W_valid = True
            W_vals = []
            for ik in range(4):
                # System: sum_r W_ik_r * M_r[:,:,r] = target_ik
                # A_mat: 36 x 10
                A_mat = M_r.reshape(10, 36).T # 36 x 10
                b_vec = targets[ik].flatten()
                
                # Check consistency and solve
                # We want integer or half-integer solution
                # Use least squares to find real solution, then check if it's close to int/half-int
                W_real, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if residuals are small
                if np.linalg.norm(A_mat @ W_real - b_vec) > 1e-6:
                    W_valid = False
                    break
                    
                # Check if W_real is close to half-integers
                W_half = np.round(W_real * 2) / 2
                if not np.allclose(W_real, W_half, atol=1e-6):
                    W_valid = False
                    break
                W_vals.append(W_half)
                
            if W_valid:
                # Construct U, V, W for M_{2,3,2}
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = [w.tolist() for w in W_vals]
                return U_list, V_list, W_list
                
        return None

    sol_10 = search_rank_10_m232()
    
    if sol_10:
        U_10, V_10, W_10 = sol_10
        # Construct M_{2,3,6} from three M_{2,3,2} blocks
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            for r in range(10):
                col = block * 10 + r
                # U is same for all blocks (depends on A)
                for i in range(6):
                    U[i][col] = U_10[r][i]
                    
                # V: local index j*2+k -> global j*6 + (block*2+k)
                for j in range(3):
                    for k in range(2):
                        l_idx = j * 2 + k
                        g_idx = j * p + (block * 2 + k)
                        V[g_idx][col] = V_10[r][l_idx]
                        
                # W: local index i*2+k -> global (block*2+k)*n + i
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
# EVOLVE-BLOCK-END
