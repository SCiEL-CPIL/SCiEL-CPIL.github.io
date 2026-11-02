import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 2, 4, 6
    
    # Fallback: Rank 42 solution using 6 Strassen blocks
    def get_rank_42_solution():
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        strassen_U = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1}
        ]
        strassen_V = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        strassen_W = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]

        col_offset = 0
        for j_block in range(3):
            for side in ['L', 'R']:
                if side == 'L':
                    u_col_offset = 0
                    v_row_offset = 0
                else:
                    u_col_offset = 2
                    v_row_offset = 2

                for prod_idx in range(7):
                    col = col_offset + prod_idx

                    for local_idx, sign in strassen_U[prod_idx].items():
                        if local_idx == 0: u_idx = u_col_offset
                        elif local_idx == 1: u_idx = u_col_offset + 1
                        elif local_idx == 2: u_idx = 4 + u_col_offset
                        else: u_idx = 4 + u_col_offset + 1
                        U[u_idx][col] += sign

                    for local_idx, sign in strassen_V[prod_idx].items():
                        if local_idx == 0: v_idx = v_row_offset * 6 + 2 * j_block
                        elif local_idx == 1: v_idx = v_row_offset * 6 + 2 * j_block + 1
                        elif local_idx == 2: v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                        else: v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                        V[v_idx][col] += sign

                    for local_idx, sign in strassen_W[prod_idx].items():
                        if local_idx == 0: w_idx = 2 * (2 * j_block)
                        elif local_idx == 1: w_idx = 2 * (2 * j_block + 1)
                        elif local_idx == 2: w_idx = 2 * (2 * j_block) + 1
                        else: w_idx = 2 * (2 * j_block + 1) + 1
                        W[w_idx][col] += sign

                col_offset += 7

        return {"U": U, "V": V, "W": W}

    # Try to find Rank 35 solution using 5 blocks of Rank 7 (2x2x3)
    # Tensor for 2x2 A, 2x3 B -> 2x3 C has rank 7 (Winograd)
    def find_223_decomp():
        # Tensor for 2x2 A, 2x3 B -> 2x3 C
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a_idx = i * 2 + j
                    b_idx = j * 3 + k
                    c_idx = i * 3 + k
                    T[a_idx, b_idx, c_idx] = 1.0

        rank = 7
        
        def unpack(params):
            U = params[:4*rank].reshape(4, rank)
            V = params[4*rank:4*rank+6*rank].reshape(6, rank)
            W = params[4*rank+6*rank:].reshape(6, rank)
            return U, V, W

        def loss(params):
            U, V, W = unpack(params)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T)**2)
            return err

        def try_decomposition(U0, V0, W0, max_iter=3000, ftol=1e-14):
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(loss, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
            
            if res.fun > 1e-6:
                return None
            
            U, V, W = unpack(res.x)
            
            res2 = minimize(loss, res.x, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-10})
            if res2.fun < res.fun:
                U, V, W = unpack(res2.x)
            
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_snap, V_snap, W_snap
            
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            T_recon_i = np.einsum('ar,br,cr->abc', U_snap_i, V_snap_i, W_snap_i)
            if np.allclose(T_recon_i, T, atol=1e-9):
                return U_snap_i, V_snap_i, W_snap_i
            
            return None

        best_result = None
        
        # Structured initializations
        inits = []
        for r in range(rank):
            U0 = np.zeros((4, rank))
            V0 = np.zeros((6, rank))
            W0 = np.zeros((6, rank))
            U0[r % 4, r] = 1.0
            V0[r % 6, r] = 1.0
            W0[r % 6, r] = 1.0
            inits.append((U0, V0, W0))
            
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            np.random.seed(seed)
            scale = 0.1
            U0 = np.random.randn(4, rank) * scale
            V0 = np.random.randn(6, rank) * scale
            W0 = np.random.randn(6, rank) * scale
            inits.append((U0, V0, W0))
            
        for U0, V0, W0 in inits:
            if time.time() > deadline - 5:
                break
            result = try_decomposition(U0, V0, W0)
            if result is not None:
                best_result = result
                break
        
        return best_result

    if time.time() < deadline - 10:
        decomp = find_223_decomp()
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            rank_sub = U_sub.shape[1]
            total_rank = 4 * rank_sub
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Place 4 blocks
            # Block (side, col_half)
            # side: 0 (left A cols 0,1 / top B rows 0,1), 1 (right A cols 2,3 / bottom B rows 2,3)
            # col_half: 0 (left C cols 0,1,2), 1 (right C cols 3,4,5)
            
            for side in [0, 1]:
                for col_half in [0, 1]:
                    col_offset = (side * 2 + col_half) * rank_sub
                    # A offset
                    u_row_offset = side * 2
                    # B offset
                    v_row_offset = side * 2 # Rows of B
                    v_col_offset = col_half * 3 # Cols of B
                    # C offset
                    w_row_offset = col_half * 3 # Cols of C
                    
                    for r in range(rank_sub):
                        for i in range(2):
                            for j in range(2):
                                a_idx = i * 2 + j
                                u_idx = u_row_offset + a_idx
                                U[u_idx][col_offset + r] += U_sub[a_idx, r]
                                
                        for j in range(2):
                            for k in range(3):
                                b_idx = j * 3 + k
                                # B index in full: row = side*2 + j, col = col_half*3 + k
                                v_idx_full = (side * 2 + j) * 6 + (col_half * 3 + k)
                                V[v_idx_full][col_offset + r] += V_sub[b_idx, r]
                                
                        for i in range(2):
                            for k in range(3):
                                c_idx = i * 3 + k
                                # C index in full: row = i, col = col_half*3 + k
                                # W shape is (p*n, rank) = (12, rank).
                                # Convention: W[k*n+i, r]
                                # k is col of C (0..5), i is row of C (0..1)
                                # So w_idx = k * 2 + i
                                w_idx_full = (col_half * 3 + k) * 2 + i
                                W[w_idx_full][col_offset + r] += W_sub[c_idx, r]
            
            return {"U": U, "V": V, "W": W}

    return get_rank_42_solution()
