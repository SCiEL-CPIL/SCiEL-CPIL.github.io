import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 2, 4, 6
    
    # Fallback: Rank 42 solution using 6 Strassen blocks (3 blocks of AB+CD with rank 14)
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

    # Try to find Rank 36 solution using 3 blocks of Rank 12 (AB+CD)
    def find_abcd_decomp(rank_target=12):
        # Tensor for AB + CD where A,B,C,D are 2x2
        # X = (A, C) -> 8 entries
        # Y = (B, D) -> 8 entries
        # Z = E -> 4 entries
        
        T = np.zeros((8, 8, 4))
        
        # AB part: A is 0-3, B is 0-3, E is 0-3
        # A_idx = i*2+j, B_idx = j*2+k, E_idx = i*2+k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a_idx = i * 2 + j
                    b_idx = j * 2 + k
                    e_idx = i * 2 + k
                    T[a_idx, b_idx, e_idx] = 1.0
        
        # CD part: C is 4-7, D is 4-7, E is 0-3
        # C_idx = 4 + i*2+j, D_idx = 4 + j*2+k, E_idx = i*2+k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    c_idx = 4 + i * 2 + j
                    d_idx = 4 + j * 2 + k
                    e_idx = i * 2 + k
                    T[c_idx, d_idx, e_idx] = 1.0

        rank = rank_target
        
        def unpack(params):
            U = params[:8*rank].reshape(8, rank)
            V = params[8*rank:8*rank+8*rank].reshape(8, rank)
            W = params[8*rank+8*rank:].reshape(4, rank)
            return U, V, W

        def loss(params):
            U, V, W = unpack(params)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T)**2)
            return err

        def try_decomposition(U0, V0, W0, max_iter=2000, ftol=1e-14):
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(loss, x0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-8})
            
            if res.fun > 1e-6:
                return None
            
            U, V, W = unpack(res.x)
            
            # Polish
            res2 = minimize(loss, res.x, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-10})
            if res2.fun < res.fun:
                U, V, W = unpack(res2.x)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T, atol=1e-9):
                return U_snap, V_snap, W_snap
            
            # Snap to integers
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            T_recon_i = np.einsum('ar,br,cr->abc', U_snap_i, V_snap_i, W_snap_i)
            if np.allclose(T_recon_i, T, atol=1e-9):
                return U_snap_i, V_snap_i, W_snap_i
            
            return None

        best_result = None
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            scale = np.random.choice([0.01, 0.05, 0.1])
            U0 = np.random.randn(8, rank) * scale
            V0 = np.random.randn(8, rank) * scale
            W0 = np.random.randn(4, rank) * scale
            
            result = try_decomposition(U0, V0, W0)
            if result is not None:
                best_result = result
                break
        
        return best_result

    # Try rank 12 first
    if time.time() < deadline - 10:
        decomp = find_abcd_decomp(12)
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            rank_sub = U_sub.shape[1]
            total_rank = 3 * rank_sub
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Place 3 blocks for the 3 output column pairs
            for col_pair in range(3):
                col_offset = col_pair * rank_sub
                
                # A is 2x4, split into A_L (cols 0,1) and A_R (cols 2,3)
                # In the sub-problem, A is the first 4 entries of U_sub
                # C is the last 4 entries of U_sub
                # A_L corresponds to A in sub-problem?
                # Wait, the sub-problem is AB + CD.
                # In the full problem, for a fixed output column pair, we have:
                # C_out = A_L B_L + A_R B_R
                # Here A_L is 2x2, B_L is 2x2.
                # A_R is 2x2, B_R is 2x2.
                # So we map A_L -> A (first 4 of U_sub)
                # A_R -> C (last 4 of U_sub)
                # B_L -> B (first 4 of V_sub)
                # B_R -> D (last 4 of V_sub)
                # E -> W_sub (4 entries)
                
                for r in range(rank_sub):
                    # A_L (2x2) -> U rows 0,1 (i=0,1; j=0,1)
                    # A_L[i, j] is at U_sub[i*2+j, r]
                    # Full U index: i*4 + j
                    for i in range(2):
                        for j in range(2):
                            u_idx_full = i * 4 + j
                            U[u_idx_full][col_offset + r] += U_sub[i*2+j, r]
                            
                    # A_R (2x2) -> U rows 2,3 (i=0,1; j=2,3)
                    # A_R[i, j-2] is at U_sub[4 + i*2 + (j-2), r]
                    for i in range(2):
                        for j in range(2):
                            u_idx_full = i * 4 + (2 + j)
                            U[u_idx_full][col_offset + r] += U_sub[4 + i*2 + j, r]
                            
                    # B_L (2x2) -> V rows 0,1 (j=0,1; k=0,1)
                    # B_L[j, k] is at V_sub[j*2+k, r]
                    # Full V index: (2*col_pair + j)*6 + (2*col_pair + k)?
                    # No, V is 4x6.
                    # B is 4x6.
                    # B_L is top-left 2x2 of the 4x6 block for this col_pair?
                    # B rows 0,1, cols 2*col_pair, 2*col_pair+1
                    for j in range(2):
                        for k in range(2):
                            v_idx_full = j * 6 + (2 * col_pair + k)
                            V[v_idx_full][col_offset + r] += V_sub[j*2+k, r]
                            
                    # B_R (2x2) -> V rows 2,3 (j=2,3; k=0,1)
                    # B_R[j-2, k] is at V_sub[4 + (j-2)*2 + k, r]
                    for j in range(2):
                        for k in range(2):
                            v_idx_full = (2 + j) * 6 + (2 * col_pair + k)
                            V[v_idx_full][col_offset + r] += V_sub[4 + j*2 + k, r]
                            
                    # E (2x2) -> W
                    # E[i, k] is at W_sub[i*2+k, r]
                    # Full W index: (2*col_pair + k)*2 + i
                    for i in range(2):
                        for k in range(2):
                            w_idx_full = (2 * col_pair + k) * 2 + i
                            W[w_idx_full][col_offset + r] += W_sub[i*2+k, r]
            
            return {"U": U, "V": V, "W": W}

    return get_rank_42_solution()
