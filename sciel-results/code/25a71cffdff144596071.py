import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    n, m, p = 2, 4, 6

    # Known rank-14 decomposition for 2x4 @ 4x6 using block structure
    def get_rank_14_solution():
        rank_total = 14
        U = [[0] * rank_total for _ in range(n * m)]  # 8 x 14
        V = [[0] * rank_total for _ in range(m * p)]  # 24 x 14
        W = [[0] * rank_total for _ in range(p * n)]  # 12 x 14
        
        # We use a block-diagonal approach with 2x2 Strassen-like blocks
        # The 2x4 @ 4x6 can be seen as 2x(2*2) @ (2*2)x(2*3)
        # Split B into two 4x3 blocks, and A into two 2x2 blocks
        
        # For each of the two 2x2 A blocks and two 4x3 B blocks:
        # We need to multiply 2x2 @ 4x3, which is 2x(2x2) @ (2x2)x(2x3)
        # This can be done with rank-7 Strassen for each pair, but we need to be careful
        
        # Actually, let's use a simpler approach: direct Kronecker product structure
        # 2x4 @ 4x6 = (I_2 ⊗ A_2x4) @ (A_2x4 ⊗ I_6) ... no
        
        # Let's use the fact that 2x4 @ 4x6 can be decomposed as:
        # Split the 4 middle dimension into 2+2
        # A = [A1, A2] where A1, A2 are 2x2
        # B = [B1; B2] where B1, B2 are 2x6
        # Result = A1*B1 + A2*B2
        
        # Each A_i*B_i is 2x2 @ 2x6, which can be done with rank-7 Strassen
        # So total rank = 2 * 7 = 14
        
        strassen_U_2x2 = [
            {0: 1, 1: 1}, {2: 1, 3: 1}, {0: 1}, {1: 1},
            {0: 1, 2: 1}, {0: -1, 3: 1}, {1: 1, 3: -1}
        ]
        strassen_V_2x2 = [
            {0: 1, 1: 1}, {0: 1}, {2: 1, 3: -1}, {1: 1, 0: -1},
            {1: 1}, {0: 1, 2: 1}, {2: 1, 3: 1}
        ]
        strassen_W_2x2 = [
            {0: 1, 1: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {1: 1}, {0: 1}
        ]
        
        for block_idx in range(2):
            col_offset = block_idx * 7
            
            # A_block is rows block_idx*2 to block_idx*2+1 of A (2x2)
            # B_block is rows block_idx*2 to block_idx*2+1 of B (2x6)
            # Result is 2x6
            
            for prod_idx in range(7):
                col = col_offset + prod_idx
                
                # U: 2x2 block of A
                for local_idx, sign in strassen_U_2x2[prod_idx].items():
                    u_idx = block_idx * 2 + local_idx
                    U[u_idx][col] += sign
                
                # V: 2x6 block of B, split into 2x2 @ 3
                # B_block is 2x6, we can think of it as 2x(2x3)
                # For each 2x2 sub-block in the 6 columns
                for col_group in range(3):
                    for local_idx, sign in strassen_V_2x2[prod_idx].items():
                        if local_idx < 2:
                            # First 2 columns of this 2x2 sub-block
                            v_idx = block_idx * 2 * 6 + local_idx * 3 + col_group * 2
                        else:
                            # Second 2 columns
                            v_idx = block_idx * 2 * 6 + (local_idx - 2) * 3 + col_group * 2 + 1
                        V[v_idx][col] += sign
                
                # W: 2x6 result
                for col_group in range(3):
                    for local_idx, sign in strassen_W_2x2[prod_idx].items():
                        w_idx = col_group * 2 + local_idx
                        W[w_idx][col] += sign
        
        return {"U": U, "V": V, "W": W}
    
    def get_rank_42_solution():
        # Same as before, rank 42
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

    def find_abcd_decomp(rank_target):
        # Tensor for AB + CD
        T = np.zeros((8, 8, 4))
        
        # AB part
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a_idx = i * 2 + j
                    b_idx = j * 2 + k
                    e_idx = i * 2 + k
                    T[a_idx, b_idx, e_idx] = 1.0
        
        # CD part
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

        def try_decomposition(U0, V0, W0):
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            res = minimize(loss, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
            
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
        # Try more seeds for rank 11
        num_seeds = 50 if rank_target == 11 else 20
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            # Sparse random initialization might help
            U0 = np.random.randn(8, rank) * 0.1
            V0 = np.random.randn(8, rank) * 0.1
            W0 = np.random.randn(4, rank) * 0.1
            
            result = try_decomposition(U0, V0, W0)
            if result is not None:
                best_result = result
                break
        
        return best_result

    def build_full_solution(U_sub, V_sub, W_sub, rank_sub):
        total_rank = 3 * rank_sub
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for col_pair in range(3):
            col_offset = col_pair * rank_sub
            
            for r in range(rank_sub):
                # A_L (2x2) -> U rows 0,1
                for i in range(2):
                    for j in range(2):
                        u_idx_full = i * 4 + j
                        U[u_idx_full][col_offset + r] += U_sub[i*2+j, r]
                        
                # A_R (2x2) -> U rows 2,3
                for i in range(2):
                    for j in range(2):
                        u_idx_full = i * 4 + (2 + j)
                        U[u_idx_full][col_offset + r] += U_sub[4 + i*2 + j, r]
                        
                # B_L (2x2) -> V rows 0,1
                for j in range(2):
                    for k in range(2):
                        v_idx_full = j * 6 + (2 * col_pair + k)
                        V[v_idx_full][col_offset + r] += V_sub[j*2+k, r]
                        
                # B_R (2x2) -> V rows 2,3
                for j in range(2):
                    for k in range(2):
                        v_idx_full = (2 + j) * 6 + (2 * col_pair + k)
                        V[v_idx_full][col_offset + r] += V_sub[4 + j*2 + k, r]
                        
                # E (2x2) -> W
                for i in range(2):
                    for k in range(2):
                        w_idx_full = (2 * col_pair + k) * 2 + i
                        W[w_idx_full][col_offset + r] += W_sub[i*2+k, r]
            
        return {"U": U, "V": V, "W": W}

    # Try rank 11 first
    if time.time() < deadline - 20:
        decomp = find_abcd_decomp(11)
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            rank_sub = U_sub.shape[1]
            return build_full_solution(U_sub, V_sub, W_sub, rank_sub)

    # Try rank 12
    if time.time() < deadline - 10:
        decomp = find_abcd_decomp(12)
        if decomp is not None:
            U_sub, V_sub, W_sub = decomp
            rank_sub = U_sub.shape[1]
            return build_full_solution(U_sub, V_sub, W_sub, rank_sub)

    return get_rank_42_solution()
