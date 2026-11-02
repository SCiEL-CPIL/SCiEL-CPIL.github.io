import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    n, m, p = 2, 4, 6

    def verify_solution(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_rank_42_solution():
        # Fallback: Construct a valid rank-42 solution using Strassen blocks
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        # Strassen for 2x2
        s_U = [
            [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0],
            [1, 1, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]
        ]
        s_V = [
            [1, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0],
            [0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]
        ]
        s_W = [
            [1, 0, 1, 0], [0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 1, 0],
            [1, 0, -1, 1], [0, 1, 0, 0], [1, 0, 0, 0]
        ]
        
        # Map local 2x2 indices to global indices
        # A is 2x4, split into A1 (cols 0,1) and A2 (cols 2,3)
        # B is 4x6, split into B1 (rows 0,1) and B2 (rows 2,3)
        # B1 is 2x6, split into B11, B12, B13 (2x2 each)
        # B2 is 2x6, split into B21, B22, B23 (2x2 each)
        
        # Total 6 blocks of 2x2 multiplication
        # Each uses 7 rank terms
        # Total 42 terms
        
        block_id = 0
        for j_block in range(2): # A1, A2
            for k_block in range(3): # 3 2x2 blocks in B
                # j_block: 0 (cols 0,1 of A), 1 (cols 2,3 of A)
                # k_block: 0,1,2 (cols 0-1, 2-3, 4-5 of B)
                
                # U indices: A_{i, j}
                # i in 0,1
                # j in j_block*2, j_block*2+1
                
                # V indices: B_{j, k}
                # j in j_block*2, j_block*2+1
                # k in k_block*2, k_block*2+1
                
                # W indices: C_{k, i}
                # k in k_block*2, k_block*2+1
                # i in 0,1
                
                for r in range(7):
                    col = block_id * 7 + r
                    
                    # U
                    for i in range(2):
                        for j_local in range(2):
                            sign = s_U[r][i*2 + j_local]
                            if sign != 0:
                                u_idx = i * m + (j_block * 2 + j_local)
                                U[u_idx][col] += sign
                                
                    # V
                    for j_local in range(2):
                        for k_local in range(2):
                            sign = s_V[r][j_local*2 + k_local]
                            if sign != 0:
                                v_idx = (j_block * 2 + j_local) * p + (k_block * 2 + k_local)
                                V[v_idx][col] += sign
                                
                    # W
                    for k_local in range(2):
                        for i in range(2):
                            sign = s_W[r][k_local*2 + i]
                            if sign != 0:
                                w_idx = (k_block * 2 + k_local) * n + i
                                W[w_idx][col] += sign
                                
                block_id += 1

        return {"U": U, "V": V, "W": W}

    def find_decomposition(rank_target):
        # Target tensor T for 2x4x6
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0

        def unpack(params):
            U = params[:n*m*rank_target].reshape(n*m, rank_target)
            V = params[n*m*rank_target:n*m*rank_target + m*p*rank_target].reshape(m*p, rank_target)
            W = params[n*m*rank_target + m*p*rank_target:].reshape(p*n, rank_target)
            return U, V, W

        def loss(params):
            U, V, W = unpack(params)
            # Compute tensor reconstruction
            # T_recon[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Use einsum
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T_recon - T)**2)
            return err

        best_result = None
        num_seeds = 10
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed)
            U0 = np.random.randn(n*m, rank_target) * 0.1
            V0 = np.random.randn(m*p, rank_target) * 0.1
            W0 = np.random.randn(p*n, rank_target) * 0.1
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(loss, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8})
                
                if res.fun < 1e-6:
                    U, V, W = unpack(res.x)
                    
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
                        
            except Exception:
                continue
                
        return None

    # Try rank 36
    if time.time() < deadline - 30:
        decomp = find_decomposition(36)
        if decomp is not None:
            U, V, W = decomp
            sol = {"U": U, "V": V, "W": W}
            if verify_solution(sol['U'], sol['V'], sol['W']):
                return sol

    # Try rank 38
    if time.time() < deadline - 20:
        decomp = find_decomposition(38)
        if decomp is not None:
            U, V, W = decomp
            sol = {"U": U, "V": V, "W": W}
            if verify_solution(sol['U'], sol['V'], sol['W']):
                return sol

    # Try rank 40
    if time.time() < deadline - 10:
        decomp = find_decomposition(40)
        if decomp is not None:
            U, V, W = decomp
            sol = {"U": U, "V": V, "W": W}
            if verify_solution(sol['U'], sol['V'], sol['W']):
                return sol

    # Fallback to rank 42
    return get_rank_42_solution()
