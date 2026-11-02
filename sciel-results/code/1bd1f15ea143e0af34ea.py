import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    First attempts rank 20 via perturbation search from Strassen's rank-21 solution.
    Falls back to rank 21 if rank 20 is not found.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j      # i*2+j
                b_idx = j * p + k      # j*6+k
                c_idx = k * n + i      # k*2+i (W[k*n+i] convention)
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Strassen steps for 2x2 multiplication
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Build rank-21 Strassen decomposition
    def build_strassen_21():
        rank = 21
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        v_indices = []
        w_indices = []
        for g in range(3):
            k0 = 2 * g
            k1 = 2 * g + 1
            v_indices.append([
                0 * 6 + k0, 
                0 * 6 + k1, 
                1 * 6 + k0, 
                1 * 6 + k1
            ])
            w_indices.append([
                k0 * 2 + 0,
                k1 * 2 + 0,
                k0 * 2 + 1,
                k1 * 2 + 1
            ])
        
        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                step = strassen_steps[s]
                u_list = step['u']
                v_list = step['v']
                w_list = step['w']
                v_idx_list = v_indices[g]
                w_idx_list = w_indices[g]
                
                for local_idx in range(4):
                    u_val = u_list[local_idx]
                    if u_val != 0:
                        U[local_idx, r] = u_val
                    v_val = v_list[local_idx]
                    if v_val != 0:
                        V[v_idx_list[local_idx], r] = v_val
                    w_val = w_list[local_idx]
                    if w_val != 0:
                        W[w_idx_list[local_idx], r] = w_val
        
        return U, V, W
    
    U21, V21, W21 = build_strassen_21()
    
    # Verify decomposition
    def verify_decomposition(U, V, W, T, tol=1e-10):
        R = U.shape[1]
        T_recon = np.zeros_like(T)
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_recon, T, atol=tol)
    
    if not verify_decomposition(U21, V21, W21, T):
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    # Try to find rank-20 by removing one term and absorbing into others
    R21 = 21
    best_rank = 21
    best_U, best_V, best_W = U21, V21, W21
    
    num_eq = nm * mp * pn  # 576
    
    for t in range(R21):
        if time.time() > deadline:
            break
        
        keep = [r for r in range(R21) if r != t]
        
        # Target: the removed rank-1 tensor
        target = np.einsum('i,j,k->ijk', U21[:, t], V21[:, t], W21[:, t])
        target_flat = target.flatten()
        
        # Build linear system for first-order perturbation
        # Unknowns: du_r (nm each), dv_r (mp each), dw_r (pn each) for 20 terms
        num_unknowns = len(keep) * (nm + mp + pn)
        A_mat = np.zeros((num_eq, num_unknowns))
        
        col = 0
        for r in keep:
            ur = U21[:, r]
            vr = V21[:, r]
            wr = W21[:, r]
            
            # du_r ⊗ v_r ⊗ w_r
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        eq_idx = i * mp * pn + j * pn + k
                        A_mat[eq_idx, col + i] = vr[j] * wr[k]
            col += nm
            
            # u_r ⊗ dv_r ⊗ w_r
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        eq_idx = i * mp * pn + j * pn + k
                        A_mat[eq_idx, col + j] = ur[i] * wr[k]
            col += mp
            
            # u_r ⊗ v_r ⊗ dw_r
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        eq_idx = i * mp * pn + j * pn + k
                        A_mat[eq_idx, col + k] = ur[i] * vr[j]
            col += pn
        
        # Solve least squares
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, target_flat, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check residual
        A_sol = A_mat @ sol
        residual = np.linalg.norm(A_sol - target_flat)
        
        if residual < 1e-8:
            # Apply perturbation
            U_new = np.zeros((nm, 20))
            V_new = np.zeros((mp, 20))
            W_new = np.zeros((pn, 20))
            
            col = 0
            for idx, r in enumerate(keep):
                du = sol[col:col+nm]
                col += nm
                dv = sol[col:col+mp]
                col += mp
                dw = sol[col:col+pn]
                col += pn
                
                U_new[:, idx] = U21[:, r] + du
                V_new[:, idx] = V21[:, r] + dv
                W_new[:, idx] = W21[:, r] + dw
            
            if verify_decomposition(U_new, V_new, W_new, T, tol=1e-9):
                best_rank = 20
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    if best_rank == 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
