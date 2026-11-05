import numpy as np
import time

def construct():
    deadline = time.time() + 115
    
    n, m, p = 2, 4, 5
    
    def build_rank36():
        """Build the known rank-36 construction."""
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        
        def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
            nonlocal col
            idx_p00 = a_row * m + a_col
            idx_p01 = a_row * m + (a_col + 1)
            idx_p10 = (a_row + 1) * m + a_col
            idx_p11 = (a_row + 1) * m + (a_col + 1)
            
            idx_q00 = b_row * p + b_col
            idx_q01 = b_row * p + (b_col + 1)
            idx_q10 = (b_row + 1) * p + b_col
            idx_q11 = (b_row + 1) * p + (b_col + 1)
            
            idx_c00 = c_col * n + c_row
            idx_c01 = (c_col + 1) * n + c_row
            idx_c10 = c_col * n + (c_row + 1)
            idx_c11 = (c_col + 1) * n + (c_row + 1)
            
            U[idx_p00, col] += 1
            U[idx_p11, col] += 1
            V[idx_q00, col] += 1
            V[idx_q11, col] += 1
            W[idx_c00, col] += 1
            W[idx_c11, col] += 1
            col += 1
            
            U[idx_p10, col] += 1
            U[idx_p11, col] += 1
            V[idx_q00, col] += 1
            W[idx_c10, col] += 1
            W[idx_c11, col] -= 1
            col += 1
            
            U[idx_p00, col] += 1
            V[idx_q01, col] += 1
            V[idx_q11, col] -= 1
            W[idx_c01, col] += 1
            W[idx_c11, col] += 1
            col += 1
            
            U[idx_p11, col] += 1
            V[idx_q10, col] += 1
            V[idx_q00, col] -= 1
            W[idx_c00, col] += 1
            W[idx_c10, col] += 1
            col += 1
            
            U[idx_p00, col] += 1
            U[idx_p01, col] += 1
            V[idx_q11, col] += 1
            W[idx_c00, col] -= 1
            W[idx_c01, col] += 1
            col += 1
            
            U[idx_p10, col] += 1
            U[idx_p00, col] -= 1
            V[idx_q00, col] += 1
            V[idx_q01, col] += 1
            W[idx_c11, col] += 1
            col += 1
            
            U[idx_p01, col] += 1
            U[idx_p11, col] -= 1
            V[idx_q10, col] += 1
            V[idx_q11, col] += 1
            W[idx_c00, col] += 1
            col += 1

        def add_schoolbook_221(a_row, a_col, b_row, b_col, c_row, c_col):
            nonlocal col
            idx_p00 = a_row * m + a_col
            idx_p01 = a_row * m + (a_col + 1)
            idx_p10 = (a_row + 1) * m + a_col
            idx_p11 = (a_row + 1) * m + (a_col + 1)
            
            idx_q0 = b_row * p + b_col
            idx_q1 = (b_row + 1) * p + b_col
            
            idx_c0 = c_col * n + c_row
            idx_c1 = c_col * n + (c_row + 1)
            
            U[idx_p00, col] += 1
            V[idx_q0, col] += 1
            W[idx_c0, col] += 1
            col += 1
            
            U[idx_p01, col] += 1
            V[idx_q1, col] += 1
            W[idx_c0, col] += 1
            col += 1
            
            U[idx_p10, col] += 1
            V[idx_q0, col] += 1
            W[idx_c1, col] += 1
            col += 1
            
            U[idx_p11, col] += 1
            V[idx_q1, col] += 1
            W[idx_c1, col] += 1
            col += 1

        add_strassen(0, 0, 0, 0, 0, 0)
        add_strassen(0, 2, 2, 0, 0, 0)
        add_strassen(0, 0, 0, 2, 0, 2)
        add_strassen(0, 2, 2, 2, 0, 2)
        add_schoolbook_221(0, 0, 0, 4, 0, 4)
        add_schoolbook_221(0, 2, 2, 4, 0, 4)
        
        return U, V, W, rank

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        T = np.zeros((nm, mp, pn))
        for i in range(nm):
            a_row = i // m
            a_col = i % m
            for j in range(mp):
                b_row = j // p
                b_col = j % p
                if a_row == b_row and a_col == b_col:
                    c_col = a_col
                    c_row = a_row
                    k = c_col * n + c_row
                    T[i, j, k] = 1.0
        
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def try_numerical_search(rank, time_budget):
        """Try to find a rank-r decomposition via numerical optimization."""
        t_start = time.time()
        nm = n * m
        mp = m * p
        pn = p * n
        
        T = np.zeros((nm, mp, pn))
        for i in range(nm):
            a_row = i // m
            a_col = i % m
            for j in range(mp):
                b_row = j // p
                b_col = j % p
                if a_row == b_row and a_col == b_col:
                    c_col = a_col
                    c_row = a_row
                    k = c_col * n + c_row
                    T[i, j, k] = 1.0
        
        T_flat = T.flatten()
        
        best_err = np.inf
        best_params = None
        
        np.random.seed(42)
        
        for attempt in range(30):
            if time.time() - t_start > time_budget:
                break
            
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for step in range(100):
                if time.time() - t_start > time_budget:
                    break
                
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                recon_flat = recon.flatten()
                err = np.sum((recon_flat - T_flat) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-12:
                    break
                
                diff = recon - T
                gU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                gV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                gW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                vU = momentum * vU + lr * gU
                vV = momentum * vV + lr * gV
                vW = momentum * vW + lr * gW
                
                U -= vU
                V -= vV
                W -= vW
            
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            recon_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
            err_snap = np.sum((recon_snap.flatten() - T_flat) ** 2)
            
            if err_snap < best_err:
                best_err = err_snap
                best_params = (U_snap.copy(), V_snap.copy(), W_snap.copy())
            
            if best_err < 1e-10:
                break
        
        if best_params is not None and best_err < 1e-9:
            return best_params[0], best_params[1], best_params[2], rank
        return None
    
    U36, V36, W36, rank36 = build_rank36()
    
    best_U, best_V, best_W, best_rank = U36, V36, W36, rank36
    
    if time.time() < deadline - 5:
        result = try_numerical_search(35, 30)
        if result is not None:
            U, V, W, r = result
            if verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, r
    
    if best_rank == 35 and time.time() < deadline - 5:
        result = try_numerical_search(34, 30)
        if result is not None:
            U, V, W, r = result
            if verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, r
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
