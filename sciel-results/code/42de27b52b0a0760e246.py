"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    
    # Known Rank 18 Solution (Fallback)
    # 2 Strassen blocks (rank 7 each) for first 4 cols, 1 Schoolbook (rank 4) for last col
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    # Attempt to find better decomposition for 2x2x3 part
    # If we find rank r for 2x2x3, total rank is 7 + r
    # Target: r < 11 to beat rank 18
    
    n_sub = 2
    m_sub = 2
    p_sub = 3
    
    def error_223(params, rank):
        U = params[:n_sub*m_sub*rank].reshape((n_sub*m_sub, rank))
        V = params[n_sub*m_sub*rank : n_sub*m_sub*rank + m_sub*p_sub*rank].reshape((m_sub*p_sub, rank))
        W = params[n_sub*m_sub*rank + m_sub*p_sub*rank :].reshape((p_sub*n_sub, rank))
        
        err = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                        err += (val - target) ** 2
        return err

    def check_exact_223(U, V, W, rank, tol=1e-5):
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_find_223_decomposition(target_rank):
        if time.time() > DEADLINE - 10:
            return None
        
        # Try multiple restarts with different strategies
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                np.random.seed(restart * 101 + target_rank)
                x0 = np.random.uniform(-1, 1, size=n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
            elif restart % 3 == 1:
                np.random.seed(restart * 101 + target_rank + 1)
                x0 = np.random.uniform(-2, 2, size=n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
            else:
                np.random.seed(restart * 101 + target_rank + 2)
                # Sparse initialization
                x0 = np.zeros(n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank)
                x0[np.random.choice(len(x0), size=len(x0)//2, replace=False)] = np.random.choice([-1, 1], size=len(x0)//2)
            
            # Use Powell or L-BFGS-B for better convergence
            method = 'Powell' if restart % 2 == 0 else 'L-BFGS-B'
            options = {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False}
            if method == 'L-BFGS-B':
                 options = {'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}

            res = minimize(error_223, x0, args=(target_rank,), method=method, options=options)
            
            if res.fun < 1e-8:
                U = res.x[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
                V = res.x[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
                W = res.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))
                
                # Try snapping to half-integers
                U_r = snap_to_half_integers(U)
                V_r = snap_to_half_integers(V)
                W_r = snap_to_half_integers(W)
                
                if U_r is not None and V_r is not None and W_r is not None:
                    if check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                        return (U_r, V_r, W_r)
                
                # If raw float is exact, try harder snapping
                if check_exact_223(U, V, W, target_rank, tol=1e-6):
                    # Try rounding to integers first
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if check_exact_223(U_i, V_i, W_i, target_rank, tol=1e-6):
                        return (U_i, V_i, W_i)
                    
                    # Try half-integers with tighter tolerance
                    U_r2 = np.round(U * 2) / 2
                    V_r2 = np.round(V * 2) / 2
                    W_r2 = np.round(W * 2) / 2
                    if check_exact_223(U_r2, V_r2, W_r2, target_rank, tol=1e-6):
                        return (U_r2, V_r2, W_r2)
        
        return None

    # Try to find rank 9 for 2x2x3 (would give total rank 16)
    sub_decomp = None
    if time.time() < DEADLINE - 25:
        sub_decomp = try_find_223_decomposition(9)
    
    # Try to find rank 10 for 2x2x3 (would give total rank 17)
    if sub_decomp is None and time.time() < DEADLINE - 25:
        sub_decomp = try_find_223_decomposition(10)
        
    if sub_decomp:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 7 + rank_sub
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Copy Strassen block (cols 0,1 of B)
        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        
        # Copy sub-decomposition (cols 2,3,4 of B)
        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(rank_sub):
                    U[i*m_sub + j][7 + c] = U_sub[i*m_sub + j, c]
                    
        for j in range(m_sub):
            for k_sub in range(p_sub):
                k_global = 2 + k_sub
                for c in range(rank_sub):
                    V[j*p + k_global][7 + c] = V_sub[j*p_sub + k_sub, c]
                    
        for k_sub in range(p_sub):
            k_global = 2 + k_sub
            for i in range(n_sub):
                for c in range(rank_sub):
                    W[k_global*n + i][7 + c] = W_sub[k_sub*n_sub + i, c]
                    
        return {"U": U, "V": V, "W": W}
    else:
        return get_rank_18_solution()

# EVOLVE-BLOCK-END
