"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    def check_decomposition(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for t in range(rank):
                            val += U[i*m + j][t] * V[jp*p + k][t] * W[k*n + i][t]
                        if abs(val - target) > tol:
                            return False
        return True
    
    # Known rank-18 solution (fallback)
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
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
    
    # Try ALS-based search for lower rank
    def try_als_search(target_rank, max_restarts=10):
        if time.time() > DEADLINE - 10:
            return None
        
        n_sub, m_sub, p_sub = n, m, p
        
        def error_func(params, rank):
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
        
        def check_exact(U, V, W, rank, tol=1e-6):
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
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            np.random.seed(restart * 101 + target_rank)
            total_vars = n_sub*m_sub*target_rank + m_sub*p_sub*target_rank + p_sub*n_sub*target_rank
            x0 = np.random.uniform(-1, 1, size=total_vars)
            
            from scipy.optimize import minimize
            res = minimize(error_func, x0, args=(target_rank,), method='Powell', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'disp': False})
            
            if res.fun < 1e-8:
                U = res.x[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
                V = res.x[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
                W = res.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))
                
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_exact(U_r, V_r, W_r, target_rank, tol=1e-6):
                    return (U_r, V_r, W_r)
                
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                
                if check_exact(U_i, V_i, W_i, target_rank, tol=1e-6):
                    return (U_i, V_i, W_i)
        
        return None
    
    # Try rank 15
    if time.time() < DEADLINE - 20:
        result = try_als_search(15, max_restarts=8)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 16
    if time.time() < DEADLINE - 10:
        result = try_als_search(16, max_restarts=5)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 18
    return get_rank_18_solution()

# EVOLVE-BLOCK-END
