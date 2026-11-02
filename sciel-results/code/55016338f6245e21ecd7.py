"""Tensor decomposition for 2x2 by 2x5 matrix multiplication."""

import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    n, m, p = DIMENSIONS
    
    # Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a*m + b
                j = b*p + c
                k = c*n + a
                T_target[i,j,k] = 1.0

    # Function to check if U, V, W decompose T_target
    def check_decomposition(U, V, W, tol=1e-9):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        T_calc = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
        return np.allclose(T_calc, T_target, atol=tol)

    # Function to snap to integers or half-integers
    def snap_to_half_int(arr):
        # Multiply by 2, round to nearest integer, divide by 2
        return np.round(arr * 2) / 2.0

    def snap_to_int(arr):
        return np.round(arr)

    # Generate the known Rank-18 solution
    def get_rank18_solution():
        rank = 18
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_strassen_block(t_offset, k_start):
            U[0][t_offset] = 1
            U[3][t_offset] = 1
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
            V[k0][t_offset] = 1
            V[p+k1][t_offset] = 1
            V[k0][t_offset+1] = 1
            V[k1][t_offset+2] = 1
            V[p+k1][t_offset+2] = -1
            V[p+k0][t_offset+3] = 1
            V[k0][t_offset+3] = -1
            V[p+k1][t_offset+4] = 1
            V[k0][t_offset+5] = 1
            V[k1][t_offset+5] = 1
            V[p+k0][t_offset+6] = 1
            V[p+k1][t_offset+6] = 1
            
            W[k0*n][t_offset] = 1
            W[k1*n+1][t_offset] = 1
            W[k0*n+1][t_offset+1] = 1
            W[k1*n+1][t_offset+1] = -1
            W[k1*n][t_offset+2] = 1
            W[k1*n+1][t_offset+2] = 1
            W[k0*n][t_offset+3] = 1
            W[k0*n+1][t_offset+3] = 1
            W[k0*n][t_offset+4] = -1
            W[k1*n][t_offset+4] = 1
            W[k1*n+1][t_offset+5] = 1
            W[k0*n][t_offset+6] = 1

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
        
        return U, V, W

    U18, V18, W18 = get_rank18_solution()

    # Try to find a decomposition of rank R
    def find_decomposition(rank, max_iters=200, time_limit=20.0, init_U=None, init_V=None, init_W=None):
        if init_U is None:
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
        else:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
            
        # Normalize
        for i in range(rank):
            if np.linalg.norm(U[:, i]) > 1e-9:
                U[:, i] /= np.linalg.norm(U[:, i])
            if np.linalg.norm(V[:, i]) > 1e-9:
                V[:, i] /= np.linalg.norm(V[:, i])
            if np.linalg.norm(W[:, i]) > 1e-9:
                W[:, i] /= np.linalg.norm(W[:, i])
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > start_time + time_limit:
                break
                
            # Update U
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(V[:, r], W[:, r])
                Num = np.einsum('ijk,jk->i', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    U[:, r] = Num / Den
            
            # Update V
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(U[:, r], W[:, r])
                Num = np.einsum('ijk,ik->j', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    V[:, r] = Num / Den

            # Update W
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            Residual = T_target - T_calc
            for r in range(rank):
                Term_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res_r = Residual + Term_r
                M_r = np.outer(U[:, r], V[:, r])
                Num = np.einsum('ijk,ij->k', Res_r, M_r)
                Den = np.sum(M_r**2)
                if Den > 1e-12:
                    W[:, r] = Num / Den
            
            T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T_calc - T_target)
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if error < 1e-8:
                break

        # Polish with least squares if error is small
        if best_error < 1e-3:
            def flatten_params(u, v, w):
                return np.concatenate([u.flatten(), v.flatten(), w.flatten()])
            
            def objective(params):
                u = params[:n*m*rank].reshape(n*m, rank)
                v = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                w = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                t_calc = np.einsum('ir,jr,kr->ijk', u, v, w)
                return t_calc - T_target.flatten()
                
            x0 = flatten_params(best_U, best_V, best_W)
            try:
                res = least_squares(objective, x0, method='lm', max_nfev=1000)
                if res.success:
                    u_pol = res.x[:n*m*rank].reshape(n*m, rank)
                    v_pol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    w_pol = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Check if polished solution is better
                    t_pol = np.einsum('ir,jr,kr->ijk', u_pol, v_pol, w_pol)
                    err_pol = np.linalg.norm(t_pol - T_target)
                    if err_pol < best_error:
                        best_U, best_V, best_W = u_pol, v_pol, w_pol
                        best_error = err_pol
            except Exception:
                pass

        # Snap to half-integers and check
        U_snap = snap_to_half_int(best_U)
        V_snap = snap_to_half_int(best_V)
        W_snap = snap_to_half_int(best_W)
        
        if check_decomposition(U_snap, V_snap, W_snap, tol=1e-9):
            return U_snap, V_snap, W_snap, True
        
        # Try snapping to integers
        U_int = snap_to_int(best_U)
        V_int = snap_to_int(best_V)
        W_int = snap_to_int(best_W)
        
        if check_decomposition(U_int, V_int, W_int, tol=1e-9):
            return U_int, V_int, W_int, True
            
        return best_U, best_V, best_W, False

    # Search for lower ranks, starting from 17 down to 13
    # Use the rank-18 solution to initialize the search for lower ranks
    # We can try to drop columns from the rank-18 solution and let ALS recover
    
    for rank in range(17, 12, -1):
        if time.time() > deadline - margin:
            break
            
        # Strategy: Initialize by taking the first 'rank' columns of the rank-18 solution
        # and adding small noise. This helps ALS converge to a nearby low-rank solution.
        if rank < 18:
            U_init = U18[:, :rank] + np.random.randn(n*m, rank) * 0.1
            V_init = V18[:, :rank] + np.random.randn(m*p, rank) * 0.1
            W_init = W18[:, :rank] + np.random.randn(p*n, rank) * 0.1
        else:
            U_init = None
            V_init = None
            W_init = None

        U, V, W, success = find_decomposition(rank, max_iters=100, time_limit=15.0, init_U=U_init, init_V=V_init, init_W=W_init)
        if success:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to hardcoded rank 18
    U, V, W = get_rank18_solution()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
