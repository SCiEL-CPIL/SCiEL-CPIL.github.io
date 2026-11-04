import numpy as np
import time
from fractions import Fraction
from itertools import product
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Build the matrix multiplication tensor T[u,v,w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    schoolbook_rank = n * m * p
    min_rank = max(n * p, m * p, n * m)
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
    
    def check_solution(U, V, W, tol=1e-9):
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T)
        return err < tol
    
    def snap_to_half_int(U, V, W):
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        return U_int, V_int, W_int
    
    def snap_to_int(U, V, W):
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        return U_int, V_int, W_int
    
    def polish_solution(U, V, W, max_iters=100, tol=1e-10):
        """Refine solution with alternating least squares until convergence."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update W
            A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
            T_W = T.reshape(n * m * m * p, p * n)
            try:
                W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                W = W_T.T
            except:
                continue
            
            # Update U
            A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
            T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
            try:
                U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                U = U_T.T
            except:
                continue
            
            # Update V
            A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
            T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
            try:
                V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                V = V_T.T
            except:
                continue
            
            if it % 10 == 0:
                T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                err = np.linalg.norm(T_calc - T)
                if err < tol:
                    break
        
        return U, V, W
    
    def rational_reconstruct(vals, max_den=16):
        """Try to find rational numbers with small denominators close to vals."""
        result = []
        for v in vals:
            best_frac = None
            best_diff = float('inf')
            for den in range(1, max_den + 1):
                num = round(v * den)
                frac_val = Fraction(num, den)
                diff = abs(float(frac_val) - v)
                if diff < best_diff:
                    best_diff = diff
                    best_frac = float(frac_val)
            result.append(best_frac)
        return np.array(result)
    
    def solve_als(r, max_iters=80, restarts=12, use_structured_init=False, seed_offset=0):
        """Alternating least squares for rank-r tensor decomposition."""
        best_result = None
        best_err = float('inf')
        
        for restart in range(restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + seed_offset + restart)
            
            if use_structured_init:
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                for t in range(r):
                    i = np.random.randint(n)
                    j = np.random.randint(m) 
                    k = np.random.randint(p)
                    
                    for ii in range(n):
                        for jj in range(m):
                            U[ii * m + jj, t] = (1.0 if ii == i and jj == j else 0.0) + 0.1 * np.random.randn()
                    
                    for jj in range(m):
                        for kk in range(p):
                            V[jj * p + kk, t] = (1.0 if jj == j and kk == k else 0.0) + 0.1 * np.random.randn()
                    
                    for kk in range(p):
                        for ii in range(n):
                            W[kk * n + ii, t] = (1.0 if kk == k and ii == i else 0.0) + 0.1 * np.random.randn()
            else:
                U = np.random.uniform(-1, 1, (n * m, r))
                V = np.random.uniform(-1, 1, (m * p, r))
                W = np.random.uniform(-1, 1, (p * n, r))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break

                # Update W
                A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
                T_W = T.reshape(n * m * m * p, p * n)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                    W = W_T.T
                except:
                    continue

                # Update U
                A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
                T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                    U = U_T.T
                except:
                    continue

                # Update V
                A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
                T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                    V = V_T.T
                except:
                    continue

                if it % 5 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-6:
                        U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=80)
                        T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
                        err_pol = np.linalg.norm(T_calc_pol - T)
                        
                        if err_pol < 1e-9:
                            # Try snapping to half-integers
                            U_int, V_int, W_int = snap_to_half_int(U_pol, V_pol, W_pol)
                            if check_solution(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                            # Try snapping to integers
                            U_int2, V_int2, W_int2 = snap_to_int(U_pol, V_pol, W_pol)
                            if check_solution(U_int2, V_int2, W_int2):
                                return U_int2, V_int2, W_int2
                            # Try rational reconstruction
                            U_rat = rational_reconstruct(U_pol.flatten()).reshape(U_pol.shape)
                            V_rat = rational_reconstruct(V_pol.flatten()).reshape(V_pol.shape)
                            W_rat = rational_reconstruct(W_pol.flatten()).reshape(W_pol.shape)
                            if check_solution(U_rat, V_rat, W_rat):
                                return U_rat, V_rat, W_rat
                        elif err_pol < err:
                            best_err = err_pol
                            best_result = (U_pol.copy(), V_pol.copy(), W_pol.copy())

        if best_result is not None:
            U, V, W = best_result
            U, V, W = polish_solution(U, V, W, max_iters=60)
            T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
            err = np.linalg.norm(T_calc - T)
            if err < 1e-9:
                U_int, V_int, W_int = snap_to_half_int(U, V, W)
                if check_solution(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                U_int2, V_int2, W_int2 = snap_to_int(U, V, W)
                if check_solution(U_int2, V_int2, W_int2):
                    return U_int2, V_int2, W_int2
        
        return None
    
    def try_strassen_like(r):
        """Try to find a Strassen-like decomposition using known patterns."""
        # For 4x5x5, try rank 40-60 with structured approach
        best_result = None
        best_err = float('inf')
        
        for restart in range(8):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart)
            
            # Initialize with a structured pattern based on sub-matrix blocks
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Divide into blocks and assign random linear combinations
            block_size = max(1, r // 20)
            
            for t in range(r):
                # Random linear combination of basis elements
                n_terms = np.random.randint(1, 4)
                for _ in range(n_terms):
                    i = np.random.randint(n)
                    j = np.random.randint(m) 
                    k = np.random.randint(p)
                    coeff = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    
                    U[i * m + j, t] += coeff
                    V[j * p + k, t] += coeff
                    W[k * n + i, t] += coeff
            
            for it in range(50):
                if time.time() > deadline:
                    break
                
                A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
                T_W = T.reshape(n * m * m * p, p * n)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                    W = W_T.T
                except:
                    continue
                
                A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
                T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                    U = U_T.T
                except:
                    continue
                
                A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
                T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                    V = V_T.T
                except:
                    continue
                
                if it % 10 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-8:
                        U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=50)
                        T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
                        err_pol = np.linalg.norm(T_calc_pol - T)
                        if err_pol < 1e-9:
                            U_int, V_int, W_int = snap_to_half_int(U_pol, V_pol, W_pol)
                            if check_solution(U_int, V_int, W_int):
                                return U_int, V_int, W_int
                            U_int2, V_int2, W_int2 = snap_to_int(U_pol, V_pol, W_pol)
                            if check_solution(U_int2, V_int2, W_int2):
                                return U_int2, V_int2, W_int2
        
        return None
    
    # Search ranks from lower bound up to schoolbook
    # Known bounds for 4x5x5: lower bound ~40, upper bound 100 (schoolbook)
    search_start = max(min_rank, 40)
    
    for r in range(search_start, min(schoolbook_rank + 1, 75)):
        if time.time() > deadline - 10:
            break
        
        # More restarts for smaller ranks
        restarts = 10 if r < 55 else 6
        max_iters = 60 if r < 55 else 40
        
        # Try structured initialization first
        result = solve_als(r, max_iters=max_iters, restarts=restarts, use_structured_init=True, seed_offset=0)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        if time.time() > deadline - 5:
            break
        
        # Try random initialization
        result = solve_als(r, max_iters=max_iters, restarts=restarts, use_structured_init=False, seed_offset=100)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        if time.time() > deadline - 3:
            break
        
        # Try Strassen-like approach
        result = try_strassen_like(r)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
