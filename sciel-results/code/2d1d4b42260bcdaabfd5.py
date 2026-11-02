import numpy as np
import time
import random

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=tol, rtol=0):
        return False
    return True

def _round_to_half_integers(mat):
    return np.round(mat * 2) / 2.0

def _try_rank_r(n, m, p, r, T, deadline):
    start_time = time.time()
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # ALS iterations
        converged = False
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Update U
            VW = np.einsum('bs,cs->bcs', V, W)
            Num_U = np.einsum('abc,bcs->as', T, VW)
            Denom_U = np.einsum('bcs,bcs->s', VW, VW)
            Denom_U[Denom_U == 0] = 1e-10
            U = Num_U / Denom_U
            
            # Update V
            UW = np.einsum('as,cs->acs', U, W)
            Num_V = np.einsum('abc,acs->bs', T, UW)
            Denom_V = np.einsum('acs,acs->s', UW, UW)
            Denom_V[Denom_V == 0] = 1e-10
            V = Num_V / Denom_V
            
            # Update W
            UV = np.einsum('as,bs->abs', U, V)
            Num_W = np.einsum('abc,abs->cs', T, UV)
            Denom_W = np.einsum('abs,abs->s', UV, UV)
            Denom_W[Denom_W == 0] = 1e-10
            W = Num_W / Denom_W
            
            # Check residual
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_recon)
            if residual < 1e-8:
                converged = True
                break
        
        if not converged:
            continue
            
        # Try to snap to half-integers
        U_rounded = _round_to_half_integers(U)
        V_rounded = _round_to_half_integers(V)
        W_rounded = _round_to_half_integers(W)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-6):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
            
        # Try to snap to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-6):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        # Repair step: 
        # If the solution is close to integers, maybe we can fix some and solve for others?
        # This is complex. Let's try a few more random restarts.
        
    return None

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = 2, 3, 7
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    max_time = 140.0 # 10s margin
    
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_solution = None
    best_rank = n * m * p
    
    # Try ranks from 41 down to 30
    for r in range(best_rank - 1, 29, -1):
        if time.time() > max_time:
            break
            
        # Allocate time based on remaining time
        remaining = max_time - (time.time() - start_time)
        if remaining < 5:
            break
            
        # Give more time to higher ranks (closer to 42) as they are more likely?
        # Or just distribute evenly.
        time_slice = min(remaining / (best_rank - 1 - r + 1), 10.0)
        
        sol = _try_rank_r(n, m, p, r, T, time.time() + time_slice)
        
        if sol is not None:
            U, V, W = sol
            best_solution = (U, V, W)
            best_rank = r
            # Continue to try for lower ranks
            
    if best_solution:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
