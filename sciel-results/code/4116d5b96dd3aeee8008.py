"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_schoolbook(n, m, p):
    """Build the rank n*m*p schoolbook decomposition as lists of lists of ints."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify exact tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for r in range(rank):
                    s += ui[r] * vj[r] * wk[r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(val, max_denom=2):
    """Snap a float to the nearest multiple of 1/max_denom if close enough."""
    for denom in [1, 2]:
        num = round(val * denom)
        if abs(val - num / denom) < 1e-8:
            return num / denom
    return val

def _try_newton_rank(target_rank, n, m, p, max_time=20.0, seed=None):
    """
    Try to find a rank-`target_rank` decomposition via Newton-Raphson on the
    polynomial system. Returns (U_list, V_list, W_list) or None.
    """
    if seed is not None:
        np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    R = target_rank
    
    # Precompute index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    T = len(triples)
    
    # Random initialization
    U = np.random.randn(nm, R) * 0.5
    V = np.random.randn(mp, R) * 0.5
    W = np.random.randn(pn, R) * 0.5
    
    start_time = time.time()
    
    for iteration in range(30):
        if time.time() - start_time > max_time:
            return None
        
        # Compute residuals: f[t] = 1 - sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        F = np.zeros(T)
        # Also build Jacobian structure: for each term t and each variable, the partial derivative
        # Variables are ordered: U (nm*R), V (mp*R), W (pn*R)
        # Total variables: nm*R + mp*R + pn*R
        
        # For efficiency, compute F and the Jacobian in one pass
        # Jacobian J[t, var] = df_t/dvar
        # For term (i,j,k):
        #   d/dU[a, r] = V[j*p+k, r] * W[k*n+i, r] if a == i*m+j else 0
        #   d/dV[b, r] = U[i*m+j, r] * W[k*n+i, r] if b == j*p+k else 0
        #   d/dW[c, r] = U[i*m+j, r] * V[j*p+k, r] if c == k*n+i else 0
        
        # This is sparse. We'll use a dense Jacobian since dimensions are small.
        n_vars = nm * R + mp * R + pn * R
        J = np.zeros((T, n_vars))
        
        for t, (i, j, k) in enumerate(triples):
            ui = i * m + j
            vj = j * p + k
            wk = k * n + i
            
            # Residual
            s = 0.0
            for r in range(R):
                s += U[ui, r] * V[vj, r] * W[wk, r]
            F[t] = 1.0 - s
            
            # Jacobian row
            for r in range(R):
                uvr = U[ui, r]
                vwr = V[vj, r]
                wwr = W[wk, r]
                
                # dF/dU[ui, r] = -V[vj, r] * W[wk, r]
                J[t, ui * R + r] = -vwr * wwr
                
                # dF/dV[vj, r] = -U[ui, r] * W[wk, r]
                J[t, nm * R + vj * R + r] = -uwr * wwr
                
                # dF/dW[wk, r] = -U[ui, r] * V[vj, r]
                J[t, nm * R + mp * R + wk * R + r] = -uwr * vwr
        
        # Check if we're close enough
        if np.max(np.abs(F)) < 1e-10:
            break
        
        # Solve J * dx = -F using least squares (since J may be under/over-determined)
        try:
            dx, residuals, rank_J, sv = np.linalg.lstsq(J, -F, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Line search: find alpha in (0, 1] that reduces ||F||
        best_alpha = 0.0
        best_norm = np.linalg.norm(F)
        
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            U_trial = U + alpha * dx[:nm * R].reshape(nm, R)
            V_trial = V + alpha * dx[nm * R:nm * R + mp * R].reshape(mp, R)
            W_trial = W + alpha * dx[nm * R + mp * R:].reshape(pn, R)
            
            # Compute new residual norm
            F_new = np.zeros(T)
            for t, (i, j, k) in enumerate(triples):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0.0
                for r in range(R):
                    s += U_trial[ui, r] * V_trial[vj, r] * W_trial[wk, r]
                F_new[t] = 1.0 - s
            
            new_norm = np.linalg.norm(F_new)
            if new_norm < best_norm:
                best_norm = new_norm
                best_alpha = alpha
                U, V, W = U_trial, V_trial, W_trial
        
        if best_alpha == 0.0:
            # No improvement, try smaller step or give up
            return None
        
        if best_norm < 1e-12:
            break
    
    # Final verification
    if np.max(np.abs(F)) < 1e-8:
        # Snap to half-integers
        U_list = [[_snap_to_half_integer(U[i, j]) for j in range(R)] for i in range(nm)]
        V_list = [[_snap_to_half_integer(V[i, j]) for j in range(R)] for i in range(mp)]
        W_list = [[_snap_to_half_integer(W[i, j]) for j in range(R)] for i in range(pn)]
        
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to reduce rank using Newton-Raphson on the polynomial system
    # Lower bound: max(n*p, m*p, n*m) = max(20, 30, 6) = 30
    # Try ranks from 30 upward
    for target_rank in range(30, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        time_left = deadline - time.time() - 5
        if time_left <= 0:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = _try_newton_rank(target_rank, n, m, p, max_time=min(15.0, time_left), seed=seed)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_result = {"U": U_sol, "V": V_sol, "W": W_sol}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break  # Found the lowest rank, stop
    
    return best_result
# EVOLVE-BLOCK-END
