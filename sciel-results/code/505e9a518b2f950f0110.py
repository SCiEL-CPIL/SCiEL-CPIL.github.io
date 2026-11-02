"""
Tensor decomposition search for 2x3 x 3x3 matrix multiplication.
Strategy: ALS-based randomized search with exact rational reconstruction.
"""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(arr, tol=1e-6):
    """Round array to nearest half-integers if close enough."""
    doubled = 2.0 * arr
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _als_optimize(U, V, W, n=2, m=3, p=3, max_iter=50, tol=1e-10):
    """
    Alternating Least Squares for tensor decomposition.
    Fixed U,V -> solve for W; Fixed V,W -> solve for U; Fixed U,W -> solve for V.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(max_iter):
        # Solve for W: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed (i,j,k), this is linear in W[k*n+i,:]
        # Actually, for each column c, we have:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        
        # Group equations by W row index (k*n+i)
        # For a given (i,j,k), the equation involves W[k*n+i, :]
        # Let's build the linear system properly
        
        # Number of equations: n*m*p
        # Number of unknowns: p*n*rank
        rank = U.shape[1]
        
        # Build A matrix where A[eq_idx, unk_idx] * W_vec[unk_idx] = 1
        # eq_idx corresponds to (i,j,k)
        # unk_idx corresponds to (k*n+i, c)
        
        num_eqs = n * m * p
        num_unk = p * n * rank
        
        A = np.zeros((num_eqs, num_unk))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    for c in range(rank):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            A[eq_idx, w_row * rank + c] = coeff
                    eq_idx += 1
        
        b = np.ones(num_eqs)
        
        # Solve least squares
        try:
            W_vec, residuals, r, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        W_new = W_vec.reshape(p*n, rank)
        
        # Check if improved
        diff_W = np.linalg.norm(W_new - W)
        W = W_new
        
        # Similarly solve for U
        # For fixed (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by U row index (i*m+j)
        A_U = np.zeros((num_eqs, n*m*rank))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = i*m + j
                    for c in range(rank):
                        coeff = V[j*p+k, c] * W[k*n+i, c]
                        if coeff != 0:
                            A_U[eq_idx, u_row * rank + c] = coeff
                    eq_idx += 1
        
        try:
            U_vec, residuals, r, sv = np.linalg.lstsq(A_U, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        U_new = U_vec.reshape(n*m, rank)
        diff_U = np.linalg.norm(U_new - U)
        U = U_new
        
        # Solve for V
        A_V = np.zeros((num_eqs, m*p*rank))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    v_row = j*p + k
                    for c in range(rank):
                        coeff = U[i*m+j, c] * W[k*n+i, c]
                        if coeff != 0:
                            A_V[eq_idx, v_row * rank + c] = coeff
                    eq_idx += 1
        
        try:
            V_vec, residuals, r, sv = np.linalg.lstsq(A_V, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        V_new = V_vec.reshape(m*p, rank)
        diff_V = np.linalg.norm(V_new - V)
        V = V_new
        
        if diff_W < tol and diff_U < tol and diff_V < tol:
            break
    
    return U, V, W

def _try_rank(target_rank, n=2, m=3, p=3, max_trials=100, time_budget=30, seed=42):
    """
    Try to find a rank-`target_rank` decomposition using random initialization + ALS.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    for trial in range(max_trials):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
        
        # Ensure not all zeros in any column
        for c in range(target_rank):
            if np.all(U[:, c] == 0):
                U[np.random.randint(n*m), c] = 1
            if np.all(V[:, c] == 0):
                V[np.random.randint(m*p), c] = 1
            if np.all(W[:, c] == 0):
                W[np.random.randint(p*n), c] = 1
        
        # Run ALS
        U_opt, V_opt, W_opt = _als_optimize(U, V, W, n, m, p, max_iter=100)
        
        # Check if exact
        if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
            # Try to round to half-integers
            U_rounded = _round_to_half_integer(U_opt, tol=1e-6)
            V_rounded = _round_to_half_integer(V_opt, tol=1e-6)
            W_rounded = _round_to_half_integer(W_opt, tol=1e-6)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-10):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
            # Even if not half-integers, if exact, it might be valid if entries are rational
            # But the task requires integer/half-integer entries
            # Let's try harder rounding
            for tol_round in [1e-5, 1e-4, 1e-3]:
                U_r2 = _round_to_half_integer(U_opt, tol=tol_round)
                V_r2 = _round_to_half_integer(V_opt, tol=tol_round)
                W_r2 = _round_to_half_integer(W_opt, tol=tol_round)
                if U_r2 is not None and V_r2 is not None and W_r2 is not None:
                    if _verify(U_r2.tolist(), V_r2.tolist(), W_r2.tolist(), n, m, p, tol=1e-8):
                        return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
        
        # Also check if the ALS solution is exact even without rounding
        # and see if we can find a nearby exact solution
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_opt[i*m+j, c] * V_opt[j*p+k, c] * W_opt[k*n+i, c] 
                             for c in range(target_rank))
                    residual += (val - 1.0) ** 2
        residual = np.sqrt(residual) / (n*m*p)
        
        if residual < 1e-8:
            # Very close to exact, try rounding with looser tolerance
            U_r = _round_to_half_integer(U_opt, tol=1e-4)
            V_r = _round_to_half_integer(V_opt, tol=1e-4)
            W_r = _round_to_half_integer(W_opt, tol=1e-4)
            if U_r is not None and V_r is not None and W_r is not None:
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _structured_patterns(target_rank, n=2, m=3, p=3):
    """Generate structured candidate matrices."""
    candidates = []
    
    # Pattern: use the schoolbook structure but merge columns
    # For rank < 18, we need to find linear dependencies
    
    # Try binary patterns
    for mask_u in [np.ones(n*m), np.zeros(n*m)]:
        mask_u[:n*m//2] = 1
        for mask_v in [np.ones(m*p)]:
            mask_v[:m*p//2] = 1
            U = np.tile(mask_u.reshape(-1, 1), (1, target_rank))
            V = np.tile(mask_v.reshape(-1, 1), (1, target_rank))
            candidates.append((U, V))
    
    # Cyclic patterns
    for shift in range(3):
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        for c in range(target_rank):
            for idx in range(n*m):
                if (idx + c + shift) % 3 == 0:
                    U[idx, c] = 1
            for idx in range(m*p):
                if (idx + c + shift) % 3 == 0:
                    V[idx, c] = 1
        candidates.append((U, V))
    
    return candidates

def construct():
    start_time = time.time()
    total_time_budget = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try ranks from low to high
    for target_rank in range(12, 18):
        elapsed = time.time() - start_time
        remaining = total_time_budget - elapsed
        if remaining < 5:
            break
        
        time_per_rank = min(remaining / (18 - target_rank), 40)
        
        result = _try_rank(target_rank, n, m, p, max_trials=200, time_budget=time_per_rank, seed=target_rank*100+42)
        if result is not None:
            return result
        
        # Also try structured patterns
        for U_pat, V_pat in _structured_patterns(target_rank, n, m, p):
            if time.time() - start_time > total_time_budget - 5:
                break
            
            U_f = U_pat.astype(float)
            V_f = V_pat.astype(float)
            
            # Initialize W randomly
            np.random.seed(target_rank * 1000 + 1)
            W_init = np.random.randn(p*n, target_rank)
            
            U_opt, V_opt, W_opt = _als_optimize(U_f, V_f, W_init, n, m, p, max_iter=50)
            
            if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
                U_r = _round_to_half_integer(U_opt, tol=1e-5)
                V_r = _round_to_half_integer(V_opt, tol=1e-5)
                W_r = _round_to_half_integer(W_opt, tol=1e-5)
                if U_r is not None and V_r is not None and W_r is not None:
                    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-8):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback to schoolbook
    return _build_schoolbook()
