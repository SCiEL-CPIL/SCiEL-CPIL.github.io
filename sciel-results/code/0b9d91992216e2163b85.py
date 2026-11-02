import time
import numpy as np
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 140.0

def _check_time():
    return time.time() > _DEADLINE - 15

def _validate_exact(U, V, W, n, m, p):
    """Check if factor matrices exactly decompose the matmul tensor using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def _recover_rationals(U, V, W, max_den=100):
    """Try to recover exact rationals from numerical solution."""
    def to_frac(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        for den in range(2, max_den + 1):
            num = x * den
            if abs(num - round(num)) < 1e-10:
                return Fraction(int(round(num)), den)
        return Fraction(x).limit_denominator(max_den)
    
    U_fr = [[to_frac(float(x)) for x in row] for row in U]
    V_fr = [[to_frac(float(x)) for x in row] for row in V]
    W_fr = [[to_frac(float(x)) for x in row] for row in W]
    return U_fr, V_fr, W_fr

def _als_step(U, V, W, n, m, p, r):
    """One ALS iteration."""
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            # System: A @ U[u_idx] = ones(p)
            # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            A = np.zeros((p, r))
            for k in range(p):
                A[k] = V[j*p+k] * W[k*n+i]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[u_idx] = sol
            except:
                pass
    
    # Update V
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            # System: A @ V[v_idx] = ones(n)
            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            A = np.zeros((n, r))
            for i in range(n):
                A[i] = U_new[i*m+j] * W[k*n+i]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new[v_idx] = sol
            except:
                pass
    
    # Update W
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            # System: A @ W[w_idx] = ones(m)
            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            A = np.zeros((m, r))
            for j in range(m):
                A[j] = U_new[i*m+j] * V_new[j*p+k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W_new[w_idx] = sol
            except:
                pass
    
    return U_new, V_new, W_new

def _compute_error(U, V, W, n, m, p):
    """Compute max error."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _search_2x3x4(target_r, max_trials=20, max_iters=500):
    """Search for rank target_r decomposition of 2x3x4 matmul tensor."""
    n, m, p = 2, 3, 4
    best_err = float('inf')
    best_sol = None
    
    for trial in range(max_trials):
        if _check_time():
            break
        
        rng = np.random.default_rng(42 + trial * 137 + target_r)
        
        # Initialize
        U = rng.standard_normal((n*m, target_r)) * 0.5
        V = rng.standard_normal((m*p, target_r)) * 0.5
        W = rng.standard_normal((p*n, target_r)) * 0.5
        
        # Normalize
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-10
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-10
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-10
        
        prev_err = float('inf')
        for it in range(max_iters):
            if _check_time():
                break
            
            U, V, W = _als_step(U, V, W, n, m, p, target_r)
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_sol = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-12:
                break
            
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
    
    if best_err < 1e-9:
        return best_sol, best_err
    return None, best_err

def _compose_2x3x8(U14, V14, W14):
    """Compose 2x3x4 solution to 2x3x8."""
    n, m, p4 = 2, 3, 4
    p8 = 8
    r = len(U14[0])
    
    U28 = []
    for row in U14:
        U28.append(row + row)
    
    V28 = []
    for j in range(m):
        for k in range(p8):
            k_mod = k % p4
            v_idx = j * p4 + k_mod
            if k < p4:
                V28.append(V14[v_idx] + [0]*r)
            else:
                V28.append([0]*r + V14[v_idx])
    
    W28 = []
    for k in range(p8):
        for i in range(n):
            k_mod = k % p4
            w_idx = k_mod * n + i
            if k < p4:
                W28.append(W14[w_idx] + [0]*r)
            else:
                W28.append([0]*r + W14[w_idx])
    
    return U28, V28, W28

def construct():
    n, m, p = 2, 3, 8
    schoolbook_rank = n * m * p
    
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Search for 2x3x4 rank 14
    for r4 in [14, 15, 16, 17, 18]:
        if _check_time():
            break
        if 2 * r4 >= best_rank:
            continue
        
        sol, err = _search_2x3x4(r4, max_trials=30, max_iters=800)
        if sol is not None:
            U14, V14, W14 = sol
            U28, V28, W28 = _compose_2x3x8(U14, V14, W14)
            
            U_fr, V_fr, W_fr = _recover_rationals(U28, V28, W28)
            if _validate_exact(U_fr, V_fr, W_fr, n, m, p):
                if 2 * r4 < best_rank:
                    best_rank = 2 * r4
                    best_result = {"U": U_fr, "V": V_fr, "W": W_fr}
                break
            else:
                # Try higher precision recovery
                U_fr2, V_fr2, W_fr2 = _recover_rationals(U28, V28, W28, max_den=500)
                if _validate_exact(U_fr2, V_fr2, W_fr2, n, m, p):
                    if 2 * r4 < best_rank:
                        best_rank = 2 * r4
                        best_result = {"U": U_fr2, "V": V_fr2, "W": W_fr2}
                    break
    
    return best_result
