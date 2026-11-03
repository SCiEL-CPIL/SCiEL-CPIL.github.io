import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
            return False, float('inf')
            
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol, max_err
    except Exception:
        return False, float('inf')

def _verify_exact(U, V, W, n, m, p):
    """Verify using exact Fraction arithmetic."""
    try:
        U_frac = [[Fraction(x) for x in row] for row in U]
        V_frac = [[Fraction(x) for x in row] for row in V]
        W_frac = [[Fraction(x) for x in row] for row in W]
        
        rank = len(U[0])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for c in range(rank):
                        val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 1:
                        return False
        return True
    except Exception:
        return False

def _try_polish(U, V, W, n, m, p, max_denom=20):
    """Try to snap to small rationals and verify."""
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for denom in [1, 2, 3, 4, 5, 10, 20]:
        U_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in U_list]
        V_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in V_list]
        W_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in W_list]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    return None

def _linear_solve_search(n, m, p, r, max_time=5, seed=42):
    """Search by fixing U, V and solving for W."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    values = np.array([-1, -0.5, 0, 0.5, 1])
    
    # We need to solve for W.
    # The equation is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let x_c = W[k*n+i, c].
    # For each (i, j, k), we have a linear equation in x_c.
    # There are n*m*p equations and r*n*p unknowns? No.
    # W has shape (p*n, r). So there are p*n*r unknowns.
    # Equations: n*m*p.
    # If r is small, the system is underdetermined.
    # We can set up the matrix A such that A * W_flat = 1.
    
    num_eqs = n * m * p
    num_unk = p * n * r
    
    # A is num_eqs x num_unk
    # A[(i, j, k), (k', i', c)] = U[i*m+j, c] * V[j*p+k, c] if k'==k and i'==i else 0
    # Actually, W index is (k, i) in (p, n) grid.
    # W[k*n+i, c].
    # So unknown index is (k*n+i)*r + c.
    
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = i * m * p + j * p + k
                for c in range(r):
                    u_val = 1 # Placeholder, we will fill U, V randomly
                    # We need to iterate over random U, V
                    # This structure is hard to vectorize with random U, V
                    # Let's just generate U, V, then build A.
                    
    # Better approach:
    # 1. Generate random U, V.
    # 2. Build A.
    # 3. Solve A W = b.
    # 4. Check W.
    
    for iteration in range(500):
        if time.time() - start_time > max_time:
            break
            
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        # Fill U, V with random values
        # Ensure some non-zeros
        for c in range(r):
            idx_U = rng.choice(n * m, size=max(1, n*m//3), replace=False)
            idx_V = rng.choice(m * p, size=max(1, m*p//3), replace=False)
            U[idx_U, c] = values[rng.randint(0, len(values), size=len(idx_U))]
            V[idx_V, c] = values[rng.randint(0, len(values), size=len(idx_V))]
            
        # Build A
        A = np.zeros((num_eqs, num_unk))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for c in range(r):
                        u_val = U[i*m+j, c]
                        v_val = V[j*p+k, c]
                        if u_val != 0 and v_val != 0:
                            # W index for (k, i, c)
                            w_idx = (k * n + i) * r + c
                            A[eq_idx, w_idx] = u_val * v_val
                            
        # Solve A W = b
        # Use least squares
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
            
        # Reshape W
        W = W_flat.reshape(p * n, r)
        
        # Check if W is close to integer/half-integer
        # And if the solution is exact
        ok, err = _verify_fast(U, V, W, n, m, p, tol=1e-8)
        if ok:
            polished = _try_polish(U, V, W, n, m, p)
            if polished:
                return polished[0], polished[1], polished[2], r
                
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try linear solve search for ranks 44 down to 30
    # Focus on 44 first as it is a known improvement
    for r in range(44, 29, -1):
        if time.time() > deadline - 10:
            break
            
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            res = _linear_solve_search(n, m, p, r, max_time=2, seed=seed)
            if res is not None:
                U, V, W, rank = res
                if rank < best_rank:
                    best_rank = rank
                    result = {"U": U, "V": V, "W": W}
                    # If we find a good one, we can try to go lower
                    # But we break to return? No, we want minimal.
                    # Continue searching for lower ranks.
                    break
    
    return result
