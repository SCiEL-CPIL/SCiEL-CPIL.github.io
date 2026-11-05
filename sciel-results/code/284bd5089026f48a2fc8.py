"""Matrix multiplication 3x3x7 tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def verify(U, V, W, r):
        """Verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        U_ld = np.array(U, dtype=np.float64)
        V_ld = np.array(V, dtype=np.float64)
        W_ld = np.array(W, dtype=np.float64)
        U3 = U_ld.reshape(n, m, r)
        V3 = V_ld.reshape(m, p, r)
        W3 = W_ld.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.all(np.abs(result - 1.0) < 1e-6)

    def compute_error(U, V, W, r):
        """Compute sum of squared errors."""
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return float(np.sum((result - 1.0)**2))

    def als_decompose(r, max_iter=300, seed=42, init_mode='random'):
        """ALS decomposition with multiple restarts and momentum."""
        if init_mode == 'structured':
            U, V, W = _structured_init(r, seed)
        else:
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        prev_err = np.inf
        stall_count = 0
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = V3[j, :, :] * W3[:, i, :]  # (p, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = sol
                    except:
                        pass
            U = U_new
            U3 = U.reshape(n, m, r)
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    M = U3[:, j, :] * W3[k, :, :]  # (n, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = sol
                    except:
                        pass
            V = V_new
            V3 = V.reshape(m, p, r)
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    M = U3[i, :, :] * V3[:, k, :]  # (m, r)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = sol
                    except:
                        pass
            W = W_new
            W3 = W.reshape(p, n, r)
            
            err = compute_error(U, V, W, r)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < prev_err:
                prev_err = err
                stall_count = 0
            else:
                stall_count += 1
            if err < 1e-12:
                break
            if stall_count > 30:
                break
        
        return best_U, best_V, best_W, best_err

    def _structured_init(r, seed):
        """Create structured initialization."""
        rng = np.random.RandomState(seed)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Distribute products across rank terms
        products_per_term = (n * m * p + r - 1) // r
        idx = 0
        for rr in range(r):
            count = min(products_per_term, n * m * p - idx)
            for _ in range(count):
                if idx >= n * m * p:
                    break
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                U[i * m + j, rr] = 1.0
                V[j * p + k, rr] = 1.0
                W[k * n + i, rr] = 1.0
                idx += 1
        
        # Add perturbation
        U += rng.randn(n * m, r) * 0.5
        V += rng.randn(m * p, r) * 0.5
        W += rng.randn(p * n, r) * 0.5
        
        return U, V, W

    def polish(U, V, W, r, max_iter=100):
        """High-precision polish."""
        U_ld = U.astype(np.float64)
        V_ld = V.astype(np.float64)
        W_ld = W.astype(np.float64)
        U3 = U_ld.reshape(n, m, r)
        V3 = V_ld.reshape(m, p, r)
        W3 = W_ld.reshape(p, n, r)
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = V3[j, :, :] * W3[:, i, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U3[i, j, :] = sol
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    M = U3[:, j, :] * W3[k, :, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V3[j, k, :] = sol
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    M = U3[i, :, :] * V3[:, k, :]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W3[k, i, :] = sol
                    except:
                        pass
            
            result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = float(np.sum((result - 1.0)**2))
            if err < 1e-15:
                break
        
        U_ld = U3.reshape(n * m, r)
        V_ld = V3.reshape(m * p, r)
        W_ld = W3.reshape(p * n, r)
        return U_ld, V_ld, W_ld

    def snap_to_rationals(M, max_denom=8):
        """Snap values to rational numbers with small denominators."""
        Frac_M = []
        for row in M:
            Frac_row = []
            for val in row:
                if abs(val) < 1e-10:
                    Frac_row.append(Fraction(0, 1))
                    continue
                best_frac = Fraction(str(val)).limit_denominator(max_denom)
                Frac_row.append(best_frac)
            Frac_M.append(Frac_row)
        return Frac_M

    def verify_fraction(U_frac, V_frac, W_frac, r):
        """Verify exact decomposition using fraction arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0, 1)
                    for rr in range(r):
                        total += U_frac[i*m+j][rr] * V_frac[j*p+k][rr] * W_frac[k*n+i][rr]
                    if total != Fraction(1, 1):
                        return False
        return True

    def to_float_matrix(Frac_M):
        """Convert fraction matrix to float matrix."""
        return [[float(x) for x in row] for row in Frac_M]

    # Search strategy
    best_result = None
    best_rank = 63
    
    # Try ranks from low to high
    # For 3x3*3x7, known lower bound is around 27, schoolbook is 63
    ranks_to_try = list(range(25, 64))
    
    # Time budgeting
    available_time = deadline - start_time - 15
    time_per_rank = available_time / len(ranks_to_try)
    
    for idx, r in enumerate(ranks_to_try):
        if time.time() > deadline - 10:
            break
        
        rank_deadline = time.time() + time_per_rank
        found = False
        
        # Multiple restarts with different seeds and initialization strategies
        n_restarts = max(1, int((rank_deadline - time.time()) / 3))
        n_restarts = min(n_restarts, 8)  # Cap restarts
        
        for restart in range(n_restarts):
            if time.time() > rank_deadline - 2:
                break
            
            seed = 42 + r * 100 + restart * 7
            init_mode = 'structured' if restart % 2 == 0 else 'random'
            
            U, V, W, err = als_decompose(r, max_iter=150, seed=seed, init_mode=init_mode)
            
            if err < 1e-8:
                # Polish
                U_ld, V_ld, W_ld = polish(U, V, W, r)
                
                # Try to snap to rationals
                for max_denom in [2, 4, 8, 16]:
                    U_frac = snap_to_rationals(U_ld, max_denom)
                    V_frac = snap_to_rationals(V_ld, max_denom)
                    W_frac = snap_to_rationals(W_ld, max_denom)
                    
                    if verify_fraction(U_frac, V_frac, W_frac, r):
                        best_rank = r
                        best_result = {
                            "U": to_float_matrix(U_frac),
                            "V": to_float_matrix(V_frac),
                            "W": to_float_matrix(W_frac)
                        }
                        found = True
                        break
                
                if found:
                    break
                
                # Try direct verification with floats
                if verify(U_ld, V_ld, W_ld, r):
                    best_rank = r
                    best_result = {
                        "U": U_ld.tolist(),
                        "V": V_ld.tolist(),
                        "W": W_ld.tolist()
                    }
                    found = True
                    break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
