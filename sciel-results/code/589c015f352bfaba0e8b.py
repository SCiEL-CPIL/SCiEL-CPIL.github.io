"""Matrix multiplication 2x4x7 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 138  # seconds, leave 12s margin for 150s limit


def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using integer arithmetic.
    U: (n*m, rank) list of lists of int/float
    V: (m*p, rank) list of lists of int/float
    W: (p*n, rank) list of lists of int/float
    """
    rank = len(U[0]) if U else 0
    # Convert to integers (multiply by 2 if half-integers)
    # First check if all entries are integers or half-integers
    def to_int2(x):
        """Convert to 2x representation (integers become 2*int, half-ints become int)."""
        v = int(round(x * 2))
        if abs(x * 2 - v) > 1e-9:
            return None
        return v
    
    U2 = [[to_int2(x) for x in row] for row in U]
    V2 = [[to_int2(x) for x in row] for row in V]
    W2 = [[to_int2(x) for x in row] for row in W]
    
    for mat in [U2, V2, W2]:
        for row in mat:
            if any(x is None for x in row):
                return False
    
    # Reshape
    U3 = np.array(U2).reshape(n, m, rank)
    V3 = np.array(V2).reshape(m, p, rank)
    W3 = np.array(W2).reshape(p, n, rank)
    
    # For each (i,j,k), check sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
                if s != 4:
                    return False
    return True


def _verify_float(U, V, W, n, m, p, atol=1e-9):
    """Verify using floating point."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
    return np.allclose(recon, 1.0, atol=atol)


def _round_half(U, V, W):
    """Round to nearest half-integers and verify."""
    n, m, p = 2, 4, 7
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _verify_exact(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None


def _build_schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}


def _als_fast(n, m, p, rank, seed, max_iters=400):
    """Fast ALS for tensor decomposition."""
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    # Normalize columns
    for c in range(rank):
        nU = np.linalg.norm(U[:, c])
        nV = np.linalg.norm(V[:, c])
        nW = np.linalg.norm(W[:, c])
        if nU > 1e-15:
            U[:, c] /= nU
        if nV > 1e-15:
            V[:, c] /= nV
        if nW > 1e-15:
            W[:, c] /= nW
    
    prev_res = np.inf
    stall = 0
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 8:
            return None
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # Fix V and W, solve for U[i*m+j,:]
                # For each k: sum_c U_c * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # So U @ M = 1 where M[c,k] = V[j*p+k,c] * W[k*n+i,c]
                Vj = V[j * p:(j + 1) * p, :]  # (p, rank)
                Wi = W[np.arange(p) * n + i, :]  # (p, rank)
                M = Vj * Wi  # (p, rank)
                # Solve U_row @ M = ones(p)
                # U_row is (1, rank), M is (p, rank), want U_row @ M^T = ones(p)
                try:
                    U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                Ui = U[np.arange(n) * m + j, :]  # (n, rank)
                Wk = W[k * n:(k + 1) * n, :]  # (n, rank)
                M = Ui * Wk  # (n, rank)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                Uj = U[np.arange(m), :]  # wait, need U[i*m+j, :] for j in range(m)
                U_slice = U.reshape(n, m, rank)[i, :, :]  # (m, rank)
                Vk = V[np.arange(m) * p + k, :]  # (m, rank)
                M = U_slice * Vk  # (m, rank)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute residual
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            recon += np.einsum('ij,jk,ki->ijk', U3[:, :, c], V3[:, :, c], W3[:, :, c])
        res = np.max(np.abs(recon - 1.0))
        
        if res < prev_res:
            prev_res = res
            stall = 0
        else:
            stall += 1
        
        if res < 1e-12:
            return U, V, W
        
        if stall > 15:
            if it % 10 == 5:
                U += rng.standard_normal((n * m, rank)) * 0.1
                V += rng.standard_normal((m * p, rank)) * 0.1
                W += rng.standard_normal((p * n, rank)) * 0.1
            else:
                U += rng.standard_normal((n * m, rank)) * 0.5
                V += rng.standard_normal((m * p, rank)) * 0.5
                W += rng.standard_normal((p * n, rank)) * 0.5
            stall = 0
    
    return None


def _als_structured(n, m, p, rank, seed, max_iters=300):
    """ALS with structured initialization exploiting m=4=2*2."""
    rng = np.random.default_rng(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        # Structured pattern based on m=2*2
        a1 = rng.choice([-1, 1])
        a2 = rng.choice([-1, 1])
        b1 = rng.choice([-1, 1])
        b2 = rng.choice([-1, 1])
        
        for i in range(n):
            for j in range(m):
                j1, j2 = j // 2, j % 2
                U[i * m + j, c] = a1 ** j1 * a2 ** j2 * rng.choice([-1, 1])
        
        for j in range(m):
            j1, j2 = j // 2, j % 2
            for k in range(p):
                V[j * p + k, c] = b1 ** j1 * b2 ** j2 * rng.choice([-1, 1])
        
        for k in range(p):
            for i in range(n):
                W[k * n + i, c] = rng.choice([-1, 1])
    
    return _als_fast(n, m, p, rank, seed, max_iters)


def _als_sparse(n, m, p, rank, seed, max_iters=300):
    """ALS with sparse initialization."""
    rng = np.random.default_rng(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        nnz_u = max(1, int(n * m * 0.4))
        nnz_v = max(1, int(m * p * 0.4))
        nnz_w = max(1, int(p * n * 0.4))
        
        idx_u = rng.choice(n * m, size=nnz_u, replace=False)
        idx_v = rng.choice(m * p, size=nnz_v, replace=False)
        idx_w = rng.choice(p * n, size=nnz_w, replace=False)
        
        U[idx_u, c] = rng.choice([-1, 1], size=nnz_u)
        V[idx_v, c] = rng.choice([-1, 1], size=nnz_v)
        W[idx_w, c] = rng.choice([-1, 1], size=nnz_w)
    
    return _als_fast(n, m, p, rank, seed, max_iters)


def construct():
    n, m, p = 2, 4, 7
    
    # Schoolbook fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 56
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 28
    # Known bounds suggest rank around 32-40 for this problem
    
    strategies = [_als_fast, _als_structured, _als_sparse]
    
    for rank in range(28, 56):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Allocate more trials to lower ranks
        if rank <= 32:
            num_trials = 8
        elif rank <= 36:
            num_trials = 5
        elif rank <= 42:
            num_trials = 3
        else:
            num_trials = 2
        
        for trial in range(num_trials):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            seed = rank * 100 + trial * 7
            strat = strategies[trial % len(strategies)]
            
            res = strat(n, m, p, rank, seed)
            
            if res is not None:
                U, V, W = res
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Try rounding to half-integers
                rounded = _round_half(U_list, V_list, W_list)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # Try integer rounding
                U_int = [[int(round(x)) for x in row] for row in U_list]
                V_int = [[int(round(x)) for x in row] for row in V_list]
                W_int = [[int(round(x)) for x in row] for row in W_list]
                if _verify_exact(U_int, V_int, W_int, n, m, p):
                    return {"U": U_int, "V": V_int, "W": W_int}
                
                # Try rounding to integers with sign preservation
                U_int2 = [[(1 if x >= 0 else -1) if abs(x) > 0.5 else 0 for x in row] for row in U_list]
                V_int2 = [[(1 if x >= 0 else -1) if abs(x) > 0.5 else 0 for x in row] for row in V_list]
                W_int2 = [[(1 if x >= 0 else -1) if abs(x) > 0.5 else 0 for x in row] for row in W_list]
                if _verify_exact(U_int2, V_int2, W_int2, n, m, p):
                    return {"U": U_int2, "V": V_int2, "W": W_int2}
                
                # Try 0, ±1, ±2 rounding
                def round_to_small(x):
                    if abs(x) < 0.25:
                        return 0
                    elif abs(x) < 0.75:
                        return 1 if x > 0 else -1
                    elif abs(x) < 1.75:
                        return 2 if x > 0 else -2
                    else:
                        return 3 if x > 0 else -3
                U_s = [[round_to_small(x) for x in row] for row in U_list]
                V_s = [[round_to_small(x) for x in row] for row in V_list]
                W_s = [[round_to_small(x) for x in row] for row in W_list]
                if _verify_exact(U_s, V_s, W_s, n, m, p):
                    return {"U": U_s, "V": V_s, "W": W_s}
    
    return best
