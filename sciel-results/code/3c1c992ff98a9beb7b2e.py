"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145  # seconds, leave 5s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    recon = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        recon += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
    return np.allclose(recon, 1.0, atol=1e-9)

def _is_valid_half_int(arr):
    """Check if all entries are half-integers."""
    for row in arr:
        for x in row:
            if abs(x * 2 - round(x * 2)) > 1e-6:
                return False
    return True

def _try_round_half(U, V, W):
    """Try to round to half-integers and verify."""
    n, m, p = DIMENSIONS
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    if _is_valid_half_int(U_r) and _is_valid_half_int(V_r) and _is_valid_half_int(W_r):
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
    return None

def _als_decompose(n, m, p, rank, seed, max_iters=150):
    """ALS with multiple restart strategies."""
    rng = np.random.default_rng(seed)
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_res = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(3):
        # Different initialization strategies
        if restart == 0:
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
        elif restart == 1:
            # Sparse initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            for c in range(rank):
                idx_u = rng.choice(n * m, size=max(1, n*m//4), replace=False)
                idx_v = rng.choice(m * p, size=max(1, m*p//4), replace=False)
                idx_w = rng.choice(p * n, size=max(1, p*n//4), replace=False)
                U[idx_u, c] = rng.standard_normal(len(idx_u))
                V[idx_v, c] = rng.standard_normal(len(idx_v))
                W[idx_w, c] = rng.standard_normal(len(idx_w))
        else:
            # Binary-like initialization
            U = rng.integers(0, 2, (n * m, rank)).astype(np.float64)
            V = rng.integers(0, 2, (m * p, rank)).astype(np.float64)
            W = rng.integers(0, 2, (p * n, rank)).astype(np.float64)
        
        prev_res = np.inf
        stall = 0
        
        for it in range(max_iters):
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                return best_U, best_V, best_W
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((rank, p))
                    for k in range(p):
                        for c in range(rank):
                            A[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, T[i,j,:].T, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((rank, n))
                    for i in range(n):
                        for c in range(rank):
                            A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, T[:,j,k].T, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((rank, m))
                    for j in range(m):
                        for c in range(rank):
                            A[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, T[i,:,k].T, rcond=None)[0]
                    except:
                        pass
            
            # Compute residual
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            recon = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                recon += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            res = np.linalg.norm(recon - 1.0)
            
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res < 1e-10:
                return U, V, W
            
            if res < prev_res * 0.99:
                prev_res = res
                stall = 0
            else:
                stall += 1
            
            if stall > 10:
                # Perturb
                U += rng.standard_normal((n*m, rank)) * 0.1
                V += rng.standard_normal((m*p, rank)) * 0.1
                W += rng.standard_normal((p*n, rank)) * 0.1
                stall = 0
    
    return best_U, best_V, best_W

def _structured_search(n, m, p, rank, seed):
    """Try structured decompositions exploiting symmetries."""
    rng = np.random.default_rng(seed)
    
    # For the all-ones tensor, try decompositions based on partitioning
    # Strategy: use rank-1 terms that are products of indicator functions
    
    # If rank >= n*m*p, schoolbook works. For lower ranks, we need algebraic identities.
    # Try: express 1 as sum of products of small integer vectors
    
    # For [2,4,7], try to find vectors a_c in R^2, b_c in R^4, c_c in R^7
    # such that sum_c a_c[i]*b_c[j]*c_c[k] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the all-ones tensor
    # in the standard 3-way format
    
    # One approach: use the fact that the all-ones tensor has a "flat" structure
    # Try rank = max(n*m, m*p, p*n) = 28 first
    
    if rank < 28:
        return None
    
    # Initialize with a structured pattern
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill with a pattern that might lead to exact solutions
    for c in range(rank):
        # Each column gets a structured pattern
        U[:, c] = rng.choice([-1, 0, 1], size=n*m)
        V[:, c] = rng.choice([-1, 0, 1], size=m*p)
        W[:, c] = rng.choice([-1, 0, 1], size=p*n)
    
    # Refine with ALS
    return _als_decompose(n, m, p, rank, seed, max_iters=100)

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 56
    
    # Schoolbook fallback
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = 28
    # Also try some ranks below 28 in case the bound is not tight
    
    # Time allocation: focus on ranks 28-35 with more trials
    rank_trials = {}
    for r in range(28, 33):
        rank_trials[r] = 20
    for r in range(33, 38):
        rank_trials[r] = 10
    for r in range(38, 45):
        rank_trials[r] = 5
    for r in range(45, 56):
        rank_trials[r] = 2
    
    for rank in range(28, 56):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        num_trials = rank_trials.get(rank, 2)
        
        for trial in range(num_trials):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            seed = rank * 100 + trial * 13
            
            # Alternate between ALS and structured search
            if trial % 2 == 0:
                res = _als_decompose(n, m, p, rank, seed)
            else:
                res = _structured_search(n, m, p, rank, seed)
            
            if res is not None:
                U, V, W = res
                # Try to round to half-integers
                rounded = _try_round_half(U.tolist(), V.tolist(), W.tolist())
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    return {"U": U_r, "V": V_r, "W": W_r}
                
                # Check if already valid (maybe exact floats)
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    # Try to round
                    U_r = [[round(x) for x in row] for row in U.tolist()]
                    V_r = [[round(x) for x in row] for row in V.tolist()]
                    W_r = [[round(x) for x in row] for row in W.tolist()]
                    if _verify(U_r, V_r, W_r, n, m, p):
                        return {"U": U_r, "V": V_r, "W": W_r}
    
    return best_result
# EVOLVE-BLOCK-END
