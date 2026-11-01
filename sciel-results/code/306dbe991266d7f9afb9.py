import time
import numpy as np
from scipy.optimize import least_squares

DEADLINE = 120.0
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE - 15

def _construct_trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _construct_block_50(n, m, p):
    """
    Reconstructs the rank-50 solution found in the previous iteration to serve
    as a high-quality starting point for optimization.
    """
    # Build 2x2x2 Strassen
    n2, m2, p2 = 2, 2, 2
    R2 = 7
    U222 = np.zeros((n2*m2, R2))
    V222 = np.zeros((m2*p2, R2))
    W222 = np.zeros((p2*n2, R2))
    
    # P1
    U222[0, 0] = 1; U222[3, 0] = 1
    V222[0, 0] = 1; V222[3, 0] = 1
    W222[0, 0] = 1; W222[3, 0] = 1
    # P2
    U222[2, 1] = 1; U222[3, 1] = 1
    V222[0, 1] = 1
    W222[2, 1] = 1; W222[3, 1] = -1
    # P3
    U222[0, 2] = 1
    V222[1, 2] = 1; V222[3, 2] = -1
    W222[1, 2] = 1; W222[3, 2] = 1
    # P4
    U222[3, 3] = 1
    V222[2, 3] = 1; V222[0, 3] = -1
    W222[0, 3] = 1; W222[2, 3] = 1
    # P5
    U222[0, 4] = 1; U222[1, 4] = 1
    V222[3, 4] = 1
    W222[0, 4] = -1; W222[3, 4] = 1
    # P6
    U222[2, 5] = 1; U222[0, 5] = -1
    V222[0, 5] = 1; V222[1, 5] = 1
    W222[3, 5] = 1
    # P7
    U222[1, 6] = 1; U222[3, 6] = -1
    V222[2, 6] = 1; V222[3, 6] = 1
    W222[0, 6] = 1; W222[2, 6] = -1

    def embed(U_src, V_src, W_src, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
        U = np.zeros((n_tgt*m_tgt, U_src.shape[1]))
        V = np.zeros((m_tgt*p_tgt, V_src.shape[1]))
        W = np.zeros((p_tgt*n_tgt, W_src.shape[1]))
        
        for i in range(n_src):
            for j in range(m_src):
                si = i*m_src + j
                ti = n_map[i]*m_tgt + m_map[j]
                U[ti, :] = U_src[si, :]
        for j in range(m_src):
            for k in range(p_src):
                sj = j*p_src + k
                tj = m_map[j]*p_tgt + p_map[k]
                V[tj, :] = V_src[sj, :]
        for k in range(p_src):
            for i in range(n_src):
                sk = k*n_src + i
                tk = p_map[k]*n_tgt + n_map[i]
                W[tk, :] = W_src[sk, :]
        return U, V, W

    # 2x2x1 Schoolbook
    U221 = np.zeros((2*2, 2*2*1))
    V221 = np.zeros((2*1, 2*2*1))
    W221 = np.zeros((1*2, 2*2*1))
    r = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U221[i*2+j, r] = 1
                V221[j*1+k, r] = 1
                W221[k*2+i, r] = 1
                r += 1

    # 2x2x3 (Strassen on p0,1 + Schoolbook on p2)
    U223a, V223a, W223a = embed(U222, V222, W222, 2, 2, 2, 2, 2, 3, [0,1], [0,1], [0,1])
    U223b, V223b, W223b = embed(U221, V221, W221, 2, 2, 1, 2, 2, 3, [0,1], [0,1], [2])
    U223 = np.hstack([U223a, U223b])
    V223 = np.hstack([V223a, V223b])
    W223 = np.hstack([W223a, W223b])
    
    # 2x4x3 (Two 2x2x3 blocks)
    U243a, V243a, W243a = embed(U223, V223, W223, 2, 2, 3, 2, 4, 3, [0,1], [0,1], [0,1,2])
    U243b, V243b, W243b = embed(U223, V223, W223, 2, 2, 3, 2, 4, 3, [0,1], [2,3], [0,1,2])
    U243 = np.hstack([U243a, U243b])
    V243 = np.hstack([V243a, V243b])
    W243 = np.hstack([W243a, W243b])
    
    # 2x4x4 (Four 2x2x2 blocks)
    U244a, V244a, W244a = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [0,1], [0,1])
    U244b, V244b, W244b = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [2,3], [0,1])
    U244c, V244c, W244c = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [0,1], [2,3])
    U244d, V244d, W244d = embed(U222, V222, W222, 2, 2, 2, 2, 4, 4, [0,1], [2,3], [2,3])
    U244 = np.hstack([U244a, U244b, U244c, U244d])
    V244 = np.hstack([V244a, V244b, V244c, V244d])
    W244 = np.hstack([W244a, W244b, W244c, W244d])
    
    # 2x4x7 (2x4x4 + 2x4x3)
    U247a, V247a, W247a = embed(U244, V244, W244, 2, 4, 4, 2, 4, 7, [0,1], [0,1,2,3], [0,1,2,3])
    U247b, V247b, W247b = embed(U243, V243, W243, 2, 4, 3, 2, 4, 7, [0,1], [0,1,2,3], [4,5,6])
    
    U = np.hstack([U247a, U247b])
    V = np.hstack([V247a, V247b])
    W = np.hstack([W247a, W247b])
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify the tensor identity:
    For all i, k:
    sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 (if j matches in a specific way? No)
    
    Correct check:
    The tensor T is defined such that C[i,k] = sum_{j,r} A[i,j] B[j,k] (coefficient).
    The coefficient of A[i,j] B[j',k] in the expansion is:
    sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r]
    
    We want C[i,k] = sum_j A[i,j] B[j,k].
    So the coefficient must be 1 if j == j', and 0 if j != j'.
    i.e., sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j, j'}
    """
    # Vectorized check
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # We need to check for all i, k, j, j':
    # M_{j, j'} = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] == I_m
    
    # Let's reshape to make this easier.
    # U_reshaped: (n, m, R)
    U_r = U.reshape(n, m, -1)
    # V_reshaped: (m, p, R) -> transpose to (p, m, R) to align k first?
    # V index is j*p + k. So V[j, k].
    V_r = V.reshape(m, p, -1)
    V_r = V_r.transpose(1, 0, 2) # (p, m, R)
    
    # W index is k*n + i. So W[k, i].
    W_r = W.reshape(p, n, -1)
    
    # We want to compute M[i, k, j, j'] = sum_r U_r[i, j, r] * V_r[k, j', r] * W_r[k, i, r]
    
    # This is a tensor contraction.
    # Let's do it in a loop for clarity and memory safety, as n, m, p are small.
    I_m = np.eye(m)
    
    for i in range(n):
        for k in range(p):
            # U_i: (m, R)
            U_i = U_r[i]
            # V_k: (m, R) -- Note: V_r is (p, m, R), so V_r[k] is (m, R)
            V_k = V_r[k]
            # W_ik: (R,)
            W_ik = W_r[k, i]
            
            # M[j, j'] = sum_r U_i[j, r] * V_k[j', r] * W_ik[r]
            # M = (U_i * W_ik) @ V_k.T
            M = (U_i * W_ik) @ V_k.T
            
            if not np.allclose(M, I_m, atol=1e-6):
                return False
    return True

def _optimize_rank(U0, V0, W0, n, m, p, target_rank, time_limit=10):
    """
    Attempts to find a decomposition of rank target_rank starting from U0, V0, W0.
    Uses least_squares to minimize the residual of the tensor identity.
    """
    R = target_rank
    # Flatten U, V, W
    U_flat = U0[:, :R].flatten()
    V_flat = V0[:, :R].flatten()
    W_flat = W0[:, :R].flatten()
    
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def residuals(x):
        U = x[:U_flat.size].reshape(n*m, R)
        V = x[U_flat.size:U_flat.size+V_flat.size].reshape(m*p, R)
        W = x[U_flat.size+V_flat.size:].reshape(p*n, R)
        
        res = []
        for i in range(n):
            for k in range(p):
                U_i = U[i*m:(i+1)*m, :]
                V_k = V[np.arange(m)*p + k, :]
                W_ik = W[k*n+i, :]
                
                M = (U_i * W_ik) @ V_k.T
                # Residual is M - I
                res.extend((M - np.eye(m)).flatten())
                
        return np.array(res)
        
    try:
        # Use 'trf' method for better handling of bounds if needed, but here we are unbounded
        # However, the problem is non-convex.
        result = least_squares(residuals, x0, method='lm', max_nfev=100)
        
        if result.success or np.linalg.norm(result.fun) < 1e-5:
            U_opt = result.x[:U_flat.size].reshape(n*m, R)
            V_opt = result.x[U_flat.size:U_flat.size+V_flat.size].reshape(m*p, R)
            W_opt = result.x[U_flat.size+V_flat.size:].reshape(p*n, R)
            
            if _verify(U_opt, V_opt, W_opt, n, m, p):
                return U_opt, V_opt, W_opt
    except Exception:
        pass
        
    return None

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    
    # 1. Get the rank-50 baseline
    try:
        U50, V50, W50 = _construct_block_50(n, m, p)
        if _verify(U50, V50, W50, n, m, p):
            current_best = (50, U50, V50, W50)
        else:
            # Fallback to trivial if block construction fails
            U_t, V_t, W_t = _construct_trivial(n, m, p)
            current_best = (56, U_t, V_t, W_t)
    except Exception:
        U_t, V_t, W_t = _construct_trivial(n, m, p)
        current_best = (56, U_t, V_t, W_t)
        
    best_rank = current_best[0]
    best_U, best_V, best_W = current_best[1], current_best[2], current_best[3]
    
    # 2. Attempt optimization for lower ranks
    # We know rank 50 is valid. Can we do 49, 48, ...?
    # Theoretical lower bound is higher, but let's try.
    # Optimization is expensive, so we limit attempts.
    
    time_budget_per_attempt = 5.0
    attempts = 0
    max_attempts = 10 # 10 * 5s = 50s. Leaves room for verification and overhead.
    
    for target_rank in range(best_rank - 1, best_rank - max_attempts - 1, -1):
        if _check_deadline():
            break
        if attempts >= max_attempts:
            break
            
        # Start from the current best solution, truncated/padded to target_rank
        # If target_rank < current rank, we just take the first target_rank columns.
        # This is a heuristic initialization.
        
        U_init = best_U[:, :target_rank]
        V_init = best_V[:, :target_rank]
        W_init = best_W[:, :target_rank]
        
        # To make the optimization more likely to succeed, we can add small random noise
        # to escape local minima, but for rank reduction, truncation is the standard first try.
        
        result = _optimize_rank(U_init, V_init, W_init, n, m, p, target_rank, time_limit=time_budget_per_attempt)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            # If we found a lower rank, we can try to go even lower,
            # but we update the base for the next attempt.
            
        attempts += 1

    # Final verification
    if _verify(best_U, best_V, best_W, n, m, p):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback
        U_t, V_t, W_t = _construct_trivial(n, m, p)
        return {
            "U": U_t.tolist(),
            "V": V_t.tolist(),
            "W": W_t.tolist()
        }

if __name__ == "__main__":
    res = construct()
    print("Constructed.")
