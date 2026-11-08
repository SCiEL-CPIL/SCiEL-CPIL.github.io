"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _validate(U, V, W, n, m, p):
    """Check exact decomposition.
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    W index: W[k*n + i, c].
    Condition: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Vectorized check
    # U3: (n, m, R)
    # V3: (m, p, R)
    # W3: (p, n, R)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # We need to check sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c] == 1
    # Let's construct the full tensor product sum efficiently or just loop since n,m,p are small
    # n=2, m=3, p=8 -> 48 checks. Rank is ~48. 48*48 ops is trivial.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
                u_vec = U[i*m+j, :]
                v_vec = V[j*p+k, :]
                w_vec = W[k*n+i, :]
                s = np.dot(u_vec, v_vec * w_vec)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(R, n, m, p, seed=42, time_limit=10.0):
    """Try to find a rank-R decomposition using scipy.optimize.least_squares."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    total_vars = R * (n_m + m_p + p_n)
    
    def residuals(x):
        # x is flat: [U (n_m*R), V (m_p*R), W (p_n*R)]
        # Note: reshape order matters.
        # If x[:R*n_m] is reshaped to (n, m, R), then index (i, j, c) is at ((i*m)+j)*R + c.
        # This matches the flat index in the first block if we treat the block as U_flat.
        # Let's verify:
        # U_flat = x[:R*n_m]
        # U3 = U_flat.reshape(n, m, R)
        # U3[i, j, c] is at index ((i*m)+j)*R + c in U_flat.
        # So x[idx] corresponds to U3[i,j,c].
        
        U_flat = x[:R*n_m]
        V_flat = x[R*n_m:R*n_m + R*m_p]
        W_flat = x[R*n_m + R*m_p:]
        
        U3 = U_flat.reshape(n, m, R)
        V3 = V_flat.reshape(m, p, R)
        W3 = W_flat.reshape(p, n, R)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                    # Vectorized dot product
                    s = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                    res[idx] = s - 1.0
                    idx += 1
        return res

    best_solution = None
    best_err = float('inf')
    
    # Multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize x
        x0 = np.zeros(total_vars)
        
        # Strategy: Sparse initialization with random signs.
        # Each rank index c gets a random "center" (i_c, j_c, k_c)
        # and potentially some "spread" to other indices.
        
        # Simple effective init:
        # For each c, pick i_c, j_c, k_c.
        # Set U[i_c, j_c, c] = 1
        # Set V[j_c, k_c, c] = 1
        # Set W[k_c, i_c, c] = 1
        # This gives sum = R for each (i,j,k) if they match, 0 otherwise?
        # No, if (i,j,k) == (i_c, j_c, k_c), sum is 1.
        # If we have R such terms, and they are distinct, then for the specific (i,j,k) of term c,
        # the sum over all terms is 1 (from term c) + 0 (from others) = 1.
        # So this is a VALID decomposition of rank R if we just use R distinct triples?
        # Wait. If we pick R distinct triples (i_c, j_c, k_c), and set U,V,W to 1 at those spots,
        # then for any (i,j,k), the sum is 1 if (i,j,k) is one of the chosen triples, and 0 otherwise.
        # We need the sum to be 1 for ALL (i,j,k).
        # So this only works if R >= n*m*p and we pick ALL triples.
        # If R < n*m*p, this simple init fails for unchosen triples.
        
        # However, the optimizer can fix this by spreading the weights.
        # If we start with a "sparse" init that is close to a valid decomposition for SOME entries,
        # the optimizer can move towards satisfying the others.
        
        # Better init:
        # Randomly assign each rank index c a "cluster" of triples it tries to cover.
        # Or just random Gaussian init?
        # Least squares often works well with random Gaussian init if the landscape is smooth enough.
        # But this is a cubic problem, so landscape is rugged.
        
        # Let's try a mix:
        # 1. Start with schoolbook-like sparse init (random R triples)
        # 2. Add small noise to encourage exploration
        
        for c in range(R):
            i_c = rng.randint(n)
            j_c = rng.randint(m)
            k_c = rng.randint(p)
            
            # Primary term
            idx_u = ((i_c * m) + j_c) * R + c
            idx_v = R*n_m + ((j_c * p) + k_c) * R + c
            idx_w = R*n_m + R*m_p + ((k_c * n) + i_c) * R + c
            
            # Scale by 1 to make the "active" term contribute 1.
            # If we have R terms, and they are random, they might overlap or miss.
            # Let's just set them to 1.
            x0[idx_u] = 1.0
            x0[idx_v] = 1.0
            x0[idx_w] = 1.0
            
            # Add small random perturbations to other entries in this column c
            # This helps escape local minima
            # U part
            for idx in range(R*n_m):
                if idx != idx_u and rng.random() < 0.1:
                    x0[idx] = rng.uniform(-0.1, 0.1)
            # V part
            for idx in range(R*n_m, R*n_m + R*m_p):
                if idx != idx_v and rng.random() < 0.1:
                    x0[idx] = rng.uniform(-0.1, 0.1)
            # W part
            for idx in range(R*n_m + R*m_p, total_vars):
                if idx != idx_w and rng.random() < 0.1:
                    x0[idx] = rng.uniform(-0.1, 0.1)
        
        # Run optimization
        try:
            # Use 'lm' method if possible? No, 'trf' is more robust for large problems.
            # n*m*p = 48 residuals. total_vars = R * (6 + 24 + 16) = R * 46.
            # If R=48, vars = 2208. Residuals = 48. Underdetermined.
            # 'lm' requires M >= N. Here M < N. So 'trf' is required.
            
            sol = least_squares(residuals, x0, method='trf', max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            err = sol.cost
            if err < best_err:
                best_err = err
                best_solution = sol.x
        except Exception:
            continue

    if best_solution is not None and best_err < 1e-10:
        # Extract U, V, W
        U_flat = best_solution[:R*n_m].reshape(n, m, R)
        V_flat = best_solution[R*n_m:R*n_m + R*m_p].reshape(m, p, R)
        W_flat = best_solution[R*n_m + R*m_p:].reshape(p, n, R)
        
        # Convert back to flat (n*m, R) format
        U_out = U_flat.reshape(n*m, R)
        V_out = V_flat.reshape(m*p, R)
        W_out = W_flat.reshape(p*n, R)
        
        # Try rounding to integers
        U_int = np.round(U_out).astype(int)
        V_int = np.round(V_out).astype(int)
        W_int = np.round(W_out).astype(int)
        
        if _validate(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        # Try rounding to half-integers
        U_half = np.round(U_out * 2) / 2.0
        V_half = np.round(V_out * 2) / 2.0
        W_half = np.round(W_out * 2) / 2.0
        
        if _validate(U_half.tolist(), V_half.tolist(), W_half.tolist(), n, m, p):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
            
        # Try rounding to 1/4 integers? The problem says integer/half-integer.
        # So stick to .0 or .5
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to schoolbook
    schoolbook_rank = n * m * p
    
    for R in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Allocate time. Lower ranks are harder to find but more valuable.
        # Give more time to lower R?
        # Actually, if R is too low, it will fail quickly.
        # Let's give a fixed budget per rank, decreasing as R increases?
        # Or just fixed 10s.
        time_limit = min(15.0, deadline - time.time() - 2)
        if time_limit <= 0:
            break
            
        result = _try_rank(R, n, m, p, seed=R, time_limit=time_limit)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
