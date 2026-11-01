import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    start_time = time.time()
    time_limit = 100.0 # Leave margin for overhead

    target_rank = 20
    
    # Precompute indices for the target tensor
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Target is 1 if i=i', j=j', k=k'
    # We define the residual for all combinations of (i,j), (j',k), (k',i')
    
    def residuals(params):
        # params is a flat vector of length target_rank * (n*m + m*p + p*n)
        U_flat = params[:target_rank * n * m]
        V_flat = params[target_rank * n * m : target_rank * (n * m + m * p)]
        W_flat = params[target_rank * (n * m + m * p) :]
        
        U = U_flat.reshape(target_rank, n * m).T # (n*m, rank)
        V = V_flat.reshape(target_rank, m * p).T # (m*p, rank)
        W = W_flat.reshape(target_rank, p * n).T # (p*n, rank)
        
        res = []
        # Iterate over all index combinations
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for jp in range(m * p):
                    # jp corresponds to j', k
                    j_prime = jp // p
                    k = jp % p
                    v_idx = jp
                    for kp in range(p * n):
                        # kp corresponds to k', i'
                        k_prime = kp // n
                        i_prime = kp % n
                        w_idx = kp
                        
                        val = np.dot(U[u_idx], np.dot(V[v_idx], W[w_idx]))
                        # Wait, np.dot(U[u_idx], V[v_idx]) is a vector of length rank.
                        # Then dot with W[w_idx].
                        # This is sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r]
                        
                        # Target value
                        if i == i_prime and j == j_prime and k == k_prime:
                            target = 1.0
                        else:
                            target = 0.0
                            
                        res.append(val - target)
                        
        return np.array(res)

    # This residual function is very slow if called repeatedly in Python loops.
    # We need to vectorize it.
    
    def residuals_vectorized(params):
        U_flat = params[:target_rank * n * m]
        V_flat = params[target_rank * n * m : target_rank * (n * m + m * p)]
        W_flat = params[target_rank * (n * m + m * p) :]
        
        U = U_flat.reshape(target_rank, n * m).T # (n*m, rank)
        V = V_flat.reshape(target_rank, m * p).T # (m*p, rank)
        W = W_flat.reshape(target_rank, p * n).T # (p*n, rank)
        
        # We want sum_r U[i,j,r] V[j',k,r] W[k',i',r]
        # This is a tensor contraction.
        # Let's construct the full tensor T = sum_r outer(U[:,r], V[:,r], W[:,r])
        # T has shape (n*m, m*p, p*n)
        # We can do this by iterating r, but that's slow.
        # Alternatively, we can use einsum.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        
        # U is (n*m, rank)
        # V is (m*p, rank)
        # W is (p*n, rank)
        
        # T = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a (n*m, m*p, p*n) array.
        # 6 * 12 * 8 = 576 elements.
        # This is fast enough.
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Construct target tensor
        Target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    Target[u_idx, v_idx, w_idx] = 1.0
        
        return (T - Target).flatten()

    # We will use least_squares
    # Initial guess
    # We can try to initialize with random values
    # Or use the schoolbook solution padded/truncated?
    # Schoolbook has rank 24. We want 20.
    # We can try to set 4 columns to 0 and optimize the rest?
    # Or just random start.
    
    max_restarts = 50
    found = False
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale: entries should be around 1 or 0.5
        scale = 1.0
        params0 = np.random.randn(target_rank * (n * m + m * p + p * n)) * scale
        
        # Try to solve
        try:
            result = least_squares(residuals_vectorized, params0, method='lm', max_nfev=1000)
            if result.success or result.cost < 1e-8:
                # Check if solution is close to half-integers
                U_sol = result.x[:target_rank * n * m].reshape(target_rank, n * m).T
                V_sol = result.x[target_rank * n * m : target_rank * (n * m + m * p)].reshape(target_rank, m * p).T
                W_sol = result.x[target_rank * (n * m + m * p) :].reshape(target_rank, p * n).T
                
                # Round to nearest half-integer
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify exactness
                T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                Target = np.zeros((n * m, m * p, p * n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            Target[u_idx, v_idx, w_idx] = 1.0
                            
                if np.allclose(T_check, Target, atol=1e-9):
                    # Convert to lists of integers/half-integers
                    # The prompt asks for integer/half-integer entries.
                    # We return lists of floats (or ints if integer)
                    U_list = U_int.tolist()
                    V_list = V_int.tolist()
                    W_list = W_int.tolist()
                    
                    # Check if we found a better rank
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        found = True
                        break # Found the target rank, can stop early
        except Exception as e:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
