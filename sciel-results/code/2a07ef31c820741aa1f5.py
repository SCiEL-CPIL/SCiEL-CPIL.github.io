import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Dimensions
n, m, p = 2, 3, 5
N_U = n * m  # 6
N_V = m * p  # 15
N_W = p * n  # 10

# Precompute indices for the tensor decomposition
# U index: i * m + j
# V index: j * p + k
# W index: k * n + i
_u_indices_list = []
_v_indices_list = []
_w_indices_list = []
for i in range(n):
    for j in range(m):
        for k in range(p):
            u_indices_list_tmp = i * m + j
            v_indices_list_tmp = j * p + k
            w_indices_list_tmp = k * n + i
            _u_indices_list.append(u_indices_list_tmp)
            _v_indices_list.append(v_indices_list_tmp)
            _w_indices_list.append(w_indices_list_tmp)

U_INDICES = np.array(_u_indices_list)
V_INDICES = np.array(_v_indices_list)
W_INDICES = np.array(_w_indices_list)
NUM_TERMS = len(_u_indices_list)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Schoolbook fallback (rank 30)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_solution = get_schoolbook()
    best_rank = n * m * p
    
    # Helper to verify a solution
    def verify_solution(U, V, W):
        # U, V, W are numpy arrays
        terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-10)

    # Helper to round and verify
    def try_round_and_verify(U_opt, V_opt, W_opt, R):
        # Try rounding to nearest half-integer
        U_half = np.round(U_opt * 2) / 2
        V_half = np.round(V_opt * 2) / 2
        W_half = np.round(W_opt * 2) / 2
        
        if verify_solution(U_half, V_half, W_half):
            return U_half, V_half, W_half
        
        # Try rounding to nearest integer
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        
        if verify_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try a few nearby rounding candidates
        for delta in [-0.25, 0.25]:
            U_try = np.round(U_opt + delta) / 2 * 2
            V_try = np.round(V_opt + delta) / 2 * 2
            W_try = np.round(W_opt + delta) / 2 * 2
            if verify_solution(U_try, V_try, W_try):
                return U_try, V_try, W_try
        
        return None

    # Better initialization: use structured random with some sparsity
    def structured_init(R, seed):
        np.random.seed(seed)
        
        # Start with small random values
        U_init = np.random.randn(N_U, R) * 0.5
        V_init = np.random.randn(N_V, R) * 0.5
        W_init = np.random.randn(N_W, R) * 0.5
        
        # Add some structured components: for each term, pick a random (i,j,k) and set corresponding entries
        for r in range(R):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            
            # With some probability, set these to 1
            if np.random.rand() < 0.3:
                U_init[i * m + j, r] = 1.0
            if np.random.rand() < 0.3:
                V_init[j * p + k, r] = 1.0
            if np.random.rand() < 0.3:
                W_init[k * n + i, r] = 1.0
        
        return U_init, V_init, W_init

    # Search loop
    for R in range(15, best_rank):
        if time.time() > deadline - 10:
            break
            
        found_solution = None
        
        # More restarts for smaller R since they're harder
        max_restarts = 25 if R <= 18 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            seed = 12345 + R * 1000 + restart
            
            # Use structured initialization
            U_init, V_init, W_init = structured_init(R, seed)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(x):
                U = x[:N_U * R].reshape(N_U, R)
                V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W = x[N_U * R + N_V * R :].reshape(N_W, R)
                
                terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
                sums = terms.sum(axis=1)
                
                errors = sums - 1.0
                return np.sum(errors ** 2)
            
            try:
                # Use L-BFGS-B for better handling of larger problems
                res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-10})
            except Exception:
                continue
                
            # Check if converged to a low loss
            if res.fun < 1e-6:
                x_opt = res.x
                U_opt = x_opt[:N_U * R].reshape(N_U, R)
                V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Polish with a final tight optimization
                def polish_loss(x):
                    U = x[:N_U * R].reshape(N_U, R)
                    V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                    W = x[N_U * R + N_V * R :].reshape(N_W, R)
                    terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
                    sums = terms.sum(axis=1)
                    errors = sums - 1.0
                    return np.sum(errors ** 2)
                
                try:
                    res2 = minimize(polish_loss, x_opt, method='BFGS', options={'maxiter': 500, 'gtol': 1e-12})
                    if res2.fun < res.fun:
                        x_opt = res2.x
                        U_opt = x_opt[:N_U * R].reshape(N_U, R)
                        V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                        W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                except Exception:
                    pass
                
                rounded = try_round_and_verify(U_opt, V_opt, W_opt, R)
                
                if rounded is not None:
                    U_round, V_round, W_round = rounded
                    found_solution = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    best_rank = R
                    break
        
        if found_solution:
            best_solution = found_solution
            return best_solution
            
    return best_solution
