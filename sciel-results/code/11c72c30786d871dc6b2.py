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
    
    # We know the trivial rank is 30.
    # Lower bound is roughly max(N_U, N_V, N_W) = 15, but tensor rank is often higher.
    # Let's search from 15 up to 29.
    
    # Helper to verify a solution
    def verify_solution(U, V, W):
        # U, V, W are numpy arrays
        terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)

    # Helper to round and verify
    def try_round_and_verify(U_opt, V_opt, W_opt, R):
        # Round to nearest integer first, then half-integer?
        # The problem asks for integer/half-integer.
        # Let's try rounding to nearest 0.5
        U_round = np.round(U_opt * 2) / 2
        V_round = np.round(V_opt * 2) / 2
        W_round = np.round(W_opt * 2) / 2
        
        if verify_solution(U_round, V_round, W_round):
            return U_round, V_round, W_round
        
        # Try rounding to nearest integer
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        
        if verify_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
            
        return None

    # Search loop
    # We iterate R from 15 to 29.
    # For each R, we try multiple random restarts with optimization.
    
    # To save time, we can be smarter about initialization or use a different optimizer.
    # BFGS is good for small problems.
    # Let's limit the number of restarts per rank based on remaining time.
    
    # Estimated time per optimization step: ~0.1s to 1s depending on convergence.
    # Let's allow ~20 restarts per rank if time permits.
    
    for R in range(15, best_rank):
        if time.time() > deadline - 10: # Leave 10s for final checks
            break
            
        found_solution = None
        
        # Max restarts: try to fit within ~2s per rank on average?
        # If R=15 is hard, we might spend more time.
        # Let's just try a fixed number of restarts, e.g., 10, and see.
        max_restarts = 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Using a fixed seed for reproducibility in the search,
            # but varying it per restart to explore different basins.
            seed = 12345 + R * 1000 + restart
            np.random.seed(seed)
            
            # Scale initialization: values should be roughly O(1/sqrt(R)) to keep products O(1)
            scale = 1.0 / np.sqrt(R)
            U_init = np.random.randn(N_U, R) * scale
            V_init = np.random.randn(N_V, R) * scale
            W_init = np.random.randn(N_W, R) * scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def loss(x):
                U = x[:N_U * R].reshape(N_U, R)
                V = x[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W = x[N_U * R + N_V * R :].reshape(N_W, R)
                
                # Compute sum for each (i,j,k)
                terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
                sums = terms.sum(axis=1)
                
                errors = sums - 1.0
                return np.sum(errors ** 2)
            
            try:
                # Use BFGS
                # maxiter=100, gtol=1e-8
                res = minimize(loss, x0, method='BFGS', options={'maxiter': 100, 'gtol': 1e-8})
            except Exception:
                continue
                
            # Check if converged to a low loss
            if res.fun < 1e-4: # Threshold for "close enough" to try rounding
                x_opt = res.x
                U_opt = x_opt[:N_U * R].reshape(N_U, R)
                V_opt = x_opt[N_U * R : N_U * R + N_V * R].reshape(N_V, R)
                W_opt = x_opt[N_U * R + N_V * R :].reshape(N_W, R)
                
                rounded = try_round_and_verify(U_opt, V_opt, W_opt, R)
                
                if rounded is not None:
                    U_round, V_round, W_round = rounded
                    found_solution = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    best_rank = R
                    break # Found minimal R for this loop iteration? No, we break restart loop.
        
        if found_solution:
            best_solution = found_solution
            # Since we iterate R increasing, the first valid R is the minimal one found.
            # We can return immediately as we are searching for the minimum.
            return best_solution
            
    return best_solution
