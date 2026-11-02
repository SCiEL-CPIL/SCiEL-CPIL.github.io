import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import product
from collections import defaultdict

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal decomposition for 2x4 * 4x5 matrix multiplication.
    Starts with a Rank-36 Block-Strassen decomposition and uses numerical optimization
    to reduce the rank (try 35, 34, 33, 32) by minimizing the tensor reconstruction error.
    If a lower rank is found with near-zero error, it rounds to nearest half-integer
    and verifies. Returns the best valid decomposition found.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # This corresponds to the standard matrix multiplication tensor where
    # C_ik = sum_j A_ij B_jk
    T_target = np.ones((n, m, p))
    
    def get_block_strassen_36():
        terms = []
        # Strassen 2x2x2 terms
        def add_strassen_2x2x2(i0, j0, k0):
            # 7 terms
            terms.append((
                {(i0, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(i0, k0): 1, (i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0): 1, (i0+1, j0+1): 1},
                {(j0, k0): 1},
                {(i0+1, k0): 1, (i0+1, k0+1): -1}
            ))
            terms.append((
                {(i0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(i0, k0+1): 1, (i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(i0, k0): 1, (i0+1, k0): 1}
            ))
            terms.append((
                {(i0, j0): 1, (i0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(i0, k0): -1, (i0, k0+1): 1}
            ))
            terms.append((
                {(i0+1, j0): 1, (i0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(i0+1, k0+1): 1}
            ))
            terms.append((
                {(i0, j0+1): 1, (i0+1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(i0, k0): 1}
            ))
            
        def add_schoolbook_2x2x1(i0, j0, k0):
            # 4 terms
            terms.append(( {(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1} ))
            terms.append(( {(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1} ))
            terms.append(( {(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1} ))
            terms.append(( {(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1} ))

        # Block decomposition for 2x4 * 4x5
        # Split m=4 into 2+2.
        add_strassen_2x2x2(0, 0, 0)
        add_strassen_2x2x2(0, 2, 0)
        add_strassen_2x2x2(0, 0, 2)
        add_strassen_2x2x2(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return terms

    def terms_to_matrices(terms):
        R = len(terms)
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        for r, (u_d, v_d, w_d) in enumerate(terms):
            for (i, j), val in u_d.items():
                U[i*m + j, r] += val
            for (j, k), val in v_d.items():
                V[j*p + k, r] += val
            for (i, k), val in w_d.items():
                W[k*n + i, r] += val
        return U, V, W

    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return T

    def error_metric(U_flat, target_shape_U, V_flat, target_shape_V, W_flat, target_shape_W):
        U = U_flat.reshape(target_shape_U)
        V = V_flat.reshape(target_shape_V)
        W = W_flat.reshape(target_shape_W)
        T_comp = compute_tensor(U, V, W)
        return np.sum((T_comp - T_target)**2)

    # Generate initial 36 terms
    init_terms = get_block_strassen_36()
    U0, V0, W0 = terms_to_matrices(init_terms)
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = 36
    best_err = 1e-6 # Should be 0
    
    def try_optimize(target_rank, U_init, V_init, W_init, max_iter=100):
        """
        Attempts to optimize a given initial guess for a specific rank.
        Returns (success, U_opt, V_opt, W_opt)
        """
        if time.time() > deadline - 5:
            return False, None, None, None
        
        U_size = n * m * target_rank
        V_size = m * p * target_rank
        W_size = p * n * target_rank
        total_size = U_size + V_size + W_size
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U = x[:U_size].reshape(n*m, target_rank)
            V = x[U_size:U_size+V_size].reshape(m*p, target_rank)
            W = x[U_size+V_size:].reshape(p*n, target_rank)
            return error_metric(U, (n*m, target_rank), V, (m*p, target_rank), W, (p*n, target_rank))
        
        try:
            # Use L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter, 'maxfun': 5000})
            
            if res.fun < 1e-8:
                U_opt = res.x[:U_size].reshape(n*m, target_rank)
                V_opt = res.x[U_size:U_size+V_size].reshape(m*p, target_rank)
                W_opt = res.x[U_size+V_size:].reshape(p*n, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify rounded version
                T_rounded = compute_tensor(U_rounded, V_rounded, W_rounded)
                if np.allclose(T_rounded, T_target, atol=1e-9):
                    return True, U_rounded, V_rounded, W_rounded
                else:
                    # If rounding fails, try integer rounding if close
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    T_int = compute_tensor(U_int, V_int, W_int)
                    if np.allclose(T_int, T_target, atol=1e-9):
                        return True, U_int, V_int, W_int
            return False, None, None, None
        except Exception:
            return False, None, None, None

    # Try to reduce rank from 35 down to 28
    for target_rank in range(35, 27, -1):
        if time.time() > deadline - 10:
            break
            
        # Strategy 1: Take first target_rank columns from 36-rank decomp
        U_init1 = U0[:, :target_rank]
        V_init1 = V0[:, :target_rank]
        W_init1 = W0[:, :target_rank]
        
        success, U_opt, V_opt, W_opt = try_optimize(target_rank, U_init1, V_init1, W_init1)
        if success:
            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
            best_rank = target_rank
            continue # Found a valid rank, try to go lower
            
        # Strategy 2: Random restarts (limited)
        if time.time() < deadline - 20:
            for _ in range(5):
                if time.time() > deadline - 15:
                    break
                U_rand = np.random.randn(n*m, target_rank) * 0.5
                V_rand = np.random.randn(m*p, target_rank) * 0.5
                W_rand = np.random.randn(p*n, target_rank) * 0.5
                success, U_opt, V_opt, W_opt = try_optimize(target_rank, U_rand, V_rand, W_rand, max_iter=50)
                if success:
                    best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    best_rank = target_rank
                    break # Found valid rank for this target, move to lower rank

    return best_result
# EVOLVE-BLOCK-END
