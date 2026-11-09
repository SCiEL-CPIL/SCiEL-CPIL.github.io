# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 4, 8]

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] using einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    return np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

def _verify_decomposition(decomp, T, tol=1e-9):
    """Check if the decomposition exactly reproduces T."""
    U, V, W = decomp['U'], decomp['V'], decomp['W']
    T_approx = _tensor_from_decomp(U, V, W)
    return np.allclose(T_approx, T, atol=tol, rtol=0)

def _loss_and_grad(U, V, W, T):
    """
    Compute loss ||T - sum_r U_r V_r W_r||^2 and gradients dU, dV, dW.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    T: (n*m, m*p, p*n)
    """
    T_pred = np.einsum('ar,br,cr->abc', U, V, W)
    diff = T - T_pred
    
    loss = np.sum(diff ** 2)
    
    dU = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
    dV = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
    dW = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return loss, dU, dV, dW

def _optimize_decomposition_l_bfgs(n, m, p, T, R, max_iters=200, deadline=None, seed=42):
    """
    Optimize decomposition of rank R using L-BFGS-B with better initialization.
    """
    # Flatten U, V, W into a single vector
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Try multiple initialization strategies
    best_result = None
    best_loss = np.inf
    
    init_strategies = [
        'random_small', 'random_medium', 'structured', 'mixed'
    ]
    
    for strategy in init_strategies:
        if deadline is not None and time.time() > deadline:
            break
            
        np.random.seed(seed)
        
        if strategy == 'random_small':
            U0 = np.random.randn(dim_U, R) * 0.1
            V0 = np.random.randn(dim_V, R) * 0.1
            W0 = np.random.randn(dim_W, R) * 0.1
        elif strategy == 'random_medium':
            U0 = np.random.randn(dim_U, R) * 0.5
            V0 = np.random.randn(dim_V, R) * 0.5
            W0 = np.random.randn(dim_W, R) * 0.5
        elif strategy == 'structured':
            # Initialize with sparse patterns
            U0 = np.zeros((dim_U, R))
            V0 = np.zeros((dim_V, R))
            W0 = np.zeros((dim_W, R))
            for r in range(R):
                # Randomly select a few non-zero entries
                u_idx = np.random.choice(dim_U, size=min(3, dim_U), replace=False)
                v_idx = np.random.choice(dim_V, size=min(3, dim_V), replace=False)
                w_idx = np.random.choice(dim_W, size=min(3, dim_W), replace=False)
                U0[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                V0[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                W0[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
        else:  # mixed
            U0 = np.random.randn(dim_U, R) * 0.3
            V0 = np.random.randn(dim_V, R) * 0.3
            W0 = np.random.randn(dim_W, R) * 0.3
            # Add some structure
            for r in range(min(5, R)):
                U0[0, r] = 1.0
                V0[0, r] = 1.0
                W0[0, r] = 1.0
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(x):
            if deadline is not None and time.time() > deadline:
                return 1e10, np.zeros_like(x)
                
            U = x[:dim_U * R].reshape(dim_U, R)
            V = x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
            W = x[dim_U * R + dim_V * R :].reshape(dim_W, R)
            
            loss, dU, dV, dW = _loss_and_grad(U, V, W, T)
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            return loss, grad

        # Use L-BFGS-B
        try:
            res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', jac=True, 
                                           options={'maxiter': max_iters, 'disp': False})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res
        except Exception as e:
            continue
    
    if best_result is not None and (best_result.success or best_result.fun < 1e-8):
        U_opt = best_result.x[:dim_U * R].reshape(dim_U, R)
        V_opt = best_result.x[dim_U * R : dim_U * R + dim_V * R].reshape(dim_V, R)
        W_opt = best_result.x[dim_U * R + dim_V * R :].reshape(dim_W, R)
        return U_opt, V_opt, W_opt, best_result.fun
    else:
        return None, None, None, best_loss

def _snap_to_half_integers(U, V, W, tol=1e-6):
    """
    Snap values to nearest half-integer if close enough.
    """
    def snap(arr):
        arr_2 = arr * 2
        arr_rounded = np.round(arr_2)
        if np.allclose(arr_2, arr_rounded, atol=tol):
            return arr_rounded / 2.0
        else:
            return arr

    U_snapped = snap(U)
    V_snapped = snap(V)
    W_snapped = snap(W)
    
    T_pred = np.einsum('ar,br,cr->abc', U_snapped, V_snapped, W_snapped)
    
    return U_snapped, V_snapped, W_snapped, T_pred

def _try_find_smaller_decomposition(n, m, p, deadline):
    """
    Search for a decomposition with rank < n*m*p.
    """
    schoolbook_rank = n * m * p
    T = _build_target_tensor(n, m, p)
    
    best_rank = schoolbook_rank
    best_decomp = _schoolbook(n, m, p)
    
    # Try ranks from schoolbook_rank - 1 down to a limit
    # We will try a few ranks below schoolbook
    ranks_to_try = []
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 1), -1):
        ranks_to_try.append(R)
        
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for restart in range(8):
            if time.time() > deadline:
                break
                
            # Different seed for each restart
            seed = 42 + restart * 7
            # Use L-BFGS-B with multiple initialization strategies
            U_opt, V_opt, W_opt, loss = _optimize_decomposition_l_bfgs(n, m, p, T, R, max_iters=150, deadline=deadline, seed=seed)
            
            if U_opt is not None and loss < 1e-8:
                # Try to snap to half-integers
                U_s, V_s, W_s, T_pred = _snap_to_half_integers(U_opt, V_opt, W_opt)
                if np.allclose(T_pred, T, atol=1e-9, rtol=0):
                    decomp = {
                        "U": U_s.tolist(),
                        "V": V_s.tolist(),
                        "W": W_s.tolist()
                    }
                    return R, decomp
                    found = True
                    break
                    
    return best_rank, best_decomp

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s buffer for the 150s limit
    
    # Try to find a smaller decomposition
    rank, decomp = _try_find_smaller_decomposition(n, m, p, deadline)
    
    # Verify the decomposition to ensure it is correct
    T = _build_target_tensor(n, m, p)
    if not _verify_decomposition(decomp, T):
        # Fallback to schoolbook if verification fails
        decomp = _schoolbook(n, m, p)
    
    return decomp
# EVOLVE-BLOCK-END
