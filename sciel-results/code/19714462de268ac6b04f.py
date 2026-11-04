"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x5 by 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 135  # Leave 15s safety margin
    
    # Precompute index mappings for vectorized operations
    ijks = list(product(range(n), range(m), range(p)))
    U_idx = np.array([i*m + j for i,j,k in ijks])
    V_idx = np.array([j*p + k for i,j,k in ijks])
    W_idx = np.array([k*n + i for i,j,k in ijks])
    N_triples = n * m * p  # 100
    
    def compute_sums(U, V, W):
        """Vectorized computation: returns array of sum_r U*V*W for each (i,j,k)."""
        terms = U[U_idx] * V[V_idx] * W[W_idx]
        return terms.sum(axis=1)
    
    def compute_errors(U, V, W):
        """Vectorized error computation: returns array of |sum_r U*V*W - 1| for each (i,j,k)."""
        sums = compute_sums(U, V, W)
        return np.abs(sums - 1.0)
    
    # Start with schoolbook as baseline
    best_rank = schoolbook_rank
    best_U = np.zeros((n*m, schoolbook_rank))
    best_V = np.zeros((m*p, schoolbook_rank))
    best_W = np.zeros((p*n, schoolbook_rank))
    for idx, (i,j,k) in enumerate(ijks):
        best_U[i*m+j, idx] = 1.0
        best_V[j*p+k, idx] = 1.0
        best_W[k*n+i, idx] = 1.0
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using numpy."""
        errors = compute_errors(U, V, W)
        return np.all(errors < 1e-9)
    
    def try_local_optimization(U, V, W, max_iters=300):
        """Iteratively adjust entries to reduce max error using smooth optimization."""
        R = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            sums = compute_sums(U, V, W)
            errors = np.abs(sums - 1.0)
            max_err = errors.max()
            if max_err < 1e-10:
                return True
            
            # Find worst (i,j,k)
            worst_idx = int(np.argmax(errors))
            i, j, k = ijks[worst_idx]
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            
            current = sums[worst_idx]
            residual = 1.0 - current
            
            best_improvement = 0
            best_update = None
            
            # Try updating U entries
            for r in range(R):
                vw_prod = V[v_idx, r] * W[w_idx, r]
                if abs(vw_prod) < 1e-12:
                    continue
                old_val = U[u_idx, r]
                # Newton-like step: delta = residual / (vw_prod * R_eff)
                # Use damped step to avoid overshooting
                delta = residual / vw_prod
                step = 0.1 * np.sign(delta) * min(abs(delta), 2.0)
                U[u_idx, r] = old_val + step
                new_sums = compute_sums(U, V, W)
                new_max = np.abs(new_sums - 1.0).max()
                if new_max < max_err:
                    best_improvement = max_err - new_max
                    best_update = ('U', r, U[u_idx, r])
                U[u_idx, r] = old_val
            
            # Try updating V entries
            for r in range(R):
                uw_prod = U[u_idx, r] * W[w_idx, r]
                if abs(uw_prod) < 1e-12:
                    continue
                old_val = V[v_idx, r]
                delta = residual / uw_prod
                step = 0.1 * np.sign(delta) * min(abs(delta), 2.0)
                V[v_idx, r] = old_val + step
                new_sums = compute_sums(U, V, W)
                new_max = np.abs(new_sums - 1.0).max()
                if new_max < max_err:
                    best_improvement = max_err - new_max
                    best_update = ('V', r, V[v_idx, r])
                V[v_idx, r] = old_val
            
            # Try updating W entries
            for r in range(R):
                uv_prod = U[u_idx, r] * V[v_idx, r]
                if abs(uv_prod) < 1e-12:
                    continue
                old_val = W[w_idx, r]
                delta = residual / uv_prod
                step = 0.1 * np.sign(delta) * min(abs(delta), 2.0)
                W[w_idx, r] = old_val + step
                new_sums = compute_sums(U, V, W)
                new_max = np.abs(new_sums - 1.0).max()
                if new_max < max_err:
                    best_improvement = max_err - new_max
                    best_update = ('W', r, W[w_idx, r])
                W[w_idx, r] = old_val
            
            # Apply best update
            if best_update is not None:
                factor, r, val = best_update
                if factor == 'U':
                    U[u_idx, r] = val
                elif factor == 'V':
                    V[v_idx, r] = val
                else:
                    W[w_idx, r] = val
            else:
                # No improvement found; try random perturbation to escape
                if it % 5 == 3:
                    rand_r = np.random.randint(R)
                    rand_factor = np.random.choice(['U', 'V', 'W'])
                    if rand_factor == 'U':
                        U[u_idx, rand_r] += np.random.randn() * 0.2
                    elif rand_factor == 'V':
                        V[v_idx, rand_r] += np.random.randn() * 0.2
                    else:
                        W[w_idx, rand_r] += np.random.randn() * 0.2
                else:
                    break
            
        errors = compute_errors(U, V, W)
        return errors.max() < 1e-10
    
    def try_exact_polish(U, V, W, max_iters=50):
        """High-precision polish: solve linear systems for exact correction."""
        R = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            sums = compute_sums(U, V, W)
            errors = np.abs(sums - 1.0)
            max_err = errors.max()
            if max_err < 1e-12:
                return True
            
            # For each error, try to correct by adjusting one entry
            worst_idx = int(np.argmax(errors))
            i, j, k = ijks[worst_idx]
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            
            current = sums[worst_idx]
            residual = 1.0 - current
            
            # Try all (factor, r) combinations and pick the one that reduces error most
            best_delta = None
            best_new_err = max_err
            
            for r in range(R):
                vw_prod = V[v_idx, r] * W[w_idx, r]
                if abs(vw_prod) > 1e-12:
                    delta = residual / vw_prod
                    U[u_idx, r] += delta
                    new_max = np.abs(compute_sums(U, V, W) - 1.0).max()
                    if new_max < best_new_err:
                        best_new_err = new_max
                        best_delta = ('U', r, delta)
                    U[u_idx, r] -= delta
                
                uw_prod = U[u_idx, r] * W[w_idx, r]
                if abs(uw_prod) > 1e-12:
                    delta = residual / uw_prod
                    V[v_idx, r] += delta
                    new_max = np.abs(compute_sums(U, V, W) - 1.0).max()
                    if new_max < best_new_err:
                        best_new_err = new_max
                        best_delta = ('V', r, delta)
                    V[v_idx, r] -= delta
                
                uv_prod = U[u_idx, r] * V[v_idx, r]
                if abs(uv_prod) > 1e-12:
                    delta = residual / uv_prod
                    W[w_idx, r] += delta
                    new_max = np.abs(compute_sums(U, V, W) - 1.0).max()
                    if new_max < best_new_err:
                        best_new_err = new_max
                        best_delta = ('W', r, delta)
                    W[w_idx, r] -= delta
            
            if best_delta is not None:
                factor, r, delta = best_delta
                if factor == 'U':
                    U[u_idx, r] += delta
                elif factor == 'V':
                    V[v_idx, r] += delta
                else:
                    W[w_idx, r] += delta
            else:
                break
        
        return compute_errors(U, V, W).max() < 1e-12
    
    # Try to reduce rank
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 4
        success = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Assign each of the n*m*p triples to one of target_rank columns
            for idx, (i,j,k) in enumerate(ijks):
                r = idx % target_rank
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            
            # Add small random perturbations to break symmetry
            U += np.random.randn(n*m, target_rank) * 0.02
            V += np.random.randn(m*p, target_rank) * 0.02
            W += np.random.randn(p*n, target_rank) * 0.02
            
            # Try local optimization then exact polish
            if try_local_optimization(U, V, W, max_iters=200):
                if verify(U, V, W, target_rank):
                    try_exact_polish(U, V, W, max_iters=30)
                    if verify(U, V, W, target_rank):
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        success = True
                        break
        
        if success:
            continue
        else:
            # Try sparse random initialization
            if time.time() > deadline:
                break
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                n_u = np.random.randint(2, 5)
                n_v = np.random.randint(2, 5)
                n_w = np.random.randint(2, 5)
                
                u_rows = np.random.choice(n*m, n_u, replace=False)
                v_rows = np.random.choice(m*p, n_v, replace=False)
                w_rows = np.random.choice(p*n, n_w, replace=False)
                
                U[u_rows, r] = np.random.choice([-1.0, 1.0], n_u)
                V[v_rows, r] = np.random.choice([-1.0, 1.0], n_v)
                W[w_rows, r] = np.random.choice([-1.0, 1.0], n_w)
            
            if try_local_optimization(U, V, W, max_iters=150):
                if verify(U, V, W, target_rank):
                    try_exact_polish(U, V, W, max_iters=30)
                    if verify(U, V, W, target_rank):
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
    
    # Final polish: try to verify and clean up
    if best_rank < schoolbook_rank:
        try_exact_polish(best_U, best_V, best_W, max_iters=50)
        errors = compute_errors(best_U, best_V, best_W)
        if errors.max() < 1e-9:
            # Round small values to zero for cleaner output
            best_U[np.abs(best_U) < 1e-8] = 0
            best_V[np.abs(best_V) < 1e-8] = 0
            best_W[np.abs(best_W) < 1e-8] = 0
            # Re-verify after rounding
            if not verify(best_U, best_V, best_W, best_rank):
                # Restore if rounding broke it
                best_U = np.array(best_U, dtype=float)
                best_V = np.array(best_V, dtype=float)
                best_W = np.array(best_W, dtype=float)
                try_exact_polish(best_U, best_V, best_W, max_iters=30)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
