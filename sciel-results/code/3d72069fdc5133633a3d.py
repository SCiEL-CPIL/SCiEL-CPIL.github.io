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
    
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Precompute index mappings for vectorized operations
    # For each (i,j,k) triple, we need:
    #   U_idx = i*m + j
    #   V_idx = j*p + k  
    #   W_idx = k*n + i
    ijks = list(product(range(n), range(m), range(p)))
    U_idx = np.array([i*m + j for i,j,k in ijks])
    V_idx = np.array([j*p + k for i,j,k in ijks])
    W_idx = np.array([k*n + i for i,j,k in ijks])
    
    def compute_errors(U, V, W):
        """Vectorized error computation: returns array of |sum_r U*V*W - 1| for each (i,j,k)."""
        R = U.shape[1]
        # For each (i,j,k) and each r, compute U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Shape: (n*m*p, R)
        terms = U[U_idx] * V[V_idx] * W[W_idx]
        sums = terms.sum(axis=1)
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
    
    def try_local_optimization(U, V, W, max_iters=200):
        """Iteratively adjust entries to reduce max error."""
        R = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline:
                return False
            
            errors = compute_errors(U, V, W)
            max_err = errors.max()
            if max_err < 1e-9:
                return True
            
            # Find worst (i,j,k)
            worst_idx = np.argmax(errors)
            i, j, k = ijks[worst_idx]
            u_idx = i*m + j
            v_idx = j*p + k
            w_idx = k*n + i
            
            current = (U[u_idx] * V[v_idx] * W[w_idx]).sum()
            
            # Try adjusting each of the three factors
            # For each factor, compute gradient-like update
            best_improvement = 0
            best_update = None
            
            # Try updating U
            for r in range(R):
                if U[u_idx, r] == 0:
                    continue
                old_val = U[u_idx, r]
                # Adjust U to reduce error at worst point
                # We want sum_r U*V*W = 1, so delta_U should be (1-current)/V*W
                target_delta = (1.0 - current) / max(abs(V[v_idx, r] * W[w_idx, r]), 1e-10)
                U[u_idx, r] = old_val + 0.05 * np.sign(target_delta) * min(abs(target_delta), 1.0)
                new_errors = compute_errors(U, V, W)
                new_max = new_errors.max()
                if new_max < max_err:
                    best_improvement = max_err - new_max
                    best_update = ('U', r, U[u_idx, r])
                U[u_idx, r] = old_val
            
            # Try updating V
            for r in range(R):
                if V[v_idx, r] == 0:
                    continue
                old_val = V[v_idx, r]
                target_delta = (1.0 - current) / max(abs(U[u_idx, r] * W[w_idx, r]), 1e-10)
                V[v_idx, r] = old_val + 0.05 * np.sign(target_delta) * min(abs(target_delta), 1.0)
                new_errors = compute_errors(U, V, W)
                new_max = new_errors.max()
                if new_max < max_err:
                    best_improvement = max_err - new_max
                    best_update = ('V', r, V[v_idx, r])
                V[v_idx, r] = old_val
            
            # Try updating W
            for r in range(R):
                if W[w_idx, r] == 0:
                    continue
                old_val = W[w_idx, r]
                target_delta = (1.0 - current) / max(abs(U[u_idx, r] * V[v_idx, r]), 1e-10)
                W[w_idx, r] = old_val + 0.05 * np.sign(target_delta) * min(abs(target_delta), 1.0)
                new_errors = compute_errors(U, V, W)
                new_max = new_errors.max()
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
            
            # Also try random perturbations to escape local minima
            if it % 10 == 5:
                rand_r = np.random.randint(R)
                rand_factor = np.random.choice(['U', 'V', 'W'])
                if rand_factor == 'U':
                    U[u_idx, rand_r] += np.random.randn() * 0.1
                elif rand_factor == 'V':
                    V[v_idx, rand_r] += np.random.randn() * 0.1
                else:
                    W[w_idx, rand_r] += np.random.randn() * 0.1
        
        errors = compute_errors(U, V, W)
        return errors.max() < 1e-9
    
    # Try to reduce rank
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 0), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 3
        success = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook-like structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Assign each of the n*m*p triples to one of target_rank columns
            # Use round-robin assignment
            for idx, (i,j,k) in enumerate(ijks):
                r = idx % target_rank
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
            
            # Add small random perturbations to break symmetry
            U += np.random.randn(n*m, target_rank) * 0.01
            V += np.random.randn(m*p, target_rank) * 0.01
            W += np.random.randn(p*n, target_rank) * 0.01
            
            # Try local optimization
            if try_local_optimization(U, V, W, max_iters=150):
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
            # If we couldn't improve at this rank, try one more time with different init
            if time.time() > deadline:
                break
            # Try a different initialization strategy: sparse random
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Random sparse initialization
            for r in range(target_rank):
                # Pick random nonzero pattern
                n_u = np.random.randint(1, 4)
                n_v = np.random.randint(1, 4)
                n_w = np.random.randint(1, 4)
                
                u_rows = np.random.choice(n*m, n_u, replace=False)
                v_rows = np.random.choice(m*p, n_v, replace=False)
                w_rows = np.random.choice(p*n, n_w, replace=False)
                
                U[u_rows, r] = np.random.choice([-1.0, 1.0], n_u)
                V[v_rows, r] = np.random.choice([-1.0, 1.0], n_v)
                W[w_rows, r] = np.random.choice([-1.0, 1.0], n_w)
            
            if try_local_optimization(U, V, W, max_iters=100):
                if verify(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
    
    # Final polish: try to verify and clean up
    if best_rank < schoolbook_rank:
        errors = compute_errors(best_U, best_V, best_W)
        if errors.max() < 1e-9:
            # Round small values to zero for cleaner output
            best_U[np.abs(best_U) < 1e-8] = 0
            best_V[np.abs(best_V) < 1e-8] = 0
            best_W[np.abs(best_W) < 1e-8] = 0
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
