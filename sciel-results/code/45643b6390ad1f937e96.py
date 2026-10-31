"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def autoconv_peak(x):
        """Compute max absolute autoconvolution value."""
        if np.all(x == 0):
            return 0.0
        ac = fftconvolve(x, x, mode='full')
        return np.max(np.abs(ac))

    def score(x):
        """max|autoconv| / (sum)^2 - lower is better."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return autoconv_peak(x) / (s * s)

    def local_search_binary(x, max_iters=100, batch_size=30):
        """Local search on binary {-1,1} sequence by flipping signs."""
        best_val = score(x)
        improved = True
        it = 0
        while improved and it < max_iters:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            trial_indices = np.random.default_rng(it).choice(N, size=min(batch_size, N), replace=False)
            for idx in trial_indices:
                if time.time() > deadline:
                    break
                orig = x[idx]
                x[idx] = -orig
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    x[idx] = orig
        return x, best_val

    def local_search_sparse(x, max_iters=50, rng=None):
        """Local search on sparse signed sequence: flip, add, remove."""
        if rng is None:
            rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
        best_val = score(x)
        improved = True
        it = 0
        while improved and it < max_iters:
            if time.time() > deadline:
                break
            improved = False
            it += 1

            # Try flipping a random non-zero entry
            non_zero_idx = np.where(x != 0)[0]
            if len(non_zero_idx) > 0:
                trial_idx = rng.choice(non_zero_idx)
                orig = x[trial_idx]
                x[trial_idx] = -orig
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    x[trial_idx] = orig

            # Try adding a new non-zero entry
            zero_idx = np.where(x == 0)[0]
            if len(zero_idx) > 0:
                add_idx = rng.choice(zero_idx)
                x[add_idx] = rng.choice([-1.0, 1.0])
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    x[add_idx] = 0

            # Try removing a non-zero entry
            non_zero_idx = np.where(x != 0)[0]
            if len(non_zero_idx) > 1:
                rem_idx = rng.choice(non_zero_idx)
                orig_rem = x[rem_idx]
                x[rem_idx] = 0
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    x[rem_idx] = orig_rem

        return x, best_val

    # Start with all ones as baseline
    best_x = np.ones(N)
    best_val = score(best_x)

    rng = np.random.default_rng(42)

    # Phase 1: Explore binary {-1,1} sign patterns
    # For unit magnitude, sum is bounded by N, and we want large |sum| while minimizing max|autoconv|
    n_binary_restarts = 30
    for r in range(n_binary_restarts):
        if time.time() > deadline - 20:
            break

        x = rng.choice([-1.0, 1.0], size=N)
        val = score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

        # Local search on this binary pattern
        x_ls, val_ls = local_search_binary(x.copy(), max_iters=50, batch_size=25)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()

    # Phase 2: Explore sparse signed sequences with various densities
    sparsity_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for density in sparsity_levels:
        if time.time() > deadline - 15:
            break

        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break

            # Create sparse signed sequence
            mask = rng.random(N) < density
            x = np.zeros(N)
            x[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            
            # Ensure non-zero sum
            s = np.sum(x)
            if abs(s) < 1e-12:
                # Flip a few elements to break symmetry
                flip_idx = rng.choice(N, size=max(1, N // 20), replace=False)
                x[flip_idx] *= -1
                s = np.sum(x)
                if abs(s) < 1e-12:
                    continue

            val = score(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

            # Local search on sparse sequence
            x_ls, val_ls = local_search_sparse(x.copy(), max_iters=40, rng=rng)
            if val_ls < best_val:
                best_val = val_ls
                best_x = x_ls.copy()

    # Phase 3: Final polish - try perturbations on the best solution found
    if time.time() < deadline - 5:
        x = best_x.copy()
        nonzero_idx = np.where(x != 0)[0]
        
        for trial in range(200):
            if time.time() > deadline - 2:
                break
            
            if len(nonzero_idx) > 0 and rng.random() < 0.7:
                # Flip or modify a non-zero entry
                idx = rng.choice(nonzero_idx)
                orig = x[idx]
                
                if abs(orig) == 1.0:
                    # Binary: flip sign
                    x[idx] = -orig
                else:
                    # Sparse: try different magnitude or sign
                    x[idx] = rng.choice([-1.0, 1.0])
                
                new_val = score(x)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    best_x = x.copy()
                    nonzero_idx = np.where(x != 0)[0]
                else:
                    x[idx] = orig
            else:
                # Add or modify a zero entry
                zero_idx = np.where(x == 0)[0]
                if len(zero_idx) > 0:
                    idx = rng.choice(zero_idx)
                    x[idx] = rng.choice([-1.0, 1.0])
                    new_val = score(x)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        best_x = x.copy()
                        nonzero_idx = np.where(x != 0)[0]
                    else:
                        x[idx] = 0

    # Ensure valid result
    if abs(np.sum(best_x)) < 1e-6:
        best_x = np.ones(N)

    return best_x
# EVOLVE-BLOCK-END
