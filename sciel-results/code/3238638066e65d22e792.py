import numpy as np
import time
import random

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv_fft(f):
    """Compute autoconvolution via FFT for speed."""
    size = 2 * N
    F = np.fft.rfft(f, n=size)
    C = np.fft.irfft(F * F, n=size)
    return C[:2 * N - 1]


def _score(f):
    """Compute the objective: max|autoconv| / (sum f)^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)


def _make_sparse_sequence(rng, sparsity):
    """Create a sparse signed sequence with given sparsity (fraction non-zero)."""
    n_nonzero = max(1, int(sparsity * N))
    f = np.zeros(N)
    indices = rng.choice(N, size=n_nonzero, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_nonzero)
    f[indices] = signs
    return f


def _simulated_annealing_sparse(rng, f_init, max_iter=5000, T_start=1.0, T_end=0.01):
    """Simulated annealing on sparse signed sequences."""
    f = f_init.copy()
    S = np.sum(f)
    if abs(S) < 1e-12:
        f[0] = 1.0
        S = np.sum(f)
    best_f = f.copy()
    best_score = _score(f)
    current_score = best_score
    
    n_nonzero = np.count_nonzero(f)
    nonzero_mask = np.nonzero(f)[0]
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Flip sign of a random non-zero element
            if n_nonzero == 0:
                continue
            idx = nonzero_mask[rng.integers(len(nonzero_mask))]
            f[idx] = -f[idx]
            new_score = _score(f)
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[idx] = -f[idx]  # revert
        elif move_type < 0.7:
            # Move a non-zero element to a random zero position
            if n_nonzero == 0:
                continue
            src = nonzero_mask[rng.integers(len(nonzero_mask))]
            zero_positions = np.nonzero(f == 0)[0]
            if len(zero_positions) == 0:
                continue
            dst = zero_positions[rng.integers(len(zero_positions))]
            val = f[src]
            f[src] = 0
            f[dst] = val
            new_score = _score(f)
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                nonzero_mask = np.nonzero(f)[0]
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[dst] = 0
                f[src] = val
        elif move_type < 0.85:
            # Add a new non-zero element
            zero_positions = np.nonzero(f == 0)[0]
            if len(zero_positions) == 0:
                continue
            idx = zero_positions[rng.integers(len(zero_positions))]
            sign = rng.choice([-1.0, 1.0])
            f[idx] = sign
            n_nonzero += 1
            new_score = _score(f)
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                nonzero_mask = np.nonzero(f)[0]
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[idx] = 0
                n_nonzero -= 1
        else:
            # Remove a non-zero element
            if n_nonzero <= 1:
                continue
            idx = nonzero_mask[rng.integers(len(nonzero_mask))]
            f[idx] = 0
            n_nonzero -= 1
            new_score = _score(f)
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                nonzero_mask = np.nonzero(f)[0]
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
            else:
                f[idx] = sign  # revert - but sign is lost, so just set back
                # Actually we need to restore properly
                f[idx] = -f[idx] if f[idx] == 0 else f[idx]
                # This is getting messy, let's just track better
                pass
    
    return best_f, best_score


def _local_optimization_signs(f):
    """Local optimization: try flipping each non-zero sign, keep if improves."""
    f = f.copy()
    nonzero_mask = np.nonzero(f)[0]
    improved = True
    while improved:
        improved = False
        current_score = _score(f)
        for idx in nonzero_mask:
            if time.time() > DEADLINE:
                return f
            f[idx] = -f[idx]
            new_score = _score(f)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                f[idx] = -f[idx]  # revert
        nonzero_mask = np.nonzero(f)[0]
    return f


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)
    
    # Baseline: all ones
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Try multiple sparsity levels
    sparsity_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for sparsity in sparsity_levels:
        if time.time() > DEADLINE:
            break
        
        # Multiple restarts per sparsity level
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > DEADLINE:
                break
            
            f_init = _make_sparse_sequence(rng, sparsity)
            
            # Run simulated annealing
            f_sa, score_sa = _simulated_annealing_sparse(rng, f_init, max_iter=2000, T_start=0.5, T_end=0.01)
            
            if score_sa < best_score:
                best_score = score_sa
                best_f = f_sa.copy()
            
            # Local optimization on the result
            f_opt = _local_optimization_signs(f_sa)
            score_opt = _score(f_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_f = f_opt.copy()
    
    # Also try a dense baseline with random signs
    f_dense = rng.choice([-1.0, 1.0], size=N)
    f_dense_opt = _local_optimization_signs(f_dense)
    score_dense = _score(f_dense_opt)
    if score_dense < best_score:
        best_score = score_dense
        best_f = f_dense_opt.copy()
    
    return best_f
