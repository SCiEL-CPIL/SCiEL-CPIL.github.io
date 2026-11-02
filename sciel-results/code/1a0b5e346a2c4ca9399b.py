"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _score_candidate(x):
    """Compute the objective: max |autoconv| / (sum x)^2 for a linear autocorrelation."""
    S = x.sum()
    if abs(S) < 1e-12:
        return np.inf
    # Linear autocorrelation: c_k = sum_{i} x_i x_{i+k} for k in [-N+1, N-1]
    # Use FFT for speed.
    n_fft = 1 << (2 * N - 1).bit_length()
    fx = np.fft.rfft(x, n_fft)
    ac = np.fft.irfft(fx * np.conj(fx), n_fft)[:N]
    # ac[k] = sum_{i=0}^{N-1-k} x_i x_{i+k} for k=0..N-1
    # We need max over all k (positive and negative). By symmetry |ac[k]| = |ac[-k]| for linear?
    # Actually ac_full[k] = sum x_i x_{i+k}. ac_full[-k] = sum x_i x_{i-k} = sum x_j x_{j+k} (sub j=i-k)
    # So yes, the magnitude is symmetric. So max |ac| = max |ac[0..N-1]|.
    # Wait, is that true?
    # c_k = sum_{i} x_i x_{i+k}
    # c_{-k} = sum_{i} x_i x_{i-k}. Let j = i-k, then sum_j x_{j+k} x_j. Same terms, same value.
    # So c_{-k} = c_k. It is symmetric!
    # Therefore, we only need to check k=0..N-1.
    max_abs = np.max(np.abs(ac))
    return max_abs / (S * S)


def _try_mutation(x, rng):
    """Attempt a random perturbation and return the improved candidate if any."""
    for _ in range(5):
        # Randomly flip a small number of signs
        n_flip = rng.integers(1, max(2, N // 20))
        idx = rng.choice(N, size=n_flip, replace=False)
        x_new = x.copy()
        x_new[idx] *= -1
        s_new = _score_candidate(x_new)
        if s_new < 1e-3:
            return x_new, s_new
    return None, None


def _refine_greedy(x, current_score, time_budget):
    """Greedy coordinate descent: flip individual signs if it improves the score."""
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() > time_budget:
                break
            x[i] *= -1
            s_new = _score_candidate(x)
            if s_new < current_score - 1e-12:
                current_score = s_new
                improved = True
            else:
                x[i] *= -1  # revert
        if time.time() > time_budget:
            break
    return x, current_score


def construct():
    t_start = time.time()
    deadline = t_start + 120.0
    
    # Seed: all ones
    x = np.ones(N)
    best_score = _score_candidate(x)
    best_x = x.copy()
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Random restarts with greedy refinement
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        # Start from a random sign pattern
        x_try = rng.choice([-1.0, 1.0], size=N)
        s_try = _score_candidate(x_try)
        
        # Greedy refine
        x_ref, s_ref = _refine_greedy(x_try.copy(), s_try, deadline - 2.0)
        
        if s_ref < best_score - 1e-12:
            best_score = s_ref
            best_x = x_ref.copy()
            print(f"Restart {n_restarts}: improved to {best_score:.6f}")
            
        # Also try perturbation from current best
        x_pert, s_pert = _try_mutation(best_x, rng)
        if s_pert is not None and s_pert < best_score - 1e-12:
            best_score = s_pert
            best_x = x_pert.copy()
            x_ref, s_ref = _refine_greedy(x_pert.copy(), s_pert, deadline - 2.0)
            if s_ref < best_score - 1e-12:
                best_score = s_ref
                best_x = x_ref.copy()
                print(f"Perturb {n_restarts}: improved to {best_score:.6f}")
    
    return best_x
# EVOLVE-BLOCK-END
