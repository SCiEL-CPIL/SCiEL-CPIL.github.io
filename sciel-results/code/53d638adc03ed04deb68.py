"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _compute_ac(s):
    """Compute full autoconvolution."""
    return fftconvolve(s, s, mode='full')

def _max_ac_after_flip(ac, s, idx):
    """
    Compute max|ac| after flipping s[idx] -> -s[idx], given current ac.
    For binary s (values ±1): flipping idx changes ac[k] by -2*s[idx]*s[idx+k]
    for all valid k. Since s[idx] = ±1, the change is ∓2*s shifted.
    
    ac[k] = sum_{i} s[i]*s[i+k] for k in [-(N-1), N-1]
    After flipping s[idx]: ac_new[k] = ac[k] - 2*s[idx]*s[idx+k]
    
    We need to handle the index mapping carefully.
    """
    n = len(s)
    val = s[idx]
    # The autoconvolution ac has length 2*n-1, with ac[k+n-1] = sum_i s[i]*s[i+k]
    # Flipping s[idx] subtracts 2*val*s[idx+k] from ac[k+n-1] for all valid k
    # i.e., ac_new[k+n-1] = ac[k+n-1] - 2*val*s[idx+k]
    
    # Build the delta: for lag k, delta[k] = -2*val*s[idx+k]
    # k ranges from -(N-1) to (N-1)
    # s[idx+k] is valid when 0 <= idx+k < N, i.e., -idx <= k <= N-1-idx
    
    # Create full delta array of length 2*n-1
    delta = np.zeros(2 * n - 1)
    
    # For each valid k, delta[k + n - 1] = -2 * val * s[idx + k]
    # k ranges from max(-idx, -(n-1)) to min(n-1-idx, n-1)
    k_min = max(-idx, -(n - 1))
    k_max = min(n - 1 - idx, n - 1)
    
    if k_min <= k_max:
        # s indices: idx + k for k in [k_min, k_max]
        s_indices = np.arange(idx + k_min, idx + k_max + 1)
        k_values = np.arange(k_min, k_max + 1)
        delta[k_values + n - 1] = -2.0 * val * s[s_indices]
    
    ac_new = ac + delta
    return np.max(np.abs(ac_new))

def _soft_max_ac(s, temperature=1.0):
    """
    Differentiable surrogate for max |autoconvolution|.
    """
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    lse = max_val + np.log(np.sum(np.exp(ac_sq - max_val)))
    return np.sqrt(lse)

def _soft_objective(x, temperature=1.0):
    """
    Soft objective: use log-sum-exp surrogate for max |ac|.
    """
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    soft_max_ac = _soft_max_ac(s, temperature)
    return soft_max_ac / sq_int

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = np.inf
    best_s = None
    
    np.random.seed(42)
    
    # Phase 1: Gradient-based initialization with L-BFGS-B
    # Use fewer iterations but with tighter settings, then binary polish
    n_inits = 15
    for init_iter in range(n_inits):
        if time.time() > deadline:
            break
        
        x0 = np.random.randn(N) * 0.5
        
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(0.5,),
                options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            s_continuous = np.tanh(result.x)
            
            # Binary refinement
            s_binary = np.sign(s_continuous)
            s_binary[s_binary == 0] = 1.0
            
            score_bin = _score_sequence(s_binary)
            if score_bin < best_score:
                best_score = score_bin
                best_s = s_binary.copy()
                
            # Also evaluate continuous
            score_cont = _score_sequence(s_continuous)
            if score_cont < best_score:
                best_score = score_cont
                best_s = s_continuous.copy()
                
        except Exception:
            continue
    
    # If nothing found, use random binary
    if best_s is None:
        best_s = np.random.choice([-1.0, 1.0], size=N)
        best_score = _score_sequence(best_s)
    
    # Phase 2: Fast local search with vectorized flip evaluation
    # For binary sequences, we can compute max|ac| after a flip in O(N)
    current_s = best_s.copy()
    current_score = best_score
    current_ac = _compute_ac(current_s)
    
    # Greedy hill climbing with random restarts
    max_local_iters = 20000
    for local_round in range(5):
        if time.time() > deadline:
            break
        
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # Try all flips, pick the best improvement
            best_delta = 0
            best_idx = -1
            
            # Vectorized: compute max|ac| after each possible flip
            # For efficiency, batch evaluate in chunks
            chunk_size = N  # evaluate all N flips at once
            
            # For each index j, compute the resulting max|ac|
            # delta_j[k] = -2*s[j]*s[j+k] for valid k
            # This is equivalent to: ac_new_j = ac - 2*s[j]*shifted_s_j
            # where shifted_s_j is s shifted so that s[j+k] aligns with lag k
            
            # Build the shift matrix approach:
            # For flip at j, the change to ac is -2*s[j] * (s convolved with delta at j)
            # Actually: ac_new[k] = ac[k] - 2*s[j]*s[j+k]
            # So ac_new = ac - 2*s[j] * S_j where S_j[k] = s[j+k] (valid k)
            
            # We can compute all N flip results in one vectorized operation
            # by building a matrix of all shifted versions
            
            # For memory efficiency with N=400, build the full matrix
            # shifts[j, k] = s[j+k] for valid (j,k), 0 otherwise
            # This is a circulant-like matrix
            
            # Actually, let's just do it in a loop but vectorized per-flip
            # Each flip evaluation is O(N) numpy ops, which is fast
            
            for idx in range(N):
                if time.time() > deadline:
                    break
                
                new_max_ac = _max_ac_after_flip(current_ac, current_s, idx)
                new_sum = np.sum(current_s) - 2.0 * current_s[idx]
                new_sq_int = new_sum ** 2
                if new_sq_int < 1e-10:
                    continue
                new_score = new_max_ac / new_sq_int
                
                if new_score < current_score - 1e-12:
                    # Accept this flip
                    current_s[idx] = -current_s[idx]
                    current_ac = _compute_ac(current_s)  # recompute for accuracy
                    current_score = new_score
                    improved = True
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_s = current_s.copy()
                    break  # restart the full scan after accepting a flip
        
        # After local optimum, try a random restart perturbation
        if local_round < 4:
            # Flip a few random positions to escape local optimum
            n_flips = np.random.randint(3, 8)
            flip_indices = np.random.choice(N, size=n_flips, replace=False)
            for idx in flip_indices:
                current_s[idx] = -current_s[idx]
            current_ac = _compute_ac(current_s)
            current_score = _score_sequence(current_s)
            if current_score < best_score:
                best_score = current_score
                best_s = current_s.copy()
    
    # Phase 3: Final polish - try all single flips one more time
    if time.time() < deadline - 5:
        current_s = best_s.copy()
        current_ac = _compute_ac(current_s)
        current_score = best_score
        
        for _ in range(50):
            if time.time() > deadline:
                break
            made_progress = False
            for idx in range(N):
                new_max_ac = _max_ac_after_flip(current_ac, current_s, idx)
                new_sum = np.sum(current_s) - 2.0 * current_s[idx]
                new_sq_int = new_sum ** 2
                if new_sq_int < 1e-10:
                    continue
                new_score = new_max_ac / new_sq_int
                if new_score < current_score - 1e-12:
                    current_s[idx] = -current_s[idx]
                    current_ac = _compute_ac(current_s)
                    current_score = new_score
                    made_progress = True
                    if new_score < best_score:
                        best_score = new_score
                        best_s = current_s.copy()
                    break
            if not made_progress:
                break
    
    if best_s is None:
        return np.ones(N)
    
    return best_s
# EVOLVE-BLOCK-END
