"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_max(h):
    """Compute maximum overlap of h with 1-h over all shifts using FFT."""
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlap = 1.0 - (2.0 / n) * autocorr
    return np.max(overlap)

def _smooth_max(h, beta=200.0):
    """Smooth approximation of max overlap via log-sum-exp."""
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    overlap = 1.0 - (2.0 / n) * autocorr
    # Log-sum-exp for smooth max
    m = np.max(overlap)
    if m < -1e10:
        return m
    return m + np.log(np.sum(np.exp(beta * (overlap - m)))) / beta

def _project(h, target_sum):
    """Project h onto the constraint set: 0<=h_i<=1, sum h = target_sum."""
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - h.sum()
    if abs(diff) > 1e-12:
        # Distribute the difference to elements that can absorb it
        if diff > 0:
            # Need to increase some elements; add to those below 1
            mask = h < 1.0 - 1e-10
            if mask.sum() > 0:
                per = diff / mask.sum()
                h[mask] = np.minimum(h[mask] + per, 1.0)
            # If still not enough, add to all
            diff2 = target_sum - h.sum()
            if diff2 > 1e-12:
                h += diff2 / len(h)
                h = np.clip(h, 0.0, 1.0)
        else:
            # Need to decrease some elements; subtract from those above 0
            mask = h > 1e-10
            if mask.sum() > 0:
                per = -diff / mask.sum()
                h[mask] = np.maximum(h[mask] - per, 0.0)
            diff2 = target_sum - h.sum()
            if diff2 < -1e-12:
                h += diff2 / len(h)
                h = np.clip(h, 0.0, 1.0)
    return h

def _random_initial(N, target_sum, seed=None):
    """Generate a random valid h."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Start with random values and project
    h = rng.uniform(0, 1, N)
    h = _project(h, target_sum)
    return h

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s limit
    target_sum = N / 2.0  # = 50
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Phase 1: Simulated Annealing on the full space
    rng = np.random.RandomState(42)
    
    # Multiple restarts
    n_restarts = 6
    time_per_restart = 18.0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize: mix of random and structured starts
        if restart == 0:
            h = np.full(N, 0.5)
        elif restart == 1:
            # Alternating pattern
            h = np.where(np.arange(N) % 2 == 0, 1.0, 0.0)
            h = _project(h, target_sum)
        elif restart == 2:
            # Block pattern
            h = np.zeros(N)
            h[::2] = 1.0
            h = _project(h, target_sum)
        elif restart == 3:
            # Smooth sinusoidal
            h = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(N) / N)
            h = _project(h, target_sum)
        else:
            h = _random_initial(N, target_sum, seed=42 + restart)
        
        current_score = _overlap_max(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # SA parameters
        T_start = 0.05
        T_end = 0.001
        n_steps = 30000
        step_size = 0.08
        
        t0 = time.time()
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            if time.time() - t0 > time_per_restart:
                break
            
            # Temperature schedule (geometric cooling)
            T = T_start * (T_end / T_start) ** (step / n_steps)
            
            # Pick a random index to perturb
            i = rng.randint(N)
            delta = step_size * (rng.random() * 2 - 1)
            
            h_new = h.copy()
            h_new[i] = np.clip(h_new[i] + delta, 0.0, 1.0)
            h_new = _project(h_new, target_sum)
            
            new_score = _overlap_max(h_new)
            delta_score = new_score - current_score
            
            if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
                h = h_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
        
        # Phase 2: Local coordinate descent polish
        if time.time() > deadline - 5:
            break
        
        # Coordinate descent with small steps
        cd_steps = 5000
        cd_t0 = time.time()
        for cd_step in range(cd_steps):
            if time.time() > deadline - 5:
                break
            if time.time() - cd_t0 > 10:
                break
            
            i = cd_step % N
            # Try small adjustments at position i
            best_local = h
            best_local_score = current_score
            
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                h_try = h.copy()
                h_try[i] = np.clip(h_try[i] + delta, 0.0, 1.0)
                h_try = _project(h_try, target_sum)
                s_try = _overlap_max(h_try)
                if s_try < best_local_score:
                    best_local_score = s_try
                    best_local = h_try.copy()
            
            if best_local_score < current_score:
                h = best_local
                current_score = best_local_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
    
    # Final: ensure valid output
    best_h = _project(best_h, target_sum)
    return best_h
# EVOLVE-BLOCK-END
