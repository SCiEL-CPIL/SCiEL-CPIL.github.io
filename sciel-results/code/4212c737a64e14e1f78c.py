import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    # Compute max autocorrelation efficiently using FFT
    def max_autocorr(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        return np.max(acf[:N])
    
    # Simulated annealing with flip and swap moves
    def simulated_anneal(h0, max_iters, t0, t1, rng):
        current_h = h0.copy()
        current_val = max_autocorr(current_h)
        best_h = current_h.copy()
        best_val = current_val
        
        for it in range(max_iters):
            if time.time() >= deadline - 15.0:
                break
            
            # Adaptive temperature
            t = t0 * (t1 / t0) ** (it / max_iters)
            
            # Choose move: 70% flip, 30% swap
            if rng.random() < 0.7:
                idx = rng.randint(N)
                trial = current_h.copy()
                trial[idx] = -trial[idx]
            else:
                idx1 = rng.randint(N)
                idx2 = rng.randint(N)
                if idx1 == idx2:
                    continue
                trial = current_h.copy()
                trial[idx1] = -trial[idx1]
                trial[idx2] = -trial[idx2]
            
            trial_val = max_autocorr(trial)
            
            # Accept if better or with probability exp(-delta/T)
            delta = trial_val - current_val
            if delta < 0 or (t > 1e-10 and rng.random() < np.exp(-delta / t)):
                current_h = trial
                current_val = trial_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
        
        return best_h, best_val
    
    # Generate structured initial patterns
    def make_pattern(seed):
        rng = np.random.RandomState(seed)
        # Mix of patterns: random, block-based, alternating
        pattern_type = rng.randint(4)
        if pattern_type == 0:
            return rng.choice([1, -1], N)
        elif pattern_type == 1:
            # Alternating blocks of size 1-4
            h = np.empty(N)
            i = 0
            while i < N:
                block_size = rng.randint(1, 5)
                sign = rng.choice([1, -1])
                h[i:i+block_size] = sign
                i += block_size
            return h
        elif pattern_type == 2:
            # Alternating single bits with some noise
            h = np.tile([1, -1], N // 2 + 1)[:N]
            noise_mask = rng.random(N) < 0.1
            h[noise_mask] = -h[noise_mask]
            return h
        else:
            # Random walk pattern
            h = np.zeros(N)
            h[0] = rng.choice([1, -1])
            for i in range(1, N):
                if rng.random() < 0.8:
                    h[i] = h[i-1]
                else:
                    h[i] = -h[i-1]
            return h
    
    # Try multiple structured starts with simulated annealing
    best_h = None
    best_val = np.inf
    
    for seed in range(30):
        if time.time() >= deadline - 15.0:
            break
        
        h0 = make_pattern(seed)
        rng = np.random.RandomState(seed + 1000)
        
        # Simulated annealing parameters
        max_iters = 50000
        t0 = 5.0
        t1 = 0.01
        
        h, val = simulated_anneal(h0, max_iters, t0, t1, rng)
        
        if val < best_val:
            best_val = val
            best_h = h
    
    return best_h
# EVOLVE-BLOCK-END
