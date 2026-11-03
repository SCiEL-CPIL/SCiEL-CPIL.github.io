import numpy as np
import time

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute frequency basis for local search vectorization
# B[i, k] = exp(-2j * pi * i * k / FSIZE)
freqs = np.arange(_FSIZE // 2 + 1)
B = np.exp(-2j * np.pi * np.arange(N)[:, None] * freqs[None, :] / _FSIZE)

def _score(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _vectorized_flip_scores(h):
    """
    Computes the score for all single-bit flips of h efficiently.
    Returns an array of scores for each flip.
    """
    H = np.fft.rfft(h, _FSIZE)
    
    # For each i, new H is H - 2*h[i]*B[i]
    # H' (N x F/2+1)
    # delta = 2 * h[:, None] * B
    # H_prime = H[None, :] - delta
    
    h_col = h[:, None]
    delta = 2 * h_col * B
    H_prime = H[None, :] - delta
    
    # Square in frequency domain
    AC_freq = H_prime * H_prime
    
    # Inverse FFT to get time domain autoconvolution
    # Shape: (N, FSIZE)
    AC_time = np.fft.irfft(AC_freq, _FSIZE, axis=1)
    
    # We only care about the first 2*N - 1 lags
    AC_relevant = AC_time[:, :2*N-1]
    
    # Max absolute value for each flip
    max_abs_ac = np.max(np.abs(AC_relevant), axis=1)
    
    # Update integral
    # Sum(h) - 2*h[i]
    s0 = np.sum(h)
    s_new = s0 - 2 * h
    
    # Avoid division by zero
    valid = np.abs(s_new) > 1e-12
    scores = np.full(N, float('inf'))
    scores[valid] = max_abs_ac[valid] / (s_new[valid] ** 2)
    
    return scores

def _construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    best_x = None
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    # 1. Generate initial candidates
    candidates = []
    
    # All ones
    candidates.append(np.ones(N))
    
    # Random
    for _ in range(20):
        candidates.append(rng.choice([1.0, -1.0], N))
        
    # Structured: Chirps
    t = np.arange(N)
    for freq in [1.0, 2.0, 0.5, 3.0, 1.5]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
        
    # Structured: Linear
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)

    # Evaluate and pick best few
    scored_candidates = []
    for x in candidates:
        val = _score(x)
        scored_candidates.append((val, x))
        
    scored_candidates.sort(key=lambda t: t[0])
    
    best_val = scored_candidates[0][0]
    best_x = scored_candidates[0][1].copy()
    
    # 2. Vectorized Local Search / Simulated Annealing
    # We will perform multiple passes of vectorized flip search.
    # In each pass, we calculate the score of flipping *each* bit individually.
    # We then choose the best flip (or accept a worse flip with probability T) to update the state.
    # This is effectively a very fast greedy descent or SA step.
    
    # To speed up convergence, we can do multiple "steps" per evaluation?
    # No, the state changes after each flip, so the scores for other flips change.
    # However, if we just pick the globally best flip among all N, that's a valid greedy step.
    # Since we compute all N scores in one vectorized op, this is very fast.
    
    # Let's do a loop of Greedy Descent with occasional restarts if stuck.
    
    current_x = best_x.copy()
    current_val = best_val
    
    # Greedy Descent Phase
    # We keep flipping the best bit until no improvement is found.
    
    max_iter = 2000 # Safety cap
    stagnation_count = 0
    
    for iter in range(max_iter):
        if time.time() > deadline - 10:
            break
            
        scores = _vectorized_flip_scores(current_x)
        
        min_score = np.min(scores)
        min_idx = np.argmin(scores)
        
        if min_score < current_val - 1e-12:
            # Accept the flip
            current_x[min_idx] *= -1
            current_val = min_score
            stagnation_count = 0
            
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        else:
            # No single bit flip improves the score
            stagnation_count += 1
            
            # If stuck, try a random jump or break to restart
            if stagnation_count > 5:
                # Try a random block flip to escape local minimum
                # Pick a random block of size 3-5 and flip it
                bs = rng.integers(3, 6)
                start = rng.integers(0, N - bs + 1)
                current_x[start:start+bs] *= -1
                new_val = _score(current_x)
                
                if new_val < current_val:
                    current_val = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_x = current_x.copy()
                else:
                    # Revert
                    current_x[start:start+bs] *= -1
                    # If we revert, we are still stuck. Break to try a new random start?
                    # Or just break this descent chain.
                    break
                    
            elif stagnation_count > 2:
                 # Try a random single bit flip to see if we can wiggle out (unlikely to help if greedy is stuck)
                 pass
                 
    # 3. Final Polish
    # Run vectorized flip search again to ensure we are at a local minimum
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        scores = _vectorized_flip_scores(best_x)
        min_score = np.min(scores)
        min_idx = np.argmin(scores)
        
        if min_score < best_val - 1e-12:
            best_x[min_idx] *= -1
            best_val = min_score
        else:
            break

    return best_x

def construct():
    return _construct()

if __name__ == "__main__":
    x = construct()
    val = _score(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
