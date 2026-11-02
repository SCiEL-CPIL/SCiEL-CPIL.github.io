import numpy as np
import time

N = 400
DEADLINE = 140.0

def _autocorr_max_abs(x):
    """
    Compute the maximum absolute value of the autocorrelation of x.
    """
    N = len(x)
    size = 1
    while size < 2 * N:
        size *= 2
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)[:N]
    return np.max(np.abs(ac))

def _zadoff_chu_signs(N, m):
    """
    Generate signs from a Zadoff-Chu sequence.
    """
    n = np.arange(N)
    # Phase: pi * m * n * (n+1) / N
    phase = np.pi * m * n * (n + 1) / N
    # Map to signs: sign(cos(phase))
    signs = np.sign(np.cos(phase))
    # Handle zeros (unlikely)
    signs[signs == 0] = 1
    return signs.astype(np.float64)

def _sa_optimize(x_init, time_limit):
    """
    Simulated Annealing with single-bit flips.
    Uses incremental autocorrelation update for speed.
    """
    start_time = time.time()
    x = x_init.copy()
    N = len(x)
    
    # Initial autocorrelation
    # R[k] = sum_{i=0}^{N-1-k} x[i] x[i+k]
    # We store R as a numpy array of length N
    R = np.zeros(N)
    for k in range(N):
        R[k] = np.dot(x[:N-k], x[k:])
    
    current_max = np.max(np.abs(R))
    best_max = current_max
    best_x = x.copy()
    
    T = 10.0  # Initial temperature (scale of R changes)
    T_min = 0.1
    alpha = 0.9995  # Cooling rate
    
    # Pre-compute indices for updates?
    # For flip at i:
    # k ranges from 0 to N-1
    # Delta R[k] = -2 * x[i] * x[i+k] (if i+k < N) - 2 * x[i-k] * x[i] (if i-k >= 0)
    
    while time.time() - start_time < time_limit:
        # Pick random index
        i = np.random.randint(N)
        old_val = x[i]
        
        # Calculate delta for all k
        # Vectorized calculation of delta_R
        # Term 1: x[i] * x[i+k] for k where i+k < N
        # Term 2: x[i-k] * x[i] for k where i-k >= 0
        
        # Create delta_R array
        delta_R = np.zeros(N)
        
        # Term 1: i+k < N => k < N-i
        if N - i > 0:
            # k ranges 0 to N-i-1
            # x[i+k] is x[i : N]
            # delta_R[0:N-i] += -2 * x[i] * x[i:N]
            delta_R[:N-i] += -2.0 * old_val * x[i:]
            
        # Term 2: i-k >= 0 => k <= i
        if i >= 0:
            # k ranges 0 to i
            # x[i-k] is x[i::-1]
            # delta_R[0:i+1] += -2 * x[i::-1] * x[i]
            delta_R[:i+1] += -2.0 * x[i::-1] * old_val
            
        # Note: For k=0, both terms contribute?
        # R[0] = sum x[j]^2 = N.
        # Term 1 at k=0: -2 x[i] x[i] = -2
        # Term 2 at k=0: -2 x[i] x[i] = -2
        # Total delta R[0] = -4.
        # But R[0] should remain N.
        # Let's check: x[i] becomes -x[i].
        # x[i]^2 is still 1.
        # The formula derived: R'(k) = R(k) - 2 x_i x_{i+k} - 2 x_{i-k} x_i.
        # For k=0: R'(0) = R(0) - 2 x_i^2 - 2 x_i^2 = R(0) - 4.
        # This is incorrect. R(0) is invariant.
        # Why? Because the term x_i x_{i+k} at k=0 is x_i^2.
        # When we flip x_i, the term becomes (-x_i)^2 = x_i^2. No change.
        # The derivation assumed the term changes sign.
        # For k=0, the term is x_i^2. It does not change sign.
        # For k != 0, the terms are x_i x_{i+k} and x_{i-k} x_i.
        # When x_i flips, x_i x_{i+k} becomes -x_i x_{i+k}. Change: -2 x_i x_{i+k}.
        # x_{i-k} x_i becomes x_{i-k} (-x_i) = -x_{i-k} x_i. Change: -2 x_{i-k} x_i.
        # So for k != 0, the formula holds.
        # For k=0, delta is 0.
        
        delta_R[0] = 0.0
        
        # Apply delta to R
        R_new = R + delta_R
        
        # Calculate new max
        new_max = np.max(np.abs(R_new))
        
        # Acceptance
        if new_max < current_max:
            # Always accept improvement
            x[i] = -old_val
            R = R_new
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_x = x.copy()
        else:
            # Metropolis
            delta = new_max - current_max
            # Temperature scale: R values are around sqrt(N) ~ 20.
            # delta can be large.
            if np.random.random() < np.exp(-delta / T):
                x[i] = -old_val
                R = R_new
                current_max = new_max
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return best_x, best_max

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    best_x = None
    best_score = float('inf')
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Zadoff-Chu with different m
    for m in [1, 2, 3, 5, 7, 11]:
        if np.gcd(m, N) == 1:
            initializations.append(_zadoff_chu_signs(N, m))
            
    # 2. Random
    initializations.append(np.random.choice([-1.0, 1.0], N))
    
    # 3. Van der Corput (from previous code, might be useful)
    # (Omitted for brevity, assuming ZC is better)
    
    # Time per initialization
    # Total time 140s. 6 initializations -> ~20s each.
    time_per_init = 20.0
    
    for x_init in initializations:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        x_opt, score_opt = _sa_optimize(x_init, time_per_init)
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt
            
    # Final polish on the best result if time remains
    elapsed = time.time() - start_time
    if best_x is not None and elapsed < DEADLINE - 10:
        remaining_time = DEADLINE - 10 - elapsed
        x_final, score_final = _sa_optimize(best_x, remaining_time)
        if score_final < best_score:
            best_score = score_final
            best_x = x_final
            
    if best_x is None:
        # Fallback
        best_x = np.random.choice([-1.0, 1.0], N)
        
    return best_x

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    score = _autocorr_max_abs(result)
    print(f"Result shape: {result.shape}")
    print(f"Min: {np.min(result)}, Max: {np.max(result)}")
    print(f"Score (max |autocorr|): {score}")
    print(f"Time: {elapsed:.2f}s")
