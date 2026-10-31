"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(x):
    """Compute max absolute value of linear autoconvolution."""
    n = len(x)
    # Linear autoconvolution via FFT
    size = 2 * n
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    # The linear autoconvolution is in ac[0:n] and ac[n:2n-1]
    # Actually for linear convolution of x with x:
    # ac[k] = sum_i x[i] x[i+k] for k=0..n-1
    # and ac[k] = sum_i x[i] x[i-k] for k=n..2n-2
    # Due to symmetry, we only need to check one side, but let's be safe
    # Actually, autoconvolution here likely means correlation or convolution
    # "autoconvolution" usually means convolution of a signal with itself
    # Let's use the full linear convolution
    return np.max(np.abs(ac[:2*n-1]))

def _score(x):
    """Compute the score: max_abs_autoconv / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _legendre_sequence(n):
    """Generate a Legendre sequence of length n (requires n prime)."""
    def legendre_symbol(a, p):
        ls = pow(a, (p - 1) // 2, p)
        return -1 if ls == p - 1 else ls
    
    seq = np.ones(n)
    for i in range(1, n):
        seq[i] = legendre_symbol(i, n)
    return seq

def _find_next_prime(n):
    """Find the next prime >= n."""
    def is_prime(num):
        if num < 2:
            return False
        if num % 2 == 0:
            return num == 2
        for i in range(3, int(num**0.5) + 1, 2):
            if num % i == 0:
                return False
        return True
    
    p = n if n >= 2 else 2
    while not is_prime(p):
        p += 1
    return p

def _local_sign_flip(x, max_iters=5000, time_limit=10.0):
    """Local optimization by flipping signs to minimize score."""
    start_time = time.time()
    current_score = _score(x)
    x = x.copy()
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        best_improvement = 0
        best_idx = -1
        
        # Randomly sample indices to flip to speed up
        sample_size = min(100, len(x))
        indices = np.random.choice(len(x), size=sample_size, replace=False)
        
        for i in indices:
            x[i] *= -1
            new_score = _score(x)
            if new_score < current_score - best_improvement:
                best_improvement = current_score - new_score
                best_idx = i
            x[i] *= -1
        
        if best_idx != -1:
            x[best_idx] *= -1
            current_score -= best_improvement
        else:
            # No improvement found in this round, break or continue with fewer samples
            break
            
    return x, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with Legendre sequence of length 401 (prime), truncate to 400
    p = _find_next_prime(N)
    legendre = _legendre_sequence(p)
    x = legendre[:N].astype(float)
    
    best_x = x.copy()
    best_score = _score(x)
    
    # Local optimization
    x, score = _local_sign_flip(x, max_iters=10000, time_limit=20.0)
    if score < best_score:
        best_x = x
        best_score = score
    
    # Try multiple random restarts if time permits
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Random perturbation of the best solution
        x = best_x.copy()
        num_flips = np.random.randint(1, 10)
        flip_indices = np.random.choice(N, size=num_flips, replace=False)
        x[flip_indices] *= -1
        
        x, score = _local_sign_flip(x, max_iters=5000, time_limit=5.0)
        if score < best_score:
            best_x = x
            best_score = score
        else:
            # If no improvement after a few restarts, reduce effort
            if restart_count > 5:
                break
    
    return best_x
# EVOLVE-BLOCK-END
