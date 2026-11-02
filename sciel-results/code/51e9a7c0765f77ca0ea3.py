import numpy as np
import time

N = 400
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _autocorr_max_abs(s):
    """
    Compute the maximum absolute value of the non-zero lag autocorrelation.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    return np.max(np.abs(r[1:N]))

def _autocorr_soft_max(s, beta=10.0):
    """
    Compute a smooth approximation of the max absolute autocorrelation sidelobe.
    Uses log-sum-exp trick for numerical stability.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    abs_r = np.abs(r[1:N])
    # Soft max: (1/beta) * log(sum(exp(beta * x)))
    # For numerical stability, subtract max first
    max_val = np.max(abs_r)
    if max_val == 0:
        return 0.0
    shifted = beta * (abs_r - max_val)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_val
    return log_sum / beta

def _gradient_soft_max(s, beta=10.0):
    """
    Compute the gradient of the soft-max approximation of the peak autocorrelation.
    
    The soft-max of |R(k)| is: (1/beta) * log(sum_k exp(beta * |R(k)|))
    
    For gradient, we need d/dx_i of soft_max(|R(k)|).
    
    R(k) = sum_n s(n+k) * s(n) for linear autocorrelation (zero-padded FFT)
    
    Using the FFT formulation:
    Let F = FFT(s, n=2N), then R_circ = IFFT(|F|^2)
    
    For soft-max, the gradient involves weights w_k = exp(beta*|R(k)|) / sum_j exp(beta*|R(j)|)
    
    This is complex to derive directly. Instead, we use a simpler approach:
    minimize a combination of L2 and peak via surrogate.
    
    Alternative: use the fact that for binary sequences, we can optimize 
    the continuous relaxation with a smooth penalty.
    """
    n_fft = 2 * N
    F = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(np.abs(F)**2).real
    abs_r = np.abs(r[1:N])
    
    # Compute weights for soft-max
    max_val = np.max(abs_r)
    if max_val == 0:
        return np.zeros(N)
    shifted = beta * (abs_r - max_val)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of soft-max w.r.t. r[1:N]
    # d(soft_max)/d(r[k]) = weights[k] * sign(r[k])
    
    # Now we need d(r[k])/d(s[i]) for each i
    # r[k] = sum_n s[n+k]*s[n] (linear autocorrelation)
    # This is equivalent to circular autocorrelation of zero-padded signal
    
    # Using FFT: r = IFFT(|F|^2)
    # dr/ds = 2 * Re(IFFT(conj(F) * dF/ds)) ... this gets complex
    
    # Simpler: use the L2 gradient as a proxy, weighted by importance
    # The L2 gradient is: 4 * IFFT(|F|^2 * F).real[:N]
    grad_l2 = 4.0 * (np.abs(F)**2) * F
    grad_l2 = np.fft.ifft(grad_l2).real[:N]
    
    # For the soft-max, we want to prioritize reducing the largest sidelobes
    # A simple approach: use the L2 gradient but scale by how "peaked" the spectrum is
    # Or just use L2 gradient which is computationally cheap
    
    return grad_l2

def _simulated_annealing(s, rng, start_temp=1.0, end_temp=0.01, n_iter=5000, 
                          block_min=4, block_max=32, deadline=None):
    """
    Run simulated annealing with block-flip moves on the binary sequence.
    """
    current_s = s.copy()
    current_val = _autocorr_max_abs(current_s)
    best_s = current_s.copy()
    best_val = current_val
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Linear cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose block size
        block_size = int(rng.integers(block_min, block_max + 1))
        start = int(rng.integers(0, N - block_size + 1))
        
        # Flip the block
        old_block = current_s[start:start+block_size].copy()
        current_s[start:start+block_size] = -old_block
        
        new_val = _autocorr_max_abs(current_s)
        
        # Accept or reject
        delta = new_val - current_val
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / temp)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = current_s.copy()
        else:
            current_s[start:start+block_size] = old_block
    
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    best_s = None
    best_val = np.inf
    
    # Phase 1: Gradient descent with smooth surrogate, then project to binary
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Initialize continuous signal
        x = rng.uniform(-1, 1, size=N)
        
        # Gradient Descent Phase using L2 surrogate (computationally efficient)
        lr = 0.1
        for step in range(150):
            if time.time() > deadline:
                break
                
            F = np.fft.fft(x, n=2*N)
            grad = 4.0 * (np.abs(F)**2) * F
            grad = np.fft.ifft(grad).real[:N]
            x -= lr * grad
            
            # Normalize
            norm = np.linalg.norm(x)
            if norm > 0:
                x = x / norm * np.sqrt(N)
        
        # Project to binary
        s = np.sign(x)
        s[s == 0] = 1.0
        
        cur_val = _autocorr_max_abs(s)
        
        if cur_val < best_val:
            best_val = cur_val
            best_s = s.copy()
    
    if best_s is None:
        best_s = np.ones(N)
        best_val = _autocorr_max_abs(best_s)
    
    # Phase 2: Simulated annealing with block flips (main optimization)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run SA for the remaining time
        n_iter = int(remaining_time * 200)  # Estimate ~200 iterations per second
        best_s, best_val = _simulated_annealing(
            best_s, rng, start_temp=1.0, end_temp=0.001, 
            n_iter=n_iter, block_min=4, block_max=32, deadline=deadline
        )
    
    # Phase 3: Final polish with single-bit flips (if time remains)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        cur_val = _autocorr_max_abs(best_s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = best_s[i]
                best_s[i] = -old_val
                new_val = _autocorr_max_abs(best_s)
                if new_val < cur_val:
                    cur_val = new_val
                    if new_val < best_val:
                        best_val = new_val
                    improved = True
                else:
                    best_s[i] = old_val
    
    return best_s
