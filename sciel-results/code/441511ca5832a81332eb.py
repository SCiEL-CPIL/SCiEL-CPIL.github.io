import numpy as np
import time
import math

N = 400
# Precompute FFT size
n_fft = 1
while n_fft < 2 * N:
    n_fft *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _acf_delta_flip(s, i):
    """
    Exact delta in the full autocorrelation A[k] = sum_j s[j]*s[j+k]
    when we flip s[i] -> -s[i].  Returns a length-N array D such that
    A_new[k] = A_old[k] + D[k].
    
    Derivation:
      A[k] = sum_j s[j] s[j+k]
      After flipping index i, terms that include s[i] change sign.
      A_new[k] = A_old[k] - 2 * s[i] * (sum over j where {j, j+k} contains i of s[partner])
      The partner index is j = i (when j+k = i, i.e., j = i-k) or j+k = i (j = i-k)... 
      More carefully: term s[j]*s[j+k] is affected iff j == i or j+k == i.
        - if j == i: term is s[i]*s[i+k], partner = i+k (if in [0,N))
        - if j+k == i: term is s[i-k]*s[i], partner = i-k (if in [0,N))
      So delta A[k] = -2 * s[i] * (s[i+k] * [0<=i+k<N] + s[i-k] * [0<=i-k<N])
    """
    D = np.zeros(N, dtype=np.float64)
    si = s[i]
    # For each lag k in 0..N-1:
    #   partner1 = i + k, valid if i+k < N
    #   partner2 = i - k, valid if i-k >= 0
    k = np.arange(N)
    p1 = i + k
    p2 = i - k
    valid1 = p1 < N
    valid2 = p2 >= 0
    # s[p1] where valid
    contrib = np.zeros(N, dtype=np.float64)
    contrib[valid1] = s[p1[valid1]]
    contrib[valid2] += s[p2[valid2]]
    D = -2.0 * si * contrib
    return D

def _local_search(s, current_val, max_passes=50, time_limit=10.0, deadline=None):
    """Local search by flipping bits with exact O(N) delta updates."""
    start = time.time()
    improved = True
    passes = 0
    
    A = _calc_acf(s)
    # track current max-abs over lags 1..N-1
    vals = np.abs(A[1:])
    current_val = float(np.max(vals))
    
    while improved and passes < max_passes and time.time() - start < time_limit:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
            
            D = _acf_delta_flip(s, i)
            A_new = A + D
            vals_new = np.abs(A_new[1:])
            val_new = float(np.max(vals_new))
            
            if val_new < current_val - 1e-12:
                # accept
                A = A_new
                s[i] = -s[i]
                current_val = val_new
                improved = True
            # else reject; s unchanged, A unchanged
    
    return s, current_val

def _ising_cooling_init(n_iter=500, deadline=None):
    """
    Spin-glass cooling schedule for a frustrated Ising antiferromagnet.
    The energy function penalizes short-range autocorrelation peaks (L2 norm)
    and implicitly encourages long-range sign clustering to minimize L-infinity norm.
    """
    # Start with random continuous values
    x = np.random.uniform(-1, 1, size=N)
    
    # Cooling schedule
    T_start = 0.1
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute autocorrelation
        A = _calc_acf(x)
        
        # Energy: L2 norm of sidelobes (smooth surrogate for L-infinity)
        # We use the gradient of sum(A[k]^2)
        A_for_grad = np.zeros(N)
        A_for_grad[1:] = A[1:]
        
        # Gradient via FFT
        Fx = np.fft.rfft(x, n=n_fft)
        FA = np.fft.rfft(A_for_grad, n=n_fft)
        grad = np.fft.irfft(4 * Fx * FA, n=n_fft)[:N]
        
        # Metropolis acceptance with noise
        # Proposed move: x_new = x - 0.01 * grad + noise
        noise = np.random.normal(0, T * 0.1, size=N)
        x_new = x - 0.01 * grad + noise
        x_new = np.clip(x_new, -1, 1)
        
        # Compute energy change (approximate)
        # For speed, we just accept if T is high, or if it's a clear improvement
        # A rigorous Metropolis would require exact energy calculation, which is expensive.
        # We use a heuristic: accept with probability 1 if T > 0.05, else only if improved.
        # Actually, let's just do gradient descent with noise (simulated annealing on continuous variables)
        
        # To be more efficient, we can just update x directly
        x = x_new
        T *= cooling_rate
        
    # Threshold to binary
    s = np.where(x >= 0, 1.0, -1.0)
    return s

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin well under 150 s kill
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Strategy 1: Ising cooling initialization (multiple times)
    for init_iter in range(20):
        if time.time() > deadline - 8:
            break
        
        s = _ising_cooling_init(n_iter=300, deadline=deadline - 4)
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=40, time_limit=6.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    # Strategy 2: Random restarts with refinement
    while time.time() < deadline - 4:
        s = np.random.randint(0, 2, size=N) * 2 - 1
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search refinement
        s, current_val = _local_search(s, current_val, max_passes=30, time_limit=4.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
