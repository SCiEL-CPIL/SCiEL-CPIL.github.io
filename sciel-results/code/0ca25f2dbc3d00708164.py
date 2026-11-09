"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _compute_autocorr(h):
    """Compute full autocorrelation array R[0..N-1] for h of length N."""
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf

def _max_sidelobe(R):
    """Return max absolute value of R[1:]."""
    return np.max(np.abs(R[1:]))

def _compute_autocorr_fast(h):
    """Compute autocorrelation using direct convolution for small N."""
    N = len(h)
    R = np.correlate(h, h, mode='full')
    return R[N-1:2*N-1]

def _flip_delta(h, i):
    """
    Compute the change in autocorrelation R when flipping h[i].
    R[k] = sum_j h[j] * h[j+k] for k >= 0 (one-sided)
    When we flip h[i], the change in R is:
    - For each lag k where h[i] participates:
      * h[i] appears as h[j] when j=i, contributing h[i]*h[i+k]
      * h[i] appears as h[j+k] when j+k=i, i.e., j=i-k, contributing h[i-k]*h[i]
    
    So delta_R[k] = -2*h[i]*(h[i+k] if i+k < N else 0) - 2*h[i]*(h[i-k] if i-k >= 0 else 0)
    But we need to be careful: R is computed as full autocorrelation.
    
    Let's think again. R_full = correlate(h, h, 'full') gives indices from -(N-1) to N-1.
    We use R = R_full[N-1:2*N-1], so R[k] corresponds to lag k (0-indexed, k=0..N-1).
    
    R[k] = sum_{j=0}^{N-1-k} h[j] * h[j+k]
    
    When flipping h[i] (h[i] -> -h[i]):
    R_new[k] - R_old[k] = 
      - If i is in [0, N-1-k] as j: term h[i]*h[i+k] changes by -2*h[i]*h[i+k]
      - If i is in [k, N-1] as j+k: i.e., j=i-k is in [0, N-1-k], term h[i-k]*h[i] changes by -2*h[i-k]*h[i]
    
    So for each k:
    delta[k] = -2*h[i]*h[i+k] (if 0 <= i+k < N) - 2*h[i-k]*h[i] (if 0 <= i-k < N-1-k... wait)
    
    Actually, let me just compute it directly for correctness.
    """
    N = len(h)
    delta = np.zeros(N)
    hi = h[i]
    
    # h[i] contributes to R[k] as the first element (j=i) when k such that i+k < N, i.e., k < N-i
    # Contribution: hi * h[i+k]
    # After flip: -hi * h[i+k]
    # Change: -2*hi*h[i+k]
    for k in range(N - i):
        delta[k] -= 2.0 * hi * h[i + k]
    
    # h[i] contributes to R[k] as the second element (j+k=i, so j=i-k) when j>=0 and j<N-k, i.e., i-k>=0 and i-k<N-k
    # i-k >= 0 means k <= i
    # i-k < N-k means i < N (always true)
    # Contribution: h[i-k] * hi
    # After flip: h[i-k] * (-hi)
    # Change: -2*h[i-k]*hi
    for k in range(i + 1):
        delta[k] -= 2.0 * h[i - k] * hi
    
    return delta

def _generate_h_from_params(params, N, K):
    """
    Generate a signed sequence from sinusoidal parameters.
    params: array of length 2*K. [A1, phi1, A2, phi2, ...]
    x[n] = sum_{k=1}^K A_k * cos(2*pi*k*n/N + phi_k)
    h[n] = sign(x[n])
    """
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(K):
        A = params[2*k]
        phi = params[2*k+1]
        x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
    
    # Handle zeros: if x[n] is 0, assign +1 or -1 arbitrarily (or based on previous)
    # sign(0) is 0 in numpy, so we map 0 to 1
    h = np.sign(x)
    h[h == 0] = 1.0
    return h

def _objective_sin_smooth(params, N, K, temp=1.0):
    """
    Objective function for sinusoidal parameter optimization using soft-sign.
    Returns a smooth surrogate for the max sidelobe (log-sum-exp of squared sidelobes).
    """
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(K):
        A = params[2*k]
        phi = params[2*k+1]
        x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
    
    # Soft sign
    h_soft = np.tanh(x / temp)
    
    R = _compute_autocorr(h_soft)
    
    # Smooth max surrogate: log-sum-exp of squared sidelobes
    sidelobes = R[1:]**2
    max_val = np.max(sidelobes)
    if max_val > 0:
        lse = max_val + np.log(np.sum(np.exp(sidelobes - max_val)))
    else:
        lse = 0.0
    return lse

def _optimize_sinusoidal(N, K, seed, time_limit):
    """
    Optimize sinusoidal parameters using Nelder-Mead.
    """
    np.random.seed(seed)
    # Initial parameters: random amplitudes and phases
    # Amplitudes in [0.5, 1.5], phases in [0, 2*pi]
    A_init = np.random.uniform(0.5, 1.5, size=K)
    phi_init = np.random.uniform(0, 2*np.pi, size=K)
    params_init = np.concatenate([A_init, phi_init])
    
    # Define callback to check time
    start_time = time.time()
    def callback(xk):
        if time.time() - start_time > time_limit:
            return True
        return False

    try:
        res = scipy.optimize.minimize(
            _objective_sin_smooth, 
            params_init, 
            args=(N, K, 1.0), 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=callback
        )
        best_params = res.x
        # Generate hard sign sequence from optimized parameters
        n = np.arange(N)
        x = np.zeros(N)
        for k in range(K):
            A = best_params[2*k]
            phi = best_params[2*k+1]
            x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
        best_h = np.sign(x)
        best_h[best_h == 0] = 1.0
        return best_h
    except Exception:
        return None

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses a hybrid approach:
    1. Coarse optimization using sinusoidal parameterization and Nelder-Mead.
    2. Fine-tuning using greedy local search with flips.
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    def optimize_local(h, R, current_max, time_limit):
        """
        Perform greedy local search starting from h, R, current_max.
        Uses O(N) delta computation per flip.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N):
                if time.time() > time_limit:
                    return h, R, current_max
                
                delta = _flip_delta(h, i)
                new_R = R - delta
                new_max = _max_sidelobe(new_R)
                
                if new_max < current_max - 1e-12:
                    h[i] *= -1.0
                    R = new_R
                    current_max = new_max
                    improved = True
                    
        return h, R, current_max

    # Strategy: Try different numbers of harmonics K
    # K=1 is square wave (poor), K=5, 10, 20
    K_values = [5, 10, 15, 20, 25]
    seed_counter = 0
    
    while time.time() < DEADLINE - SAFETY_MARGIN:
        seed_counter += 1
        K = K_values[seed_counter % len(K_values)]
        
        # Allocate time for coarse optimization
        # L-BFGS-B is fast, so we can use a smaller time limit per restart
        # to allow more restarts and local search time.
        
        coarse_time_limit = 2.0 # seconds
        if time.time() + coarse_time_limit + 5.0 > DEADLINE - SAFETY_MARGIN:
            # Not enough time for a full cycle, break
            break
            
        h_candidate = _optimize_sinusoidal(N, K, seed_counter, coarse_time_limit)
        
        if h_candidate is None:
            h_candidate = np.random.choice([-1.0, 1.0], size=N)
        
        R = _compute_autocorr(h_candidate)
        current_max = _max_sidelobe(R)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h_candidate.copy()
            
        # Local search polish - give it most of the remaining time
        remaining_time = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining_time > 2.0:
            h_candidate, R, current_max = optimize_local(h_candidate, R, current_max, time.time() + remaining_time - 1.0)
            
            if current_max < best_val:
                best_val = current_max
                best_h = h_candidate.copy()
                
    return best_h if best_h is not None else np.ones(N)
