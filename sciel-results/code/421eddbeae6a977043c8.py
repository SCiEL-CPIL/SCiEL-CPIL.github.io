"""Rewritten program: continuous phase optimization for low autoconvolution sidelobe energy."""
import numpy as np
import time

N = 400


def _compute_score(signed):
    """
    Compute the upper bound C3 for a signed step function.
    The score is the maximum absolute value of the autoconvolution
    at non-zero lags (sidelobe peak), which serves as an upper bound
    on C3 in the third autocorrelation inequality context.
    
    For a signed step function h of length N, the autoconvolution is:
      r[k] = sum_i h[i] * h[i+k] for k = -(N-1)..(N-1)
    
    The peak sidelobe is max |r[k]| for k != 0.
    We want to minimize this.
    """
    # Compute full autoconvolution via FFT
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(signed, n=n_fft)
    R = np.fft.irfft(H * np.conj(H), n=n_fft)
    # R has length n_fft; the circular autoconvolution wraps.
    # For linear autoconvolution of length 2N-1, we need to unwrap.
    # Actually, np.fft.irfft of |H|^2 gives circular correlation.
    # For linear: we can use the fact that for zero-padded signals,
    # circular = linear if padding is sufficient.
    # n_fft >= 2N-1 ensures linear autoconvolution is embedded.
    # The linear autoconvolution r[k] for k=0..N-1 is in R[0..N-1]
    # and for k=-(N-1)..-1 is in R[n_fft-N+1..n_fft-1]
    
    # Extract linear autoconvolution
    r_lin = np.empty(2 * N - 1)
    r_lin[0:N] = R[0:N]
    r_lin[N:2*N-1] = R[n_fft - (N - 1):n_fft]
    
    # Peak sidelobe: max |r_lin[k]| for k != 0
    # r_lin[0] is the zero-lag (energy), skip it
    sidelobes = np.abs(r_lin)
    sidelobes[0] = 0
    return np.max(sidelobes)


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    that minimizes the autoconvolution upper bound C3.
    
    Strategy: Represent as complex unit-modulus vector, use continuous phase
    gradient descent (projected gradient on the unit circle), then threshold
    to signs. Use multiple restarts and annealing.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_signed = np.ones(N)
    best_score = _compute_score(best_signed)
    
    # Multiple restarts with different initial phases
    rng = np.random.RandomState(42)
    
    max_restarts = 20
    steps_per_restart = 5000
    lr_base = 0.01
    lr_decay = 0.999
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: random phases on unit circle
        # Or use structured initializations for some restarts
        if restart == 0:
            # Start from all +1 (phase 0)
            phases = np.zeros(N)
        elif restart == 1:
            # Start from alternating
            phases = np.pi * (np.arange(N) % 2)
        elif restart == 2:
            # Start from a known good pattern (e.g., Legendre sequence-like)
            # Use a simple periodic pattern
            period = 7
            pattern = np.ones(period)
            pattern[3] = -1
            pattern[4] = -1
            phases = np.pi * (1 - pattern) * 0.5  # map +/-1 to 0/pi
            phases = np.tile(pattern, N // period + 1)[:N]
            phases = np.where(phases > 0, 0.0, np.pi)
        else:
            # Random phases
            phases = rng.uniform(0, 2 * np.pi, N)
        
        lr = lr_base * (0.5 ** (restart // 5))  # reduce LR for later restarts
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Current complex vector
            z = np.exp(1j * phases)
            
            # Signed version for scoring
            signed = np.sign(z.real)
            signed[signed == 0] = 1.0  # avoid zeros
            
            # Compute gradient of score w.r.t. phases
            # Score = max_k |r[k]| for k != 0
            # This is non-smooth, so use subgradient / heuristic gradient
            
            # Alternative: minimize sum of squared sidelobes (smooth surrogate)
            # This is easier to optimize and correlates with peak sidelobe
            
            # Compute autoconvolution via FFT
            n_fft = 1
            while n_fft < 2 * N - 1:
                n_fft *= 2
            H = np.fft.rfft(z, n=n_fft)
            R = np.fft.irfft(H * np.conj(H), n=n_fft)
            
            # Extract linear autoconvolution
            r_lin = np.empty(2 * N - 1)
            r_lin[0:N] = R[0:N]
            r_lin[N:2*N-1] = R[n_fft - (N - 1):n_fft]
            
            # Objective: sum of squared sidelobes (excluding zero lag)
            # J = sum_{k!=0} r[k]^2
            # Gradient w.r.t. z_i: dJ/dz_i = 2 * sum_{k!=0} r[k] * dr[k]/dz_i
            # dr[k]/dz_i = z[i+k] (for k>=0) or conj(z[i-k]) (for k<0)
            # This is complex; let's use a simpler approach.
            
            # Simpler: use the fact that for unit-modulus z,
            # we can do gradient descent on the real objective.
            
            # Compute gradient of J = sum_{k=1}^{N-1} (r[k]^2 + r[-k]^2)
            # where r[k] = sum_i z[i] * conj(z[i+k])
            
            # Actually, let's just use a heuristic: 
            # The gradient of the sidelobe energy w.r.t. phase[i] can be
            # approximated by the correlation of the sidelobe structure.
            
            # Let's use a different approach: direct gradient of 
            # sum_{k != 0} |r[k]|^2 w.r.t. phases.
            
            # r[k] = sum_i z[i] * conj(z[i+k])
            # |r[k]|^2 = r[k] * conj(r[k])
            # d| r[k] |^2 / d phase[j] = ... this gets messy.
            
            # Simpler heuristic: use the "gradient" as the correlation
            # of the signal with its shifted versions weighted by sidelobes.
            
            # For each i, the contribution to sidelobes from z[i] is:
            # sum_{k!=0} r[k] * conj(z[i+k]) (approximately)
            
            # Let's compute a gradient-like quantity:
            # grad[i] = sum_{k=1}^{N-1} r[k] * conj(z[i+k]) + conj(r[k]) * z[i+k]
            #          = 2 * Re[sum_{k=1}^{N-1} r[k] * conj(z[i+k])]
            
            # This is O(N^2) per step which is too slow for 5000 steps.
            
            # Use FFT-based approach:
            # The gradient of sum_{k!=0} |r[k]|^2 w.r.t. z is related to
            # the autocorrelation of the autocorrelation.
            
            # Let's just use a simpler iterative method: 
            # For each i, try flipping the sign and see if it improves.
            # But that's discrete. 
            
            # Let's use a continuous approach with a simpler gradient:
            # Approximate the gradient of peak sidelobe as the sidelobe
            # correlation at each position.
            
            # Compute: g[i] = sum_{k=1}^{N-1} r[k] * conj(z[i+k])
            # This can be computed via convolution.
            
            # Actually, let's just do discrete sign flips with a continuous
            # annealing schedule. This is simpler and proven to work.
            
            # Pick a random position and flip its sign
            i = rng.randint(N)
            old_signed = signed[i]
            signed[i] = -signed[i]
            
            new_score = _compute_score(signed)
            
            if new_score < best_score:
                best_score = new_score
                best_signed = signed.copy()
            elif new_score < best_score or rng.rand() < np.exp(-(new_score - best_score) / (lr_base * 0.1)):
                # Accept with Metropolis criterion
                pass
            else:
                signed[i] = old_signed
            
            # Occasionally do a swap
            if step % 10 == 0:
                i2 = rng.randint(N)
                if i2 != i:
                    signed[i], signed[i2] = signed[i2], signed[i]
                    new_score = _compute_score(signed)
                    if new_score > best_score and rng.rand() > np.exp(-(new_score - best_score) / (lr_base * 0.1)):
                        signed[i], signed[i2] = signed[i2], signed[i]
        
        # After each restart, update best
        current_score = _compute_score(signed)
        if current_score < best_score:
            best_score = current_score
            best_signed = signed.copy()
    
    return best_signed
