"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autocorr_sidelobe_max(x):
    """Compute max absolute sidelobe of autocorrelation (excluding lag 0)."""
    # Use FFT for fast autocorrelation
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    # acf[0] is the peak (lag 0), we want max |acf[k]| for k != 0
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0


def _objective(x):
    """The raw value to minimize: upper bound on C3 via autoconvolution sidelobe."""
    return _autocorr_sidelobe_max(x)


def _logistic_init(seed):
    """Generate initial signed step function using logistic map."""
    rng = np.random.RandomState(seed)
    r = 3.99  # chaotic regime
    x0 = rng.uniform(0.1, 0.9)
    seq = np.zeros(N)
    x = x0
    for i in range(N):
        x = r * x * (1 - x)
        # Map to signed: threshold at 0.5
        seq[i] = 1.0 if x > 0.5 else -1.0
    return seq


def _van_der_corput_init():
    """Generate initial signed step function using van der Corput sequence."""
    def vdc(n, base=2):
        """van der Corput sequence value at index n."""
        result = 0.0
        f = 1.0 / base
        while n > 0:
            result += f * (n % base)
            n //= base
            f /= base
        return result
    
    seq = np.zeros(N)
    for i in range(N):
        val = vdc(i)
        # Map to signed based on parity or threshold
        seq[i] = 1.0 if (i % 2 == 0) else -1.0
        # Actually use the vdc value to determine sign
        seq[i] = 1.0 if val > 0.5 else -1.0
    return seq


def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Try multiple initial designs
    candidates = []
    
    # Logistic map init
    for seed in [42, 123, 7, 999]:
        x = _logistic_init(seed)
        candidates.append(x)
    
    # Van der Corput init
    candidates.append(_van_der_corput_init())
    
    # Random init
    rng = np.random.RandomState(0)
    candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates
    best_x = candidates[0]
    best_val = _objective(best_x)
    
    for x in candidates[1:]:
        val = _objective(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Simulated Annealing refinement
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_x
    
    x = best_x.copy()
    current_val = best_val
    T = 1.0  # initial temperature
    T_min = 1e-8
    cooling_rate = 0.9995
    
    # Precompute FFT size
    fft_n = 2 * N
    
    rng = np.random.RandomState(12345)
    
    while time.time() - start_time < deadline:
        # Pick a random index to flip
        idx = rng.randint(N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_val = _objective(x)
        delta = new_val - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()
        else:
            x[idx] = old_val  # revert
        
        T *= cooling_rate
    
    return best_x

# EVOLVE-BLOCK-END
