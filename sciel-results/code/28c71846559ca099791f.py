"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3prime(x):
    """Compute max|autoconv| / integral^2 for vector x."""
    n = len(x)
    # Linear autoconvolution via FFT
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
    max_ac = np.max(np.abs(ac))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    return max_ac / (integral * integral)


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2.
    
    Uses gradient descent on continuous real-valued heights initialized with a
    low-frequency sinusoidal profile to distribute autoconvolution energy
    evenly across lags, then projects to signed values.
    """
    t0 = time.time()
    deadline = 145.0
    
    # Initialize with a smooth low-frequency profile
    # This spreads autoconv energy across lags rather than concentrating at lag 0
    x = np.ones(N)
    # Add a gentle sinusoidal perturbation to break symmetry
    x = 1.0 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
    
    # Gradient descent to minimize max|autoconv|/integral^2
    # Use smooth approximation of max: soft-max of |autoconv| values
    lr = 0.01
    best_x = x.copy()
    best_score = _compute_c3prime(x)
    
    for step in range(500):
        if time.time() - t0 > deadline:
            break
        
        n = len(x)
        # Compute autoconvolution via FFT
        f = np.fft.rfft(x, n=2*n)
        ac = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
        
        abs_ac = np.abs(ac)
        integral = np.sum(x)
        
        if abs(integral) < 1e-12:
            break
        
        # Smooth max approximation: use log-sum-exp style
        # For gradient, we need d(max|ac|)/dx
        # Use softmax weighting on |ac| values
        temp = 0.01
        weights = np.exp(abs_ac / temp - np.max(abs_ac) / temp)
        weights /= weights.sum()
        
        # Gradient of autoconv[k] w.r.t. x[i] is x[i-k] + x[i+k] (with bounds)
        # For the weighted sum: grad = sum_k weights[k] * d(ac[k])/dx
        # d(ac[k])/dx[i] = x[i-k] + x[i+k] if in bounds
        # This is equivalent to: grad = weights_convolved + flipped_weights_convolved
        w = weights.copy()
        # Pad weights for convolution
        w_pad = np.zeros(2*n)
        w_pad[:n] = w
        
        # convolve x with w to get gradient contribution
        fw = np.fft.rfft(w_pad, n=2*n)
        grad_ac = np.fft.irfft(fw * np.conj(f), n=2*n)[:n]
        
        # Also need the derivative of 1/integral^2
        # d/dx (max_ac / integral^2) = (grad_max_ac * integral^2 - max_ac * 2*integral * d(integral)/dx) / integral^4
        # = grad_max_ac / integral^2 - 2*max_ac * ones / integral^3
        score_num = np.sum(weights * abs_ac)
        grad = grad_ac / (integral * integral) - 2.0 * score_num * np.ones(N) / (integral ** 3)
        
        # Update
        x_new = x - lr * grad
        x_new = np.clip(x_new, -2.0, 2.0)
        
        new_score = _compute_c3prime(x_new)
        if new_score < best_score:
            best_score = new_score
            best_x = x_new.copy()
        x = x_new
    
    # If gradient descent didn't improve, try a few random restarts with ±1 sequences
    if time.time() - t0 < deadline - 10.0:
        rng = np.random.default_rng(42)
        for trial in range(50):
            if time.time() - t0 > deadline:
                break
            # Random ±1 sequence
            x_rand = rng.choice([-1.0, 1.0], size=N)
            s = _compute_c3prime(x_rand)
            if s < best_score:
                best_score = s
                best_x = x_rand.copy()
    
    # Also consider the all-ones baseline
    baseline = _compute_c3prime(np.ones(N))
    if baseline < best_score:
        best_x = np.ones(N)
        best_score = baseline
    
    return best_x
# EVOLVE-BLOCK-END
