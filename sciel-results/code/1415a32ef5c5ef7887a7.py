"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution

N = 400

# EVOLVE-BLOCK-START

def _autoconv_peak(x):
    """Compute max absolute autoconvolution divided by squared integral."""
    # Integral is sum of x (since step width is 1)
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.inf
    # Autoconvolution via FFT for efficiency
    # For autoconvolution: convolve x with reversed x
    # Use FFT: ifft(fft(x) * fft(reversed(x)))
    # Actually, linear autoconvolution of x with x (not reversed) is what we want?
    # The task says "autoconvolution". In signal processing, autoconvolution is x * x.
    # Let's compute linear convolution of x with x.
    # Using FFT: size needs to be at least 2*N-1
    fft_size = 2 * N
    X = np.fft.fft(x, n=fft_size)
    # Autoconvolution is ifft(X * X)
    auto = np.fft.ifft(X * X).real
    # The autoconvolution has length 2*N-1, but we only care about the valid part?
    # Usually "maximum absolute autoconvolution" refers to the peak of the full convolution.
    # However, for "signed equal-width step functions", the standard C3 problem usually looks at the convolution.
    # Let's take the max absolute value of the full linear autoconvolution.
    max_abs = np.max(np.abs(auto))
    return max_abs / (integral * integral)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_val = np.inf
    best_x = np.ones(N)  # Fallback: all ones
    
    # Strategy:
    # 1. Start with a random binary sequence (+1/-1) or a structured sequence.
    # 2. Use a local optimization approach on the signs? No, signs are discrete.
    # 3. Relax to continuous variables in [-1, 1] or R, optimize, then round?
    #    The task asks for "signed step heights". It doesn't strictly say binary (+1/-1).
    #    "Signed" usually implies non-zero, but could be any real number.
    #    However, "step functions" in this context (C3 problem) often implies values are constrained.
    #    Let's check the seed: np.ones(N). So values can be 1.
    #    If we can use any real numbers, we can scale the sequence arbitrarily, which cancels out in the ratio.
    #    So the shape is what matters.
    #    
    #    Let's try optimizing continuous variables x_i in [-1, 1] to minimize the peak.
    #    Since the objective is scale-invariant, we can fix the norm or sum.
    #    
    #    Algorithm:
    #    a. Initialize x with random values in {-1, 1}.
    #    b. Use a gradient-based or derivative-free method to adjust magnitudes/phases?
    #       Actually, if we allow continuous values, we can use `differential_evolution` or `minimize` with L-BFGS-B.
    #       But N=400 is too large for global DE.
    #    c. Local search: Start with a good candidate (e.g., Zadoff-Chu real part, or random binary),
    #       then perform local moves (flip sign, change magnitude) if they improve the score.
    #    
    #    Let's try a hybrid:
    #    1. Generate a few candidate initial sequences:
    #       - Random binary
    #       - Real part of Zadoff-Chu
    #       - Sine wave (chirp)
    #    2. For each candidate, perform local optimization:
    #       - Iterate: try flipping each bit (or adjusting each value slightly) and keep if improved.
    #       - Since N=400, one full pass is 400 evaluations.
    #       - Each evaluation is an FFT of size 800, which is fast (~microseconds).
    #       - We can do many passes.
    
    def score(x):
        return _autoconv_peak(x)
    
    def local_optimize(x0, max_iters=50):
        x = x0.copy()
        current_score = score(x)
        for i in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Randomize order to avoid bias
            idx = np.random.permutation(N)
            for j in idx:
                if time.time() > deadline:
                    break
                # Try flipping sign
                old_val = x[j]
                x[j] = -x[j]
                new_score = score(x)
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                else:
                    x[j] = old_val
            
            # Also try small perturbations in magnitude?
            # If we allow continuous values, we can try x[j] += delta
            # But binary flipping is a strong local move.
            
            if not improved:
                break
        return x, current_score

    candidates = []
    
    # Candidate 1: Random binary
    candidates.append(np.random.choice([-1, 1], N))
    
    # Candidate 2: Real part of Zadoff-Chu (m=1, N=400 is not prime, but let's use a prime N close to 400 or just use a chirp)
    # Zadoff-Chu: x[n] = exp(-j * pi * u * n * (n+1) / N)
    # Real part: cos(pi * u * n * (n+1) / N)
    # Let's use u=1
    n = np.arange(N)
    zc = np.cos(np.pi * n * (n + 1) / N)
    # Normalize to +/-1? No, just use the values. They are in [-1, 1].
    # The ratio is scale invariant, so we can use the raw values.
    candidates.append(zc)
    
    # Candidate 3: Linear chirp
    candidates.append(np.cos(np.pi * n**2 / N))
    
    # Candidate 4: Sine
    candidates.append(np.sin(2 * np.pi * n / N))
    
    # Candidate 5: All ones (seed)
    candidates.append(np.ones(N))
    
    best_overall_x = None
    best_overall_score = np.inf
    
    for x0 in candidates:
        if time.time() > deadline:
            break
        x_opt, s_opt = local_optimize(x0, max_iters=20)
        if s_opt < best_overall_score:
            best_overall_score = s_opt
            best_overall_x = x_opt
    
    if best_overall_x is not None:
        return best_overall_x
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
