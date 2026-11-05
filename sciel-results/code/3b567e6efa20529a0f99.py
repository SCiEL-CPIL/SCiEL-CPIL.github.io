import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _evaluate(h):
    """Compute the exact objective value for the final answer."""
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)
    return abs(2 * N * max_val / (s * s))

def _soft_max(x, T):
    """Numerically stable soft-max."""
    x_max = np.max(x)
    exp_x = np.exp((x - x_max) / T)
    return x_max + T * np.log(np.sum(exp_x))

def _surrogate_objective(log_mag, T):
    """
    Objective function for optimization.
    log_mag: array of shape (N,) containing log(|H_k|) for k=0..N-1.
    We fix the phase to 0 for simplicity in this initial structural change,
    or rather, we optimize the real and imaginary parts directly?
    Actually, optimizing log-magnitude with fixed phase (0) restricts to real, positive H?
    No, H_k = exp(log_mag_k) * exp(i * phi_k).
    If we fix phi_k = 0, h is real and even (symmetric).
    Let's try optimizing the real and imaginary parts of H directly, 
    but constrained to be real-valued h?
    
    Alternative: Optimize the real coefficients c_k of a cosine series.
    h[n] = sum_{k=0}^{N/2} c_k cos(2*pi*k*n/N).
    This ensures h is real.
    The autoconvolution max is related to the spectral concentration.
    
    Let's stick to the prompt's suggestion of a "different design family".
    The current program uses Fourier basis + DE.
    This program will use **Direct Coordinate Descent on h** with a **Smooth Surrogate** 
    and **Analytic Gradients**.
    
    Why? DE is slow. Coordinate descent with gradients can be very fast if the landscape is smooth.
    The max operation is non-smooth, so we use a soft-max surrogate.
    """
    # This function is a placeholder, the real logic is in construct
    pass

def _compute_grad_h(h, T):
    """
    Compute gradient of the soft-max surrogate objective with respect to h.
    Objective: J = abs(2 * N * S_max / S_sum^2)
    Let M = max(conv(h,h)). We approximate M with soft_max(conv, T).
    Let C = conv(h, h). C is a vector of length 2N-1.
    C_k = sum_i h_i h_{i+k-N+1} (roughly).
    
    Actually, computing the full gradient of the autoconvolution max is complex.
    Let's use a simpler structural change:
    
    **Strategy: Randomized Coordinate Descent with Line Search on h directly.**
    But the current program already does Fourier + DE.
    
    Let's try **Simulated Annealing** on the Fourier coefficients.
    No, the directive says "different optimization or search method".
    
    Let's try **Genetic Algorithm** with specific crossover? No, DE is a GA variant.
    
    Let's try **Nelder-Mead** directly on the 400-dimensional h? Too slow.
    
    Let's try **Gradient Descent on h** using the subgradient of the max.
    The gradient of max(conv) w.r.t h is 2 * h * delta(lag_of_max).
    This is sparse.
    We can do:
    1. Compute conv.
    2. Find lag k* where conv is max.
    3. Gradient of conv[k*] w.r.t h is 2 * h shifted by k*.
    4. Update h -= lr * grad * (objective_factor).
    
    This is a first-order method. It might get stuck in local minima.
    To escape, we can use **Momentum** and **Noise**.
    
    Let's implement **Stochastic Gradient Descent (SGD) with Momentum** on the raw h vector,
    using the subgradient of the exact max (or a small set of top-m maxima to smooth it).
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # State
    h = np.ones(N).astype(np.float64)
    velocity = np.zeros(N)
    lr = 0.01
    momentum = 0.9
    
    # We will use a "Top-K" smoothing of the max to make the gradient less sparse/noisy.
    # Instead of just the single max, we take the max of the top K values.
    # This makes the landscape smoother.
    K_smooth = 5
    
    def _objective_and_grad(h):
        s = np.sum(h)
        if s == 0:
            return 1e10, np.zeros(N)
        
        conv = np.convolve(h, h, mode='full')
        
        # Find top K indices
        # To ensure numerical stability, we take the max of the top K
        # This is a convex combination if we use softmax, but let's just use the max of top K
        # to create a "thick" peak.
        
        # For gradient:
        # If we use M = max(conv), grad_M/grad_h_i = 2 * h_{i - lag}
        # If we use M = max(conv[top_K]), it's the same for the argmax.
        # To smooth, we can use a weighted sum of the top K gradients.
        
        # Let's stick to the exact max for the value, but use a "soft" gradient.
        # Actually, let's use the gradient of the soft-max with a small T.
        T = 1.0
        conv_max = np.max(conv)
        exp_conv = np.exp((conv - conv_max) / T)
        weights = exp_conv / np.sum(exp_conv)
        
        # Gradient of soft_max(conv) w.r.t conv is weights.
        # Gradient of conv_k w.r.t h is 2 * h shifted.
        # grad_h = sum_k weights[k] * 2 * h_shifted(k)
        
        # Efficiently:
        # h_shifted(k) corresponds to correlation.
        # grad_h[i] = 2 * sum_k weights[k] * h[i - (k - (N-1))]
        
        # Let's compute this via FFT for speed?
        # weights is length 2N-1.
        # We can zero-pad weights to length 2N and multiply by FFT(h)?
        # No, it's a convolution of weights with h.
        
        # Let's just do it in time domain if N=400 is small enough.
        # 400 * 800 = 320,000 ops. Very fast.
        
        lag_offsets = np.arange(2 * N - 1) - (N - 1)
        
        # grad_h[i] = 2 * sum_k weights[k] * h[i - lag_offsets[k]]
        # This is a convolution of weights with h?
        # sum_k w[k] h[i-lag_k] is (w * h)[i] if lags are aligned correctly.
        
        # Let's construct the gradient explicitly.
        grad_h = np.zeros(N)
        
        # For each k in conv indices:
        # Contribution to h[i] comes from pairs (i, i+lag) or similar.
        # d(conv_k)/d(h_i) = h_{i-lag} + h_{i+lag} (roughly, depending on definition)
        # np.convolve(h, h)[k] = sum_j h[j] h[k-j]
        # d/dh_i = h[k-i] + h[i] (if i != k-i) or 2h[i]
        
        # This is getting complicated to vectorize perfectly in one shot without bugs.
        # Let's use a simpler structural change:
        
        # **Strategy: Nelder-Mead on a reduced set of "block" variables.**
        # Divide h into 20 blocks of 20.
        # Optimize the mean and variance of each block?
        
        # Let's go back to **Fourier Parameterization** but use **Nelder-Mead** instead of DE.
        # The current program uses DE. Nelder-Mead is a local optimizer.
        # We can run Nelder-Mead from multiple random starts.
        # This is a "different optimization method".
        
        return _evaluate(h), np.zeros(N)

    # Let's implement the Nelder-Mead on Fourier Coefficients strategy.
    # It is robust and doesn't require gradients.
    # DE is population-based. NM is simplex-based.
    # NM often converges faster to local minima in low dimensions.
    
    K_freqs = 8 # Slightly more frequencies than current (5)
    dim = 1 + 2 * K_freqs
    
    def _h_from_coeffs(coeffs):
        n = np.arange(N)
        h = np.zeros(N)
        h += coeffs[0]
        num_pairs = (len(coeffs) - 1) // 2
        for i in range(num_pairs):
            freq = i + 1
            a = coeffs[1 + 2*i]
            b = coeffs[2 + 2*i]
            h += a * np.cos(2 * np.pi * freq * n / N)
            h += b * np.sin(2 * np.pi * freq * n / N)
        return h

    def _objective(coeffs):
        h = _h_from_coeffs(coeffs)
        return _evaluate(h)

    # Run multiple Nelder-Mead optimizations from different starting points
    # This is a "Multi-Start Local Search" strategy.
    
    np.random.seed(42)
    
    # Starting points:
    # 1. Seed (DC only)
    # 2. Random low-freq
    # 3. Random higher freq
    
    starts = []
    # Seed
    c0 = np.zeros(dim)
    c0[0] = 1.0
    starts.append(c0)
    
    # Random starts
    for _ in range(5):
        c_rand = np.random.uniform(-0.5, 0.5, dim)
        c_rand[0] = 1.0 # Keep DC positive to avoid sum=0 issues initially
        starts.append(c_rand)
        
    for start_c in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                _objective, 
                start_c, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            
            if res.fun < best_val:
                best_h = _h_from_coeffs(res.x)
                best_val = res.fun
        except Exception:
            pass

    return best_h

if __name__ == "__main__":
    h = construct()
    print("Value:", _evaluate(h))
