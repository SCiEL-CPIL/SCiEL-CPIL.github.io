"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _autoconv_fft(x):
    """Compute linear autocorrelation via FFT. Returns array of length 2*N-1."""
    M = 1
    while M < 2 * N - 1:
        M *= 2
    X = np.fft.rfft(x, M)
    P = X * np.conj(X)
    ac = np.fft.irfft(P, M)
    # Linear autocorrelation: ac[k] for k=0..N-1, and ac[M-k] for k=1..N-1 correspond to negative lags
    # Actually, the full linear autocorrelation of length 2N-1 is:
    # r[k] = sum_{i} x[i] * x[i+k] for k=0..N-1
    # r[k] = sum_{i} x[i] * x[i-k] for k=1..N-1 (negative lags)
    # The irfft gives circular correlation. With zero-padding to M >= 2N-1, 
    # the first N values are r[0], r[1], ..., r[N-1]
    # The remaining M-N values wrap around.
    # Specifically, ac[0:N] gives lags 0 to N-1
    # ac[M-N+1:M] gives lags -(N-1) to -1 (reversed)
    # Let's just take the first 2*N-1 values and arrange properly
    # Actually simpler: the linear autocorrelation is just the first N positive lags and 
    # by symmetry we only need max over all. Since |r[-k]| = |r[k]|, we just need max over k=0..N-1
    return ac[:N]

def _fast_autocorr_max_abs(x):
    """Compute the maximum absolute value of the linear autocorrelation of x."""
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac))

def _objective_and_grad(x, beta=50.0):
    """
    Smooth objective: softmax approximation to max|ac| with gradient.
    Uses log-sum-exp for stability.
    """
    ac = _autoconv_fft(x)
    # We want to minimize max(|ac[k]|)
    # Softmax approximation: (1/beta) * log(sum exp(beta * |ac[k]|))
    # But |ac| is non-smooth. Instead, use sum of squared terms weighted by softmax.
    # Alternative: use the epigraph formulation or just use the max directly with subgradient.
    
    # Let's use a different smooth surrogate: 
    # f(x) = (1/beta) * log(sum_k exp(beta * ac[k]^2))
    # This is smooth and approximates max(ac^2) as beta -> inf
    # Gradient: d/dx f = (1/beta) * d/dx log(sum exp(beta * ac^2))
    #           = (1/beta) * sum_k (beta * 2 * ac[k] * d(ac[k])/dx) * exp(beta * ac[k]^2) / sum_k exp(beta * ac[k]^2)
    #           = sum_k (2 * ac[k] * d(ac[k])/dx) * w_k
    # where w_k = exp(beta * ac[k]^2) / sum_j exp(beta * ac[j]^2)
    
    ac_sq = ac ** 2
    max_ac_sq = np.max(ac_sq)
    # Shift for numerical stability
    shifted = ac_sq - max_ac_sq
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    w = exp_shifted / sum_exp  # weights, sum to 1
    
    obj = (1.0 / beta) * (np.log(sum_exp) + beta * max_ac_sq)
    
    # Gradient computation
    # ac[k] = sum_i x[i] * x[i+k] for k=0..N-1
    # d(ac[k])/dx[j] = x[j+k] + x[j-k] (with boundary conditions, x[i]=0 for i<0 or i>=N)
    # So grad = sum_k w_k * 2 * ac[k] * (x[j+k] + x[j-k])
    # This can be computed via correlation:
    # grad = 2 * (w * ac) convolved with x (in a specific way)
    # Actually: grad[j] = 2 * sum_k w_k * ac[k] * (x[j+k] + x[j-k])
    #            = 2 * sum_k w_k * ac[k] * x[j+k] + 2 * sum_k w_k * ac[k] * x[j-k]
    # Let v[k] = 2 * w[k] * ac[k] for k=0..N-1
    # Then sum_k v[k] * x[j+k] is a cross-correlation of v and x
    # And sum_k v[k] * x[j-k] is also a cross-correlation (reversed)
    
    v = 2.0 * w * ac  # length N
    # Cross-correlation: c[j] = sum_k v[k] * x[j+k] for j=0..N-1-k
    # This is like a convolution. Let's use FFT.
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    V = np.fft.rfft(v, M)
    X = np.fft.rfft(x, M)
    
    # First term: sum_k v[k] * x[j+k]
    # This is the cross-correlation of v and x, shifted.
    # In FFT domain, cross-correlation is V * conj(X)
    # But we need to be careful with the definition.
    # Let's compute it directly using the convolution theorem.
    # sum_k v[k] * x[j+k] = (v * x_rev)[j + N - 1] where x_rev is reversed x
    # Actually, let's just use the fact that this is a linear operation.
    
    # Simpler approach: compute the gradient numerically? No, too slow.
    # Let's use the analytical gradient via FFT.
    # ac[k] = sum_i x[i] x[i+k]
    # d(ac[k])/dx[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
    # So grad[j] = 2 * sum_k w[k] ac[k] (x[j+k] + x[j-k])
    #             = 2 * sum_k w[k] ac[k] x[j+k] + 2 * sum_k w[k] ac[k] x[j-k]
    
    # Let u[k] = w[k] * ac[k] for k=0..N-1
    u = w * ac
    
    # Term 1: t1[j] = sum_k u[k] x[j+k]
    # This is the cross-correlation of u and x at lag -j
    # Using FFT: t1 = ifft(conj(FFT(u)) * FFT(x)) but with proper alignment
    # Actually, let's just do it with a direct sum for clarity, but N=400 so O(N^2) is 160k ops, fast enough.
    # But we're calling this in L-BFGS many times. Let's use FFT.
    
    # t1[j] = sum_{k=0}^{N-1-j} u[k] x[j+k]
    # This is the cross-correlation. In FFT:
    # corr = ifft(FFT(u) * conj(FFT(x))) gives sum_k u[k] x[k+j]
    # So t1 = corr[0:N]
    
    U = np.fft.rfft(u, M)
    t1_full = np.fft.irfft(U * np.conj(X), M)
    t1 = t1_full[:N]
    
    # Term 2: t2[j] = sum_k u[k] x[j-k]
    # This is sum_k u[k] x[j-k] = sum_{k=0}^{j} u[k] x[j-k] (for k<=j)
    # This is the convolution of u and x at position j
    # conv[j] = sum_k u[k] x[j-k]
    # In FFT: conv = ifft(FFT(u) * FFT(x))
    
    t2_full = np.fft.irfft(U * X, M)
    t2 = t2_full[:N]
    
    grad = 2.0 * (t1 + t2)
    
    return obj, grad

def _spectral_init(rng):
    """Generate initial signal using spectral phase retrieval."""
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    half_M = M // 2
    freqs = np.arange(M)
    
    # Gaussian magnitude spectrum
    sigma = M / 6.0
    mag = np.exp(-0.5 * (freqs / sigma) ** 2)
    mag[N:] = 0  # Band-limit to avoid aliasing artifacts
    
    # Random phases for positive frequencies
    phases = rng.uniform(0, 2 * np.pi, half_M)
    
    spec = np.zeros(M, dtype=np.complex128)
    spec[0] = mag[0]
    if M % 2 == 0:
        spec[half_M] = mag[half_M]
    
    for k in range(1, half_M):
        spec[k] = mag[k] * np.exp(1j * phases[k-1])
        spec[M-k] = np.conj(spec[k])
        
    x = np.fft.ifft(spec).real
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x

def _coordinate_descent(x, current_val, rng, time_limit):
    """
    Coordinate descent on the +/-1 sequence.
    Try flipping individual bits and small groups.
    """
    current_x = x.copy()
    current_val = _fast_autocorr_max_abs(current_x)
    best_val = current_val
    best_x = current_x.copy()
    
    t0 = time.time()
    no_improve = 0
    
    while time.time() - t0 < time_limit:
        # Try random single flips
        for _ in range(50):
            i = rng.integers(0, N)
            current_x[i] *= -1
            new_val = _fast_autocorr_max_abs(current_x)
            if new_val < current_val:
                current_val = new_val
                no_improve = 0
                if new_val < best_val:
                    best_val = new_val
                    best_x = current_x.copy()
            else:
                current_x[i] *= -1
                no_improve += 1
        
        if no_improve > 200:
            # Try larger perturbations: flip a contiguous block
            start = rng.integers(0, N - 10)
            length = rng.integers(5, 20)
            current_x[start:start+length] *= -1
            new_val = _fast_autocorr_max_abs(current_x)
            if new_val < current_val:
                current_val = new_val
                no_improve = 0
                if new_val < best_val:
                    best_val = new_val
                    best_x = current_x.copy()
            else:
                current_x[start:start+length] *= -1
                no_improve += 100
        
        # Occasionally try a sinusoidal perturbation
        if rng.random() < 0.1:
            freq = rng.uniform(0.01, 0.1)
            phase = rng.uniform(0, 2 * np.pi)
            perturbation = np.sin(2 * np.pi * freq * np.arange(N) + phase)
            new_x = np.sign(current_x + 0.1 * perturbation)
            new_x[new_x == 0] = 1
            new_val = _fast_autocorr_max_abs(new_x)
            if new_val < current_val:
                current_x = new_x
                current_val = new_val
                no_improve = 0
                if new_val < best_val:
                    best_val = new_val
                    best_x = current_x.copy()
            else:
                no_improve += 50
    
    return best_x, best_val

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_val = float('inf')
    best_x = np.zeros(N, dtype=np.float64)
    
    # Generate multiple diverse initializations
    num_inits = 15
    inits = []
    for _ in range(num_inits):
        x_init = _spectral_init(rng)
        inits.append(x_init)
        
    # For each initialization, run L-BFGS on the smooth surrogate
    for x0 in inits:
        if time.time() > deadline - 30.0:
            break
            
        try:
            res = minimize(_objective_and_grad, x0, method='L-BFGS-B', 
                           jac=True,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
            
            x_opt = res.x
            val_opt = _fast_autocorr_max_abs(x_opt)
            
            # Threshold to +/-1
            x_quant = np.sign(x_opt)
            x_quant[x_quant == 0] = 1
            
            val_quant = _fast_autocorr_max_abs(x_quant)
            
            if val_quant < best_val:
                best_val = val_quant
                best_x = x_quant.copy()
                
        except Exception:
            x_quant = np.sign(x0)
            x_quant[x_quant == 0] = 1
            val_quant = _fast_autocorr_max_abs(x_quant)
            if val_quant < best_val:
                best_val = val_quant
                best_x = x_quant.copy()

    # Final local search refinement
    time_left = deadline - time.time()
    if time_left > 10.0:
        best_x, best_val = _coordinate_descent(best_x, best_val, rng, time_left - 5.0)

    return best_x.astype(np.int8)
# EVOLVE-BLOCK-END
