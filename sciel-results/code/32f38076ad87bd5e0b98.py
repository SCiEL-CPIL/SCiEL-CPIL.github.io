import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def compute_value(h):
    """Compute the task value."""
    h_sum = np.sum(h)
    if h_sum == 0 or not np.isfinite(h_sum):
        return float('inf')
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (h_sum ** 2))

def _fft_convolve(h):
    """Compute full autoconvolution using FFT for speed."""
    size = 2 * N
    H = np.fft.fft(h, size)
    return np.fft.ifft(H * H).real[:2*N-1]

def _smooth_max_value(h, alpha=50.0):
    """Compute smooth approximation of the task value."""
    h_sum = np.sum(h)
    if abs(h_sum) < 1e-10 or not np.isfinite(h_sum):
        return 1e6
    conv = _fft_convolve(h)
    # Smooth max using log-sum-exp
    shifted = conv / alpha
    max_shifted = np.max(shifted)
    smooth_max = alpha * (max_shifted + np.log(np.mean(np.exp(shifted - max_shifted))))
    val = abs(2.0 * N * smooth_max / (h_sum * h_sum))
    return val if np.isfinite(val) else 1e6

def _build_walsh_hadamard(coeffs):
    """
    Build h from Walsh-Hadamard coefficients.
    coeffs: array of length M (subset of Hadamard basis)
    h = sum_k coeffs[k] * H_k where H_k are Hadamard basis vectors
    """
    M = len(coeffs)
    # Generate first M Walsh-Hadamard basis vectors (excluding DC)
    n = np.arange(N)
    h = np.zeros(N)
    for k in range(1, M+1):  # Skip DC component (k=0)
        if coeffs[k-1] == 0:
            continue
        # Walsh-Hadamard function: sign of inner product with k
        # Use the standard Hadamard matrix construction
        # H_k[n] = (-1)^(popcount(n & k))
        # This gives orthogonal sequences with values in {-1, +1}
        x = n & k
        # Count set bits
        popcount = np.array([bin(v).count('1') for v in x])
        H_k = (-1.0) ** popcount
        h += coeffs[k-1] * H_k
    return h

def _hadamard_basis(M, N):
    """Precompute first M+1 Walsh-Hadamard basis vectors (including DC)."""
    basis = np.zeros((M+1, N))
    n = np.arange(N)
    for k in range(M+1):
        x = n & k
        popcount = np.array([bin(v).count('1') for v in x])
        basis[k] = (-1.0) ** popcount
    return basis

def _walsh_objective(coeffs, basis):
    """Objective for Walsh-Hadamard coefficient optimization."""
    h = basis.T @ coeffs  # Fast: matrix-vector multiply
    h_sum = np.sum(h)
    if abs(h_sum) < 1e-10 or not np.isfinite(h_sum):
        return 1e6
    conv = _fft_convolve(h)
    alpha = 50.0
    shifted = conv / alpha
    max_shifted = np.max(shifted)
    smooth_max = alpha * (max_shifted + np.log(np.mean(np.exp(shifted - max_shifted))))
    val = abs(2.0 * N * smooth_max / (h_sum * h_sum))
    return val if np.isfinite(val) else 1e6

def _walsh_gradient(coeffs, basis, alpha=50.0):
    """Compute gradient of smooth objective w.r.t. Walsh coefficients."""
    h = basis.T @ coeffs
    h_sum = np.sum(h)
    if abs(h_sum) < 1e-10:
        return np.zeros_like(coeffs)
    
    conv = _fft_convolve(h)
    # Gradient of smooth max
    shifted = conv / alpha
    max_shifted = np.max(shifted)
    weights = np.exp(shifted - max_shifted)
    weights /= np.sum(weights)
    
    # d(smooth_max)/d(conv_k) = weights[k]
    # d(conv_k)/d(h_i) = h_j where k = i + j (for full conv)
    # Need to backprop through convolution
    
    # Gradient of conv w.r.t. h
    # conv[k] = sum_i h[i]*h[k-i]
    # d(conv[k])/d(h[m]) = h[k-m] + h[m-k] (for valid indices)
    
    grad_h = np.zeros(N)
    for k in range(len(conv)):
        if weights[k] < 1e-15:
            continue
        wk = weights[k]
        # h[k-m] for valid k-m
        idx1 = k - np.arange(N)
        mask1 = (idx1 >= 0) & (idx1 < N)
        grad_h[mask1] += wk * h[idx1[mask1]]
        # h[m-k] for valid m-k
        idx2 = np.arange(N) - k
        mask2 = (idx2 >= 0) & (idx2 < N)
        grad_h[mask2] += wk * h[idx2[mask2]]
    
    # d(val)/d(h) where val = |2*N*smooth_max / h_sum^2|
    # Assume h_sum > 0 for simplicity (can adjust sign later)
    sign = np.sign(2.0 * N * smooth_max / (h_sum * h_sum))
    # d(val)/d(h_sum) = -2 * sign * 2*N*smooth_max / h_sum^3
    # d(val)/d(smooth_max) = sign * 2*N / h_sum^2
    # d(smooth_max)/d(h) = alpha * weights backprop'd through conv
    
    # Actually let's just use finite differences for gradient in coefficient space
    # Or compute analytically
    
    # d(val)/d(h_i) = sign * (2*N/h_sum^2) * (alpha * grad_h[i]) - sign * (2*N*smooth_max/h_sum^3) * 2 * h_sum * delta_i
    # Wait, h_sum = sum(h), so d(h_sum)/d(h_i) = 1
    # val = sign * 2*N*smooth_max / h_sum^2
    # d(val)/d(h_i) = sign * [2*N * d(smooth_max)/d(h_i) / h_sum^2 - 2*N*smooth_max * 2 / h_sum^3]
    #                = sign * 2*N / h_sum^2 * [d(smooth_max)/d(h_i) - 2*smooth_max/h_sum]
    
    d_smooth_d_h = alpha * grad_h  # This is d(smooth_max)/d(h_i) via the weight backprop
    
    d_val_d_h = sign * 2.0 * N / (h_sum * h_sum) * (d_smooth_d_h - 2.0 * smooth_max / h_sum)
    
    # Now d(val)/d(coeffs_k) = sum_i d(val)/d(h_i) * d(h_i)/d(coeffs_k) = d(val)/d(h) dot basis[k]
    grad_coeffs = basis @ d_val_d_h
    return grad_coeffs

def _optimize_walsh(basis, M, time_limit=30.0, start_time=0.0, total_limit=140.0):
    """Optimize Walsh-Hadamard coefficients using gradient descent."""
    best_coeffs = np.zeros(M)
    best_val = _walsh_objective(best_coeffs, basis)
    
    # Initialize with random coefficients
    np.random.seed(123)
    coeffs = np.random.randn(M) * 0.5
    
    # Learning rate schedule
    lr = 0.01
    patience = 0
    best_local = best_val
    
    for it in range(10000):
        if time.time() - start_time > total_limit - 5:
            break
        
        val = _walsh_objective(coeffs, basis)
        if val < best_val:
            best_val = val
            best_coeffs = coeffs.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > 100:
            # Restart with new random initialization
            coeffs = np.random.randn(M) * 0.3
            lr = 0.01
            patience = 0
            continue
        
        # Compute gradient
        grad = _walsh_gradient(coeffs, basis)
        
        # Gradient step
        coeffs = coeffs - lr * grad
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
        
    return best_coeffs, best_val

def _zadoff_chu_h(N):
    """Generate Zadoff-Chu sequence of length N (must be prime for perfect ZC)."""
    # Use N=400, find closest prime or use general formula
    # Zadoff-Chu: x[n] = exp(-j*pi*m*n*(n+1)/N) for m coprime to N
    # For N=400=16*25, use m=1
    n = np.arange(N)
    m = 1
    phase = -np.pi * m * n * (n + 1) / N
    return np.cos(phase)  # Take real part

def _windowed_zadoff_chu_h(N, window_type='hann'):
    """Zadoff-Chu with windowing."""
    zc = _zadoff_chu_h(N)
    if window_type == 'hann':
        window = np.hanning(N)
    elif window_type == 'hamming':
        window = np.hamming(N)
    else:
        window = np.ones(N)
    return zc * window

def _zadoff_chu_objective(params):
    """Objective for Zadoff-Chu based optimization."""
    A = params[0]
    f = params[1]
    offset = params[2]
    n = np.arange(N)
    phase = -np.pi * n * (n + 1) / N * f
    h = offset + A * np.cos(phase)
    return _smooth_max_value(h)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # Should be 2.0
    
    # Strategy 1: Walsh-Hadamard basis optimization
    M = 32  # Number of Walsh-Hadamard basis vectors to use
    
    if time.time() - start_time < time_limit - 20:
        basis = _hadamard_basis(M, N)
        walsh_coeffs, walsh_val = _optimize_walsh(basis, M, start_time=start_time, total_limit=time_limit)
        if walsh_val < best_val:
            best_val = walsh_val
            best_h = basis.T @ walsh_coeffs
    
    # Strategy 2: Zadoff-Chu based optimization
    if time.time() - start_time < time_limit - 15:
        bounds_zc = [
            (0.1, 2.0),    # A: amplitude
            (0.5, 5.0),    # f: frequency scaling
            (-0.5, 2.0),   # offset
        ]
        
        np.random.seed(42)
        for trial in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            x0 = np.array([
                np.random.uniform(bounds_zc[0][0], bounds_zc[0][1]),
                np.random.uniform(bounds_zc[1][0], bounds_zc[1][1]),
                np.random.uniform(bounds_zc[2][0], bounds_zc[2][1]),
            ])
            
            res = minimize(
                _zadoff_chu_objective, x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                A, f, offset = res.x
                n = np.arange(N)
                phase = -np.pi * n * (n + 1) / N * f
                best_h = offset + A * np.cos(phase)
    
    # Strategy 3: Final polish with direct gradient descent on h
    if time.time() - start_time < time_limit - 5:
        h = best_h.copy()
        h_sum = np.sum(h)
        if abs(h_sum) > 1e-10:
            h = h / h_sum  # Normalize to sum 1
            
        lr = 0.005
        for it in range(2000):
            if time.time() - start_time > time_limit - 2:
                break
            
            conv = _fft_convolve(h)
            alpha = 30.0
            shifted = conv / alpha
            max_shifted = np.max(shifted)
            weights = np.exp(shifted - max_shifted)
            weights /= np.sum(weights)
            
            # Gradient of smooth max w.r.t. h
            grad_h = np.zeros(N)
            for k in range(len(conv)):
                if weights[k] < 1e-12:
                    continue
                wk = weights[k]
                idx1 = k - np.arange(N)
                mask1 = (idx1 >= 0) & (idx1 < N)
                grad_h[mask1] += wk * h[idx1[mask1]]
                idx2 = np.arange(N) - k
                mask2 = (idx2 >= 0) & (idx2 < N)
                grad_h[mask2] += wk * h[idx2[mask2]]
            
            smooth_max = alpha * (max_shifted + np.log(np.mean(np.exp(shifted - max_shifted))))
            h_sum = np.sum(h)
            sign = np.sign(2.0 * N * smooth_max / (h_sum * h_sum))
            d_val_d_h = sign * 2.0 * N / (h_sum * h_sum) * (alpha * grad_h - 2.0 * smooth_max / h_sum)
            
            h = h - lr * d_val_d_h
            h = h - (np.sum(h) - 1.0) / N  # Keep sum = 1
            
            lr *= 0.998
            
            val = compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    return best_h
