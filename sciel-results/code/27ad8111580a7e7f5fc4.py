import numpy as np
import time
import math
import random
import sys

_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin before 150s kill

N = 24
FFT_SIZE = 1 << 17  # 131072, large enough for accurate sup-norm estimation
SQRTN = math.sqrt(N)

def _time_left():
    return _DEADLINE - time.time()

def _supnorm_fast(coeffs):
    """
    Fast sup-norm estimate via FFT on a dense grid.
    coeffs: list or array of length N
    """
    c = np.asarray(coeffs, dtype=float)
    padded = np.zeros(FFT_SIZE, dtype=float)
    padded[:N] = c
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _eval_poly(coeffs, theta):
    """
    Evaluate P(e^{i theta}) = sum c_k e^{i k theta}
    coeffs: list or array of length N
    theta: scalar or array of angles
    """
    c = np.asarray(coeffs, dtype=float)
    k = np.arange(N)
    if np.isscalar(theta):
        z = np.exp(1j * theta)
        # Compute sum c_k z^k
        # Use Horner's method or direct sum
        # Direct sum is fine for N=24
        powers = z ** k
        return float(np.sum(c * powers))
    else:
        # theta is an array
        zs = np.exp(1j * theta)
        # zs shape (M,), k shape (N,)
        # We want P_vals[m] = sum_k c_k * zs[m]^k
        # powers[m, k] = zs[m]^k
        powers = zs[:, None] ** k[None, :]
        P_vals = powers @ c
        return P_vals

def _refine_peak(coeffs, theta0, window=1e-4, n_iter=50):
    """
    Refine a peak near theta0 using golden-section search.
    Returns the maximum |P(e^{i theta})| in the neighborhood.
    """
    c = np.asarray(coeffs, dtype=float)
    
    def f(theta):
        val = _eval_poly(c, theta)
        return -abs(val)  # We want to maximize |val|, so minimize -|val|
    
    # Golden-section search to minimize f(theta) = -|P(e^{i theta})|
    # This finds the local maximum of |P|
    
    # First, do a small grid to find the best point
    thetas = np.linspace(theta0 - window, theta0 + window, 100)
    vals = np.abs(_eval_poly(c, thetas))
    best_idx = np.argmax(vals)
    theta_best = thetas[best_idx]
    val_best = vals[best_idx]
    
    # Now refine using golden section
    gr = (math.sqrt(5) - 1) / 2  # ~0.618
    
    a = theta0 - window
    b = theta0 + window
    
    # Ensure the peak is inside [a, b]
    # The grid search already found it, so it should be
    
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    
    f1 = f(c1)
    f2 = f(c2)
    
    for _ in range(n_iter):
        if f1 < f2:
            # Minimum is in [a, c2]
            b = c2
            c2 = c1
            f2 = f1
            c1 = b - gr * (b - a)
            f1 = f(c1)
        else:
            # Minimum is in [c1, b]
            a = c1
            c1 = c2
            f1 = f2
            c2 = a + gr * (b - a)
            f2 = f(c2)
    
    # The minimum of f is at either c1 or c2
    if f1 < f2:
        theta_min = c1
        f_min = f1
    else:
        theta_min = c2
        f_min = f2
    
    return -f_min  # This is the maximum |P|

def _supnorm_precise(coeffs):
    """
    More precise sup-norm using refinement of peaks from FFT.
    """
    c = np.asarray(coeffs, dtype=float)
    # First, get a coarse grid
    coarse_size = 1 << 16
    padded = np.zeros(coarse_size, dtype=float)
    padded[:N] = c
    vals = np.fft.fft(padded)
    mags = np.abs(vals)
    
    # Find the top few peaks in the coarse grid
    top_k = 20
    indices = np.argpartition(mags, -top_k)[-top_k:]
    
    max_val = 0.0
    for idx in indices:
        # The frequency is idx / coarse_size
        freq = idx / coarse_size
        theta0 = 2 * math.pi * freq
        # Also check the negative frequency (due to FFT symmetry)
        # For real coefficients, P(e^{i theta}) and P(e^{-i theta}) have same magnitude
        # So we just need to check theta0
        
        # Refine around theta0
        refined_val = _refine_peak(c, theta0, window=1.0/coarse_size)
        if refined_val > max_val:
            max_val = refined_val
            
    return max_val

def _hill_climb(coeffs, use_two_bit=True):
    """
    Perform local search with 1-bit and 2-bit flips.
    Returns the best coefficients found.
    """
    coeffs = list(coeffs)
    current_val = _supnorm_fast(coeffs)
    
    # We use a loop that runs until time is low or no improvement is found
    improved = True
    while improved and _time_left() > 2.0:
        improved = False
        
        # 1. Try all 1-bit flips
        best_1_val = current_val
        best_1_idx = -1
        for k in range(N):
            coeffs[k] = -coeffs[k]
            v = _supnorm_fast(coeffs)
            coeffs[k] = -coeffs[k]  # revert
            if v < best_1_val - 1e-12:
                best_1_val = v
                best_1_idx = k
                
        if best_1_idx != -1:
            coeffs[best_1_idx] = -coeffs[best_1_idx]
            current_val = best_1_val
            improved = True
            continue
            
        # 2. Try all 2-bit flips if no 1-bit improvement
        if use_two_bit:
            best_2_val = current_val
            best_2_i = -1
            best_2_j = -1
            for i in range(N):
                for j in range(i + 1, N):
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
                    v = _supnorm_fast(coeffs)
                    coeffs[i] = -coeffs[i]
                    coeffs[j] = -coeffs[j]
                    if v < best_2_val - 1e-12:
                        best_2_val = v
                        best_2_i = i
                        best_2_j = j
                        
            if best_2_i != -1:
                coeffs[best_2_i] = -coeffs[best_2_i]
                coeffs[best_2_j] = -coeffs[best_2_j]
                current_val = best_2_val
                improved = True
                continue
        
        # 3. If no improvement, break
        break

    return coeffs, current_val

def _golay_rudin_shapiro(n):
    """
    Generate Golay-Rudin-Shapiro sequence of length n.
    c_k = (-1)^{popcount(k)}
    """
    coeffs = []
    for k in range(n):
        if bin(k).count('1') % 2 == 0:
            coeffs.append(1)
        else:
            coeffs.append(-1)
    return coeffs

def _shifted_grs(n, shift):
    """
    Generate a shifted Golay-Rudin-Shapiro sequence.
    This provides diverse starting points.
    """
    base = _golay_rudin_shapiro(n)
    # Circular shift
    return base[shift:] + base[:shift]

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    np.random.seed(42)
    
    best_coeffs = None
    best_val = float('inf')
    
    # List of initializers
    initializers = []
    
    # 1. Golay-Rudin-Shapiro and its shifts
    for shift in range(N):
        initializers.append(_shifted_grs(N, shift))
    
    # 2. Random sequences
    for _ in range(20):
        initializers.append([random.choice([1, -1]) for _ in range(N)])
        
    # Process each initializer
    for init in initializers:
        if _time_left() < 5.0:
            break
            
        coeffs, val = _hill_climb(init)
        
        # Refine the best candidate with a precise sup-norm check
        if val < best_val:
            precise_val = _supnorm_precise(coeffs)
            if precise_val < best_val:
                best_val = precise_val
                best_coeffs = coeffs
        
    # If we have time left, do more random restarts
    while _time_left() > 10.0:
        init = [random.choice([1, -1]) for _ in range(N)]
        coeffs, val = _hill_climb(init)
        
        if val < best_val:
            precise_val = _supnorm_precise(coeffs)
            if precise_val < best_val:
                best_val = precise_val
                best_coeffs = coeffs
                
    # Final check: ensure we return a valid list of ints
    if best_coeffs is None:
        best_coeffs = _golay_rudin_shapiro(N)
        
    return [int(c) for c in best_coeffs]

if __name__ == "__main__":
    result = construct()
    # Print score for debugging
    val = _supnorm_precise(result)
    print(f"Score: {val / SQRTN}")
    print(result)
