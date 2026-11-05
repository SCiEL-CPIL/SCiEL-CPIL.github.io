"""
Flattest plus-minus-one polynomials with 24 coefficients.
Strategy:
  1. Fast FFT-based sup-norm screening on a dense grid.
  2. Golden-section refinement of the top few peaks for accurate evaluation.
  3. Multi-start local search: random restarts + simulated annealing / 2-bit flips.
  4. Deadline guard: return best found before 140 s wall-clock.
"""
import numpy as np
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150 s kill

N = 24
SQRTN = math.sqrt(N)

# Precompute powers of z for evaluation at a given theta (vectorized)
_KS = np.arange(N)

def _time_left():
    return _DEADLINE - time.time()

def _eval_poly(coeffs, theta):
    """Evaluate |P(e^{i theta})| where P(z) = sum c_k z^k."""
    z = np.exp(1j * theta)
    # Use dot product with precomputed powers
    return float(np.abs(np.dot(coeffs, z ** _KS)))

def _refine_peak(coeffs, ang, iters=50):
    """
    Golden-section refine the maximum of |P(e^{i theta})| near ang.
    We search in a small window around ang.
    """
    # Window: half-width ~ pi/512 (since grid is 8192 points)
    half_w = math.pi / 4096.0
    a = ang - half_w
    b = ang + half_w
    
    # Golden ratio
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _eval_poly(coeffs, c)
    fd = _eval_poly(coeffs, d)
    
    for _ in range(iters):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = _eval_poly(coeffs, d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = _eval_poly(coeffs, c)
    
    mid = (a + b) / 2.0
    return _eval_poly(coeffs, mid)

def _supnorm_precise(coeffs, n_grid=8192):
    """
    Compute sup-norm: FFT screening on dense grid, then refine top peaks.
    """
    # FFT screening
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    abs_vals = np.abs(vals)
    
    # Get top 10 peak indices
    top_idx = np.argsort(abs_vals)[-10:][::-1]
    
    best = 0.0
    for idx in top_idx:
        ang = 2.0 * math.pi * idx / n_grid
        val = _refine_peak(coeffs, ang, iters=40)
        if val > best:
            best = val
    
    return best

def _supnorm_fast(coeffs, n_grid=4096):
    """Fast sup-norm estimate via FFT only (no refinement)."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _local_search(coeffs, max_iters=2000):
    """
    2-bit-flip local search with simulated annealing.
    Returns best coeffs and best sup-norm found.
    """
    coeffs = list(coeffs)
    best_coeffs = list(coeffs)
    best_val = _supnorm_fast(coeffs)
    
    # Use fast screening for most iterations, precise for candidates
    temp = 0.5
    cool_rate = 0.999
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Pick two distinct indices to flip
        i, j = random.sample(range(N), 2)
        coeffs[i] = -coeffs[i]
        coeffs[j] = -coeffs[j]
        
        new_val = _supnorm_fast(coeffs)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = list(coeffs)
        elif new_val < best_val + temp:
            # Accept with some probability (annealing)
            if random.random() < math.exp(-(new_val - best_val) / (temp + 1e-12)):
                pass  # keep the move
            else:
                # Reject: undo
                coeffs[i] = -coeffs[i]
                coeffs[j] = -coeffs[j]
        else:
            # Undo
            coeffs[i] = -coeffs[i]
            coeffs[j] = -coeffs[j]
        
        temp *= cool_rate
    
    # Refine the best found
    final_val = _supnorm_precise(best_coeffs)
    return best_coeffs, final_val

def _try_golay_rudin_shapiro():
    """Golay-Rudin-Shapiro recursion, truncated to 24 coefficients."""
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    return p[:24]

def construct():
    """
    Return 24 coefficients in {+1, -1} minimizing max_{|z|=1} |P(z)| / sqrt(24).
    """
    random.seed(42)
    np.random.seed(42)
    
    # Start with Golay-Rudin-Shapiro as initial best
    best_coeffs = _try_golay_rudin_shapiro()
    best_val = _supnorm_precise(best_coeffs)
    
    # Multi-start local search with random restarts
    n_restarts = 500
    for r in range(n_restarts):
        if _time_left() < 10.0:
            break
        
        # Random initial polynomial
        coeffs = [random.choice([1, -1]) for _ in range(N)]
        
        # Local search from this start
        new_coeffs, new_val = _local_search(coeffs, max_iters=1500)
        
        if new_val < best_val:
            best_val = new_val
            best_coeffs = new_coeffs
    
    # Final precise evaluation of the best
    final_val = _supnorm_precise(best_coeffs)
    
    return best_coeffs

# Allow module-level access if needed
if __name__ == "__main__":
    result = construct()
    print(result)
