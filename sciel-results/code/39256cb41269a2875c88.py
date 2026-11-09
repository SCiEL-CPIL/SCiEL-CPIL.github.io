"""
Gradient-based optimization for signed step function autocorrelation minimization.

Strategy:
1. Initialize with structured sequences (Thue-Morse, quadratic residues, random).
2. Use continuous relaxation: map real variables x to signs via sign(x).
   Optimize a smooth surrogate using L-BFGS-B where the objective is the
   smoothed peak sidelobe (using soft-max or differentiable max approximation).
3. After continuous optimization, snap to ±1 and do discrete local search polish.
4. Multiple restarts with time-bounded search.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _autocorrelation(h):
    """Compute the autocorrelation of h using FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, n=size)
    R = np.fft.irfft(H * H, n=size)[:n]
    return R

def _score(h):
    """Compute the score: peak sidelobe of autocorrelation normalized by N."""
    R = _autocorrelation(h)
    if len(R) > 1:
        peak_sidelobe = np.max(np.abs(R[1:]))
    else:
        peak_sidelobe = 0.0
    return peak_sidelobe / N

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq[i] = 1.0 if c % 2 == 0 else -1.0
    return seq

def _quadratic_residues(n, p):
    """Generate a sequence based on quadratic residues mod p, tiled to length n."""
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    while not is_prime(p):
        p += 1
    
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    seq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        seq[i] = 1.0 if i % p in residues else -1.0
    return seq

def _smoothed_objective(x, temperature=1.0):
    """
    Smoothed objective for continuous optimization.
    x is a real vector of length N. We map to h[i] = tanh(x[i]/temperature).
    The objective is the peak sidelobe of the autocorrelation of h, normalized by N.
    To make it differentiable, we use a soft-max approximation of the max.
    """
    h = np.tanh(x / temperature)
    R = _autocorrelation(h)
    if len(R) > 1:
        sidelobes = np.abs(R[1:])
        # Soft max approximation
        beta = 50.0
        soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * sidelobes)))
        return soft_max / N
    else:
        return 0.0

def _smoothed_gradient(x, temperature=1.0):
    """
    Numerical gradient of the smoothed objective.
    """
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _smoothed_objective(x, temperature)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_smoothed_objective(x_plus, temperature) - _smoothed_objective(x_minus, temperature)) / (2 * eps)
    return grad

def _discrete_local_search(h, max_iterations=5000):
    """
    Greedy local search: flip bits to reduce the peak sidelobe.
    """
    R = _autocorrelation(h)
    current_score = np.max(np.abs(R[1:])) / N
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(N):
            old_val = h[idx]
            h[idx] *= -1.0
            new_R = _autocorrelation(h)
            new_score = np.max(np.abs(new_R[1:])) / N
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                R = new_R
                improved = True
            else:
                h[idx] = old_val
    return h, current_score

def _simulated_annealing(h, T_start=0.05, T_end=0.001, cooling=0.9995, max_iter=5000):
    """
    Simulated annealing for discrete bit-flip optimization.
    """
    R = _autocorrelation(h)
    current_score = np.max(np.abs(R[1:])) / N
    best_h = h.copy()
    best_score = current_score
    
    T = T_start
    for _ in range(max_iter):
        if T <= T_end:
            break
        idx = np.random.randint(N)
        old_val = h[idx]
        old_R = R.copy()
        h[idx] *= -1.0
        new_R = _autocorrelation(h)
        new_score = np.max(np.abs(new_R[1:])) / N
        
        delta_s = new_score - current_score
        if delta_s < 0 or np.random.random() < math.exp(-delta_s / T):
            current_score = new_score
            R = new_R
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            h[idx] = old_val
            R = old_R
        T *= cooling
    
    return best_h, best_score

def construct():
    """
    Construct a signed step function with 400 equal intervals to minimize
    the autoconvolution upper bound (peak sidelobe / N).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Structured initializations
    initializers = []
    
    # Thue-Morse
    initializers.append(_thue_morse(N))
    
    # Quadratic residues with several primes
    for p in [401, 409, 421, 431, 439, 443, 449, 457, 461, 463]:
        initializers.append(_quadratic_residues(N, p))
    
    # Random sequences
    for _ in range(10):
        initializers.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Evaluate all initializers
    candidates = []
    for h0 in initializers:
        R0 = _autocorrelation(h0)
        s0 = np.max(np.abs(R0[1:])) / N
        candidates.append((s0, h0.copy(), R0.copy()))
    
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # Take top K candidates for optimization
    K = 5
    
    for s0, h0, R0 in candidates[:K]:
        if time.time() > deadline - 20.0:
            break
        
        if s0 < best_score:
            best_score = s0
            best_h = h0.copy()
        
        # Gradient-based continuous optimization
        if time.time() < deadline - 30.0:
            # Initialize x such that tanh(x/temp) ≈ h0
            temperature = 0.5
            x0 = np.zeros(N)
            for i in range(N):
                if h0[i] > 0:
                    x0[i] = 2.0  # tanh(2/0.5) = tanh(4) ≈ 1
                else:
                    x0[i] = -2.0
            
            try:
                result = minimize(
                    _smoothed_objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.success or result.fun < s0:
                    x_opt = result.x
                    h_opt = np.sign(x_opt)
                    h_opt[h_opt == 0] = 1.0  # Handle zeros
                    s_opt = _score(h_opt)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # Discrete local search polish
        if time.time() < deadline - 10.0:
            h_polished, s_polished = _discrete_local_search(h0.copy(), max_iterations=3000)
            if s_polished < best_score:
                best_score = s_polished
                best_h = h_polished.copy()
        
        # Simulated annealing
        if time.time() < deadline - 5.0:
            h_sa, s_sa = _simulated_annealing(h0.copy(), max_iter=3000)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
    
    if best_h is None:
        return np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
