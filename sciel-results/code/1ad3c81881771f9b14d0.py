# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    best_h = np.ones(N)
    best_value = 2.0

    def evaluate(h):
        h = np.array(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        c = np.convolve(h, h)
        max_c = np.max(c)
        val = abs(2 * N * max_c / (s ** 2))
        return val

    # Zadoff-Chu sequence: z_n = exp(-j*pi*m*n*(n+1)/N) for n=0,...,N-1
    # This is a CAZAC sequence with zero periodic autocorrelation for non-zero lags.
    # We use the real part as our step heights, and optimize:
    # - m: the Zadoff-Chu index (1 <= m < N, gcd(m,N)=1)
    # - alpha: real scaling factor
    # - beta: real offset to ensure sum != 0 and optimize the objective
    # - gamma: phase rotation before taking real part
    
    def make_zadoff_chu(m, alpha, beta, gamma):
        n = np.arange(N)
        # Zadoff-Chu: exp(-j*pi*m*n*(n+1)/N)
        phase = -np.pi * m * n * (n + 1) / N
        # Apply additional phase rotation
        phase += gamma
        # Take real part, scale, and offset
        h = alpha * np.cos(phase) + beta
        return h

    def objective(params):
        m, alpha, beta, gamma = params
        # m must be integer and coprime with N for proper CAZAC properties
        m_int = int(round(m))
        # Ensure m is in valid range and coprime with N
        if m_int < 1 or m_int >= N:
            return float('inf')
        # Check gcd
        if np.gcd(m_int, N) != 1:
            return float('inf')
        h = make_zadoff_chu(m_int, alpha, beta, gamma)
        return evaluate(h)

    # Strategy: Try multiple Zadoff-Chu indices m, and for each optimize alpha, beta, gamma
    # using Nelder-Mead or other local optimization
    
    # First, find valid m values (coprime with N=400)
    # N = 400 = 2^4 * 5^2, so m must not be divisible by 2 or 5
    valid_ms = []
    for m in range(1, N):
        if np.gcd(m, N) == 1:
            valid_ms.append(m)
    
    rng = np.random.default_rng(42)
    
    # Try a subset of m values, focusing on those that might give good results
    # Sort by some heuristic or just try evenly spaced ones
    num_to_try = min(20, len(valid_ms))
    if len(valid_ms) > num_to_try:
        # Pick evenly spaced m values
        indices = np.linspace(0, len(valid_ms)-1, num_to_try, dtype=int)
        selected_ms = [valid_ms[i] for i in indices]
    else:
        selected_ms = valid_ms
    
    for m in selected_ms:
        if time.time() > deadline - 5:
            break
        
        # For this m, optimize alpha, beta, gamma
        # Initial guesses
        initial_params = [
            [m, 1.0, 1.0, 0.0],
            [m, 1.0, 0.5, np.pi/4],
            [m, 0.5, 1.0, np.pi/2],
            [m, 1.0, 1.0, np.pi],
        ]
        
        for init in initial_params:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(
                    objective,
                    init,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                if res.fun < best_value:
                    m_best, alpha_best, beta_best, gamma_best = res.x
                    m_best_int = int(round(m_best))
                    if 1 <= m_best_int < N and np.gcd(m_best_int, N) == 1:
                        best_h = make_zadoff_chu(m_best_int, alpha_best, beta_best, gamma_best)
                        best_value = res.fun
            except Exception:
                pass
    
    # If we haven't improved, try a broader search with differential evolution
    if time.time() > deadline - 10:
        return best_h
    
    # Broader DE search over a subset of m values
    # Group m values into ranges and do DE within each
    if len(valid_ms) > 0:
        # Try DE with m discretized
        def de_objective(params):
            m_idx, alpha, beta, gamma = params
            m_idx_int = int(round(m_idx))
            if m_idx_int < 0 or m_idx_int >= len(valid_ms):
                return float('inf')
            m = valid_ms[m_idx_int]
            h = make_zadoff_chu(m, alpha, beta, gamma)
            return evaluate(h)
        
        bounds_de = [
            (0, len(valid_ms)-1),  # index into valid_ms
            (0.1, 3.0),            # alpha
            (-2.0, 3.0),           # beta
            (0.0, 2*np.pi)         # gamma
        ]
        
        try:
            from scipy.optimize import differential_evolution
            res_de = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=50,
                popsize=10,
                tol=0.01,
                seed=42,
                polish=True
            )
            if res_de.fun < best_value:
                m_idx_best, alpha_best, beta_best, gamma_best = res_de.x
                m_idx_best_int = int(round(m_idx_best))
                if 0 <= m_idx_best_int < len(valid_ms):
                    m_best = valid_ms[m_idx_best_int]
                    best_h = make_zadoff_chu(m_best, alpha_best, beta_best, gamma_best)
                    best_value = res_de.fun
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
