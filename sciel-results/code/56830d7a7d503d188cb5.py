import numpy as np
import time
import random

def _eval_norm_fast(coeffs):
    """
    Fast sup-norm evaluation using vectorized numpy operations.
    coeffs: array of 24 values in {+1, -1}
    Returns: max |P(z)| for |z|=1
    """
    n = len(coeffs)
    N = 4096
    k = np.arange(n)
    theta = 2 * np.pi * np.arange(N) / N
    # P(z_j) = sum_k coeffs[k] * exp(i*k*theta_j)
    # Use matrix multiplication for speed
    exponents = np.exp(1j * np.outer(k, theta))  # shape (n, N)
    P = coeffs @ exponents  # shape (N,)
    mags = np.abs(P)
    
    # Find top peaks and refine
    top_idx = np.argsort(mags)[-5:]
    max_val = 0.0
    
    for i in top_idx:
        theta0 = theta[i]
        # Golden section refinement around theta0
        a = theta0 - 2 * np.pi / N
        b = theta0 + 2 * np.pi / N
        
        def f(t):
            z = np.exp(1j * t)
            # Compute P(z) = sum c_k z^k
            powers = z ** k
            val = np.dot(coeffs, powers)
            return abs(val)
        
        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(40):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
        
        peak = max(fc, fd)
        if peak > max_val:
            max_val = peak
    
    return max_val

def _simulated_annealing(initial_coeffs, time_budget=30.0):
    """
    Simulated annealing to minimize sup norm.
    """
    n = len(initial_coeffs)
    current = initial_coeffs.copy()
    current_val = _eval_norm_fast(current)
    
    best = current.copy()
    best_val = current_val
    
    start_time = time.time()
    
    # Cooling schedule
    T0 = 0.5  # Initial temperature (in units of sup norm difference)
    T_end = 0.01
    alpha = 0.999  # Cooling rate per iteration
    
    T = T0
    
    while time.time() - start_time < time_budget:
        # Propose a move: flip one or two random coefficients
        if random.random() < 0.7:
            # Single flip
            idx = random.randint(0, n - 1)
            new_coeffs = current.copy()
            new_coeffs[idx] *= -1
        else:
            # Double flip
            i1 = random.randint(0, n - 1)
            i2 = random.randint(0, n - 1)
            if i1 == i2:
                i2 = (i2 + 1) % n
            new_coeffs = current.copy()
            new_coeffs[i1] *= -1
            new_coeffs[i2] *= -1
        
        new_val = _eval_norm_fast(new_coeffs)
        delta = new_val - current_val
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current = new_coeffs
            current_val = new_val
            
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best, best_val

def construct():
    n = 24
    best_coeffs = None
    best_score = float('inf')
    
    start_time = time.time()
    time_limit = 115.0
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All +1
    candidates.append(np.ones(n))
    
    # 2. Alternating
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(n)]))
    
    # 3. GRS-like sequences
    # Generate using recurrence similar to GRS
    p, q = [1], [1]
    while len(p) < n + 10:
        p_new = p + q
        q_new = p + [-c for c in q]
        p, q = p_new, q_new
    candidates.append(np.array(p[:n], dtype=float))
    candidates.append(np.array(p[:n][::-1], dtype=float))
    
    # 4. Random candidates
    random.seed(12345)
    for _ in range(8):
        candidates.append(np.array([1 if random.random() < 0.5 else -1 for _ in range(n)], dtype=float))
    
    # 5. Candidates with specific structures
    # First half +1, second half -1
    c1 = np.ones(n)
    c1[n//2:] = -1
    candidates.append(c1)
    
    # First quarter +1, next quarter -1, etc.
    c2 = np.ones(n)
    c2[n//4:n//2] = -1
    c2[3*n//4:] = -1
    candidates.append(c2)
    
    # Run simulated annealing on each candidate
    for i, cand in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time based on remaining budget
        remaining = time_limit - (time.time() - start_time)
        n_remaining = len(candidates) - i
        time_per = max(remaining / n_remaining, 1.0)
        
        res_coeffs, res_score = _simulated_annealing(cand.copy(), time_budget=time_per)
        
        if res_score < best_score:
            best_score = res_score
            best_coeffs = res_coeffs
    
    if best_coeffs is None:
        best_coeffs = np.ones(n)
    
    return [int(c) for c in best_coeffs]
