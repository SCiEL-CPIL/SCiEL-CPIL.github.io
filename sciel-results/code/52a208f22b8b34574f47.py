import numpy as np
import time
import random

def evaluate(coeffs, refine=False):
    """
    Evaluates the normalized sup norm of the polynomial P(z) = sum c_k z^k.
    coeffs: list of 24 integers in {1, -1}
    refine: if True, performs golden-section refinement of the peak.
    """
    N = 20000
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Compute P(z) efficiently
    P = np.zeros(N, dtype=complex)
    zk = np.ones(N, dtype=complex)
    for k, c in enumerate(coeffs):
        P += c * zk
        zk *= z
        
    mags = np.abs(P)
    max_mag = np.max(mags)
    
    if not refine:
        return max_mag / np.sqrt(24)
        
    # Refine the peak
    idx = np.argmax(mags)
    step = 2 * np.pi / N
    t_low = thetas[(idx - 1) % N]
    t_high = thetas[(idx + 1) % N]
    
    def neg_sq_mag(t):
        zt = np.exp(1j * t)
        pt = 0.0 + 0.0j
        ztk = 1.0 + 0.0j
        for c in coeffs:
            pt += c * ztk
            ztk *= zt
        return -np.abs(pt)**2
        
    gr = (np.sqrt(5) - 1) / 2
    a, b = t_low, t_high
    c_ = b - gr * (b - a)
    d_ = a + gr * (b - a)
    fc = neg_sq_mag(c_)
    fd = neg_sq_mag(d_)
    
    for _ in range(50):
        if fc > fd:
            b = d_
            d_ = c_
            fd = fc
            c_ = b - gr * (b - a)
            fc = neg_sq_mag(c_)
        else:
            a = c_
            c_ = d_
            fc = fd
            d_ = a + gr * (b - a)
            fd = neg_sq_mag(d_)
            
    t_opt = (a + b) / 2
    z_opt = np.exp(1j * t_opt)
    p_opt = 0.0 + 0.0j
    z_opt_k = 1.0 + 0.0j
    for c in coeffs:
        p_opt += c * z_opt_k
        z_opt_k *= z_opt
        
    max_mag_refined = np.abs(p_opt)
    return max_mag_refined / np.sqrt(24)

def local_search(coeffs, time_limit=5.0):
    """
    Performs local search by flipping single bits until no improvement is found.
    Returns the best coeffs and its discrete evaluation value.
    """
    current_coeffs = list(coeffs)
    current_val = evaluate(current_coeffs)
    improved = True
    start_time = time.time()
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        indices = list(range(24))
        random.shuffle(indices)
        for k in indices:
            new_coeffs = current_coeffs[:]
            new_coeffs[k] *= -1
            new_val = evaluate(new_coeffs)
            if new_val < current_val - 1e-12:
                current_coeffs = new_coeffs
                current_val = new_val
                improved = True
                
    return current_coeffs, current_val

def construct():
    """
    Returns the 24 coefficients of the polynomial P(z) = sum c_k z^k 
    that minimizes the sup norm on the unit circle.
    """
    # Known good starting points
    # 1. The specific polynomial found by the current program
    best_coeffs = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1]
    best_val = evaluate(best_coeffs)
    
    # 2. A few other known good candidates (from literature or random search)
    # Let's try a few random ones and the GRS (Golay-Rudin-Shapiro) like constructions
    # GRS coefficients are based on the sequence: 1, 1, 1, -1, ...
    # Actually, let's just use the current best and random restarts.
    
    OPT_VAL = 1.2583758
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # First, try to improve the starting point
    best_coeffs, best_val = local_search(best_coeffs, time_limit=10.0)
    
    # If we are already at the optimum (within tolerance), return
    if best_val <= OPT_VAL + 1e-6:
        final_val = evaluate(best_coeffs, refine=True)
        return best_coeffs

    # Random restarts
    # We can also try to use the fact that the problem is symmetric under reversal and sign flip
    # P(z) and -P(z) have the same norm.
    # P(z) and P(1/z) have the same norm (for |z|=1).
    # So we can fix c_0 = 1 and c_23 = 1 (or whatever) to reduce search space, but local search doesn't care.
    
    while time.time() < deadline:
        # Generate random coefficients
        rand_coeffs = [random.choice([1, -1]) for _ in range(24)]
        
        # Local search
        c, v = local_search(rand_coeffs, time_limit=5.0)
        
        if v < best_val - 1e-12:
            best_coeffs = c
            best_val = v
            
            # Check if we hit the global optimum
            if best_val <= OPT_VAL + 1e-6:
                break
                
    return best_coeffs
