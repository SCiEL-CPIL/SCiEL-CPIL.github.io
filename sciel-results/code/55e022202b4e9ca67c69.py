"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_norm(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling (FFT) and golden section refinement for accuracy.
    Returns max |P(z)|.
    """
    n = len(coeffs)
    # Dense sampling using FFT
    # P(z_j) = sum c_k z_j^k where z_j = exp(2pi i j / M)
    # This is M * IFFT(c_padded)
    M = 4096
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:n] = coeffs
    P = M * np.fft.ifft(c_padded)
    mags = np.abs(P)
    
    # Find top K peaks
    K = 10
    if M < K:
        idx = np.arange(M)
    else:
        idx = np.argsort(mags)[-K:][::-1]
    
    max_val = 0.0
    # Golden section refinement
    # We refine around the discrete peaks
    # theta_i = 2 * pi * i / M
    # Window size: 2 * 2pi/M
    w = 2 * np.pi / M
    
    # Precompute coeffs for Horner
    # coeffs are real, z is complex
    for i in idx:
        theta_center = 2 * np.pi * i / M
        a = theta_center - w
        b = theta_center + w
        
        # Define f(theta) = |P(e^{i theta})|
        # Use Horner's rule for efficiency
        # P(z) = c_0 + z(c_1 + ... + z c_{n-1})
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            p_val = 0.0
            # Reverse coeffs for Horner
            # coeffs is c_0 ... c_{n-1}
            # Horner: ((c_{n-1} * z + c_{n-2}) * z + ...) + c_0
            for c in reversed(coeffs):
                p_val = p_val * z_val + c
            return np.abs(p_val)
        
        # Golden section search
        gr = (5 ** 0.5 - 1) / 2
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        fc = f(c_pt)
        fd = f(d_pt)
        
        # 40 iterations is enough for 1e-9 accuracy on this scale
        for _ in range(40):
            if fc < fd:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
            else:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _score(coeffs):
    """
    Computes the score: max |P(z)| / sqrt(24)
    """
    n = len(coeffs)
    norm = _eval_norm(coeffs)
    return norm / np.sqrt(n)

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    Uses Iterated Local Search (ILS) with Tabu-like perturbation.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, leaving 5s buffer for safety
    
    # Initial guess: Random or Rudin-Shapiro
    # Let's use a random start, ILS is robust
    np.random.seed(42)
    coeffs = np.random.choice([-1, 1], size=N)
    
    best_coeffs = coeffs.copy()
    best_score = _score(best_coeffs)
    
    # Local Search Function
    def local_search(current_coeffs):
        """
        Perform steepest descent local search.
        Returns (best_coeffs, best_score)
        """
        improved = True
        c = current_coeffs.copy()
        s = _score(c)
        
        while improved:
            improved = False
            best_flip_idx = -1
            best_flip_score = s
            
            # Check all single bit flips
            for i in range(N):
                # Flip bit i
                c[i] *= -1
                new_s = _score(c)
                c[i] *= -1 # Revert
                
                if new_s < best_flip_score:
                    best_flip_score = new_s
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                # Apply the best flip
                c[best_flip_idx] *= -1
                s = best_flip_score
                improved = True
                
        return c, s

    # Iterated Local Search
    # We alternate between local search and perturbation
    # Perturbation: flip 3-5 random bits
    
    while time.time() < deadline:
        # 1. Local Search from current best
        current_coeffs, current_score = local_search(best_coeffs)
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current_coeffs.copy()
        
        # 2. Perturbation
        # If we are running out of time, don't perturb, just return
        if time.time() > deadline - 5:
            break
            
        # Flip 3 random distinct bits
        perturbation_strength = 3
        indices = np.random.choice(N, perturbation_strength, replace=False)
        for i in indices:
            best_coeffs[i] *= -1
            
        # Optionally, we could start the next local search from the perturbed state
        # But the standard ILS usually starts local search from the perturbed best.
        # Here, best_coeffs is the perturbed one.
        # The loop will call local_search(best_coeffs) at the top of the next iteration.
        
    # Final check: ensure we return the best found
    # The loop ends when deadline is hit. best_coeffs holds the best found.
    
    return [int(c) for c in best_coeffs]

# EVOLVE-BLOCK-END
