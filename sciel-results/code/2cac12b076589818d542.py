import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Precompute grid
    M = 16384
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    Z = np.vstack([z**k for k in range(24)])
    
    def fast_score(c):
        P = c @ Z
        return np.max(np.abs(P))

    def precise_score(c):
        P = c @ Z
        abs_P = np.abs(P)
        N_peaks = 20
        top_indices = np.argsort(abs_P)[-N_peaks:]
        peaks = []
        dt = 2*np.pi / M
        coeffs = c
        powers = np.arange(24)
        
        for i in top_indices:
            def f(t):
                z_t = np.exp(1j * t)
                val = np.sum(coeffs * (z_t ** powers))
                return -np.abs(val)
            
            theta_i = theta[i]
            res = minimize_scalar(f, bounds=(theta_i - dt, theta_i + dt), 
                                   method='bounded', options={'xatol': 1e-12})
            peaks.append(-res.fun)
        return max(peaks) if peaks else np.max(abs_P)

    # Initial candidate: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    best_precise_score = precise_score(best_coeffs)
    
    # Multistart Hill Climbing
    # We will run many restarts.
    # Keep a list of the best candidates found (by fast score) to refine later.
    # Actually, just keep the single best by fast score, and maybe a few others.
    # Let's keep the best by fast score, and check precise score periodically.
    
    best_fast_score = fast_score(best_coeffs)
    
    restarts = 0
    while time.time() < deadline - 10.0: # Leave 10s for final refinement
        # Random start
        current_coeffs = np.random.choice([1, -1], size=24)
        current_score = fast_score(current_coeffs)
        
        # Hill Climb
        improved = True
        while improved:
            improved = False
            for i in range(24):
                old_val = current_coeffs[i]
                current_coeffs[i] = -old_val
                new_score = fast_score(current_coeffs)
                
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    improved = True
                else:
                    current_coeffs[i] = old_val
        
        # Check if this local minimum is better than our best
        if current_score < best_fast_score:
            best_fast_score = current_score
            best_coeffs = current_coeffs.copy()
            
            # Check precise score periodically or if significant improvement
            # To save time, only check precise if fast score improved a lot
            # Or just check every 100 restarts?
            # Let's check precise score if it's the new best.
            curr_precise = precise_score(best_coeffs)
            if curr_precise < best_precise_score:
                best_precise_score = curr_precise
                if best_precise_score < 1.2583759:
                    break # Found optimum
            
            restarts += 1

    # Final polish: Refine the best_coeffs found
    # Also, maybe try a few random perturbations of best_coeffs and hill climb them
    # to see if we can find a better precise score.
    
    candidates = [best_coeffs]
    
    # Add a few perturbed versions
    for _ in range(5):
        c = best_coeffs.copy()
        for _ in range(3):
            c[np.random.randint(24)] *= -1
        # Hill climb this perturbed version
        curr_score = fast_score(c)
        improved = True
        while improved:
            improved = False
            for i in range(24):
                old_val = c[i]
                c[i] = -old_val
                new_score = fast_score(c)
                if new_score < curr_score - 1e-9:
                    curr_score = new_score
                    improved = True
                else:
                    c[i] = old_val
        candidates.append(c)
        
    # Evaluate precise score for all candidates
    final_best_score = float('inf')
    final_best_coeffs = best_coeffs
    for c in candidates:
        s = precise_score(c)
        if s < final_best_score:
            final_best_score = s
            final_best_coeffs = c
            
    return [int(c) for c in final_best_coeffs]
