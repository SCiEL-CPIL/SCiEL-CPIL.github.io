import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial candidate: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # Precompute grid for fast evaluation
    M = 8192
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    # Z[k, j] = z_j^k
    Z = np.vstack([z**k for k in range(24)])
    
    def calc_score(c):
        # c is shape (24,)
        # P_j = sum c_k z_j^k
        P = c @ Z # shape (M,)
        abs_P = np.abs(P)
        # Find peaks
        # Simple approach: take max of grid, then refine top few
        # For speed in inner loop, just grid max might be enough for local search?
        # But for final score, we need accuracy.
        # For local search, grid max is a good proxy and much faster.
        # Let's use grid max for the search loop.
        return np.max(abs_P)

    def calc_score_refined(c):
        P = c @ Z
        abs_P = np.abs(P)
        # Find local maxima on grid
        peaks = []
        for i in range(M):
            if abs_P[i] > abs_P[(i-1)%M] and abs_P[i] > abs_P[(i+1)%M]:
                # Refine in neighborhood
                # Define function to maximize |P(theta)|
                # P(theta) = sum c_k exp(i k theta)
                def f(t):
                    z_t = np.exp(1j * t)
                    # Compute P at t
                    # This is slow if done naively in python loop for refinement
                    # But refinement is only for top peaks.
                    # Let's use the precomputed powers? No, t changes.
                    # We can use the fact that P is analytic.
                    # Just evaluate directly.
                    val = 0
                    for k in range(24):
                        val += c[k] * (z_t ** k)
                    return -np.abs(val)
                
                # Neighborhood width
                dt = 2*np.pi / M
                res = minimize_scalar(f, bounds=(theta[i]-dt, theta[i]+dt), method='bounded', options={'xatol': 1e-10})
                peaks.append(-res.fun)
        
        if not peaks:
            return np.max(abs_P)
        return max(peaks)

    best_score = calc_score_refined(best_coeffs)
    current_score = best_score
    current_coeffs = best_coeffs.copy()
    
    # Local Search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(24):
            # Flip bit i
            old_val = current_coeffs[i]
            current_coeffs[i] = -old_val
            
            # Evaluate score (using grid max for speed in search)
            new_score_grid = calc_score(current_coeffs)
            
            if new_score_grid < current_score - 1e-9:
                # Potential improvement, verify with refined score?
                # To save time, accept if grid score improves significantly.
                # Or just use grid score for search, and refine at the end.
                # Using grid score for search is standard.
                current_score = new_score_grid
                improved = True
            else:
                current_coeffs[i] = old_val
                
        if not improved:
            # Escape local minimum: random kick
            # Flip a random bit
            idx = np.random.randint(24)
            current_coeffs[idx] *= -1
            current_score = calc_score(current_coeffs)
            # Try to hill climb again from here
            
    # Final Polish: Refine the best found coeffs
    # The search used grid max. We need the true sup-norm for the final answer?
    # The task says "score is ... computed by dense sampling and golden-section refinement".
    # So the evaluator does this. We just need to return the coeffs.
    # But we want to ensure we found the global min.
    
    # Let's do a few restarts if time permits
    # Actually, the loop above with "kick" effectively does restarts.
    # But a cleaner restart might be better.
    
    # Let's run a more structured search:
    # 1. Hill climb from GR
    # 2. If time left, try a few random starts and hill climb.
    
    # Re-evaluate best_coeffs with refined score to be sure?
    # The evaluator will do the refinement. We just return coeffs.
    
    return [int(c) for c in best_coeffs]
