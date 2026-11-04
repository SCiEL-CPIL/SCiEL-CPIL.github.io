import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def get_correlations_linear(h):
        """
        Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
        Returns array of C_k for k != 0.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is zero-shift. We exclude it.
        # c[0:N-1] are negative shifts, c[N:2N-1] are positive shifts.
        return np.concatenate([c[:N-1], c[N:]])

    def max_overlap(h):
        """Return the maximum overlap value."""
        c = get_correlations_linear(h)
        if len(c) == 0:
            return 0.0
        return np.max(c)

    def project(h):
        """
        Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
        Uses iterative clipping and adjustment.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            diff = TARGET_SUM - np.sum(h)
            if abs(diff) < 1e-12:
                break
            # Adjust only those not at bounds if possible, but simple uniform adjustment + clip is robust enough for initialization
            # However, for optimization, we rely on L-BFGS-B bounds and a penalty term.
            # This projection is mainly for initial candidates and final cleanup.
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h

    def objective_lse(x, beta, lam_sum):
        """
        Objective function using LogSumExp approximation of the max overlap.
        x is the vector of heights.
        """
        h = np.clip(x, 0.0, 1.0)
        overlaps = get_correlations_linear(h)
        
        if len(overlaps) == 0:
            max_ov = 0.0
            lse = 0.0
        else:
            max_ov = np.max(overlaps)
            # Shift for numerical stability
            shifted = overlaps - max_ov
            # Avoid overflow in exp
            # If beta is large, shifted * beta might be large negative, which is fine.
            # If shifted is 0, exp(0)=1.
            # We sum exp(beta * shifted).
            # To avoid underflow/overflow issues, we can just use the formula directly.
            # np.logsumexp is available in scipy.special but let's use manual to be safe or import it.
            # scipy.special.logsumexp is robust.
            try:
                from scipy.special import logsumexp
                lse = max_ov + logsumexp(beta * shifted) / beta
            except ImportError:
                # Fallback if scipy.special is not behaving as expected (unlikely)
                # max_ov + (1/beta) * log(sum(exp(beta*(overlaps - max_ov))))
                exp_vals = np.exp(beta * shifted)
                lse = max_ov + np.log(np.sum(exp_vals)) / beta
        
        # Penalty for sum constraint
        # L-BFGS-B handles bounds, but sum constraint is linear equality.
        # We add a quadratic penalty to the objective.
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return lse + sum_penalty

    def run_optimization(x0, beta, lam_sum, maxiter=200):
        """
        Run L-BFGS-B optimization from x0 with given parameters.
        """
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def obj_func(x):
            return objective_lse(x, beta, lam_sum)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x
        except Exception:
            return x0

    # --- Initialization ---
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating binary-like (smoothed)
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # 3. Sinusoidal patterns
    t = np.arange(N)
    for freq in range(1, 10):
        for amp in [0.1, 0.2, 0.3]:
            for phase in [0, np.pi/2]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
                
    # 4. Random
    for _ in range(20):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
        
    # Evaluate and select best initial candidate
    best_h = None
    best_score = np.inf
    
    for h_cand in candidates:
        h_proj = project(h_cand)
        score = max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
            
    # --- Optimization Phases ---
    # We increase beta to sharpen the max approximation.
    # We also increase the sum penalty lambda to enforce constraint more strictly as we converge.
    
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    current_h = best_h.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
            
        # Increase lambda as beta increases to keep sum constraint tight
        # Early on, smooth objective is more important. Late on, exact constraint is critical.
        lam_sum = 10.0 * (beta / 5.0)
        
        # Run optimization from current best
        new_h = run_optimization(current_h, beta, lam_sum, maxiter=300)
        
        # Project to ensure strict feasibility for evaluation
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
            current_h = new_h_proj
        else:
            # If it didn't improve, keep current_h but maybe try a random restart?
            # Given time constraints, we stick to the main trajectory to be safe.
            # However, if we are stuck, a small perturbation might help.
            # Let's just keep the previous best_h as the starting point for next beta.
            pass

    # --- Final Polish ---
    # Use very high beta and high lambda for final precision
    if time.time() < deadline - 5.0:
        beta_final = 500
        lam_final = 1000.0
        new_h = run_optimization(best_h, beta_final, lam_final, maxiter=500)
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj

    # Final Validation
    best_h = project(best_h)
    
    # Ensure strict constraints
    # The projection should handle this, but let's double check
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "h out of bounds"
    # Sum constraint might be slightly off due to floating point in projection loop
    # Re-project one more time with tighter tolerance if needed
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-9:
        # Simple adjustment
        # Find indices not at 0 or 1 to adjust
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.any(mask):
            adj = diff / np.sum(mask)
            best_h[mask] += adj
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
