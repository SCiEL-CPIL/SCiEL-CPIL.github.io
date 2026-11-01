<<<<<<< SEARCH
def _soft_expansion_refine(z, time_budget):
=======
def _high_precision_hill_climb(x, y, r, w, h, time_budget):
    """
    High-precision hill climbing with random perturbations to escape local minima.
    Attempts to increase sum(r) by moving circles and adjusting radii.
    """
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    best_val = np.sum(r)
    
    # Use a fixed seed for reproducibility within the run if needed, or just default
    rng = np.random.default_rng(12345)
    
    max_iter = 5000
    temp = 0.01
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() > time_budget:
            break
            
        # Randomly select a subset of circles to perturb (e.g., 1-3 circles)
        num_perturb = rng.integers(1, 4)
        indices = rng.choice(n, num_perturb, replace=False)
        
        x_trial = x.copy()
        y_trial = y.copy()
        r_trial = r.copy()
        
        # Perturb positions
        for idx in indices:
            # Gaussian perturbation
            sigma = 0.01 * temp * 100 # Scale temp to reasonable sigma
            x_trial[idx] += rng.normal(0, sigma)
            y_trial[idx] += rng.normal(0, sigma)
            
            # Clip to box (approximate, will be checked in validity)
            x_trial[idx] = np.clip(x_trial[idx], 0, w)
            y_trial[idx] = np.clip(y_trial[idx], 0, h)
            
        # Try to expand radii for the perturbed circles (and maybe others if space allowed)
        # For simplicity, just try to expand the perturbed ones to max possible
        for idx in indices:
            r_wall = min(x_trial[idx], w - x_trial[idx], y_trial[idx], h - y_trial[idx])
            r_max = r_wall
            for j in range(n):
                if j == idx:
                    continue
                dx = x_trial[idx] - x_trial[j]
                dy = y_trial[idx] - y_trial[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r_trial[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_trial[idx] = min(r_trial[idx] * 1.01, max(0.0, r_max)) # Slight growth attempt
            
        # Check validity
        if _is_valid(x_trial, y_trial, r_trial, w, h):
            val_trial = np.sum(r_trial)
            if val_trial > best_val:
                x = x_trial
                y = y_trial
                r = r_trial
                best_val = val_trial
                temp *= cooling
            else:
                # Accept worse with small probability (Simulated Annealing style)
                if rng.random() < np.exp((val_trial - best_val) / (temp * 0.1)):
                    x = x_trial
                    y = y_trial
                    r = r_trial
                    best_val = val_trial
                    temp *= cooling
        else:
            # Penalty for invalidity in acceptance criteria? 
            # For now, just reject invalid.
            pass
            
    return x, y, r

def _soft_expansion_refine(z, time_budget):
>>>>>>> REPLACE

