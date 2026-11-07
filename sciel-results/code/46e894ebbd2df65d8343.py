"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Surrogate-guided search with L-BFGS-B refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(123)

        # ================================================================
        # Phase 1: Sobol space-filling initialization for strong early AOCC
        # ================================================================
        # Use Sobol quasi-random sequence for uniform coverage
        sobol = qmc.Sobol(d=dim, scramble=True, seed=123)
        
        # Generate 128 Sobol points (2^7) mapped to [-4.5, 4.5]^5
        sobol_pts = sobol.random(128) * 9.0 - 4.5
        sobol_pts = np.clip(sobol_pts, -5.0, 5.0)
        
        # Also add some structured points for very early progress
        structured_pts = [
            np.zeros(dim),  # center
            np.array([3.0]*dim),
            np.array([-3.0]*dim),
            np.array([0.0]*dim),
        ]
        # Add a few more structured points
        for _ in range(6):
            pt = rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0], dim)
            structured_pts.append(pt)
        
        # Combine: structured first (for immediate early progress), then Sobol
        all_initial = np.vstack([np.array(structured_pts), sobol_pts])
        
        # Reserve budget for local refinement
        reserve_for_local = 150
        max_initial_evals = min(len(all_initial), budget - reserve_for_local)
        
        for pt in all_initial[:max_initial_evals]:
            if time.time() > deadline:
                break
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # ================================================================
        # Phase 2: Surrogate-guided adaptive refinement
        # Use inverse-distance weighted model to select next points
        # ================================================================
        
        # Track all evaluated points for surrogate model
        eval_points = []
        eval_values = []
        
        # Reconstruct from what we've seen (we didn't track, so re-evaluate a subset)
        # Actually, let's just track going forward. For now, use the best as seed.
        
        # Run L-BFGS-B from best point and diverse neighbors
        start_points = [best_x.copy()]
        
        # Add points around the best at multiple scales
        for scale in [2.0, 1.0, 0.5, 0.2]:
            for _ in range(3):
                sp = best_x + rng.normal(0, scale, dim)
                sp = np.clip(sp, -5.0, 5.0)
                start_points.append(sp)
        
        # Add diverse random points
        for _ in range(5):
            sp = rng.uniform(-4.5, 4.5, dim)
            start_points.append(sp)
        
        # Remove duplicates
        unique_starts = []
        for sp in start_points:
            is_dup = False
            for us in unique_starts:
                if np.linalg.norm(sp - us) < 0.15:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(sp)
        
        # Run L-BFGS-B from each starting point
        for sp in unique_starts:
            if eval_count >= budget - 20:
                break
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 25, 'maxls': 12, 'ftol': 1e-11, 'gtol': 1e-7}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # ================================================================
        # Phase 3: Surrogate-guided global search with adaptive refinement
        # Use inverse-distance weighted (IDW) model to find promising regions
        # ================================================================
        
        # Collect evaluated points (approximate: use best and its neighborhood)
        # Build a small set of reference points for the surrogate
        ref_points = [best_x.copy()]
        for scale in [1.5, 0.8, 0.4]:
            for _ in range(4):
                rp = best_x + rng.normal(0, scale, dim)
                rp = np.clip(rp, -5.0, 5.0)
                ref_points.append(rp)
        
        # Evaluate reference points to build surrogate
        ref_values = []
        for rp in ref_points:
            if eval_count >= budget - 30:
                break
            if time.time() > deadline:
                break
            f_val = wrapped_func(rp)
            ref_values.append(f_val)
            update_best(rp, f_val)
        
        if len(ref_points) > 5:
            ref_points = np.array(ref_points[:len(ref_values)])
            ref_values = np.array(ref_values)
            
            # Use IDW to predict uncertainty and find next points
            # IDW: f(x) = sum(w_i * f_i) / sum(w_i), w_i = 1 / (dist_i^2 + eps)
            # Uncertainty ~ variance of predictions from nearby points
            
            for iteration in range(10):
                if eval_count >= budget - 10:
                    break
                if time.time() > deadline:
                    break
                
                # Generate candidate points
                candidates = []
                # Random points
                for _ in range(20):
                    cp = rng.uniform(-4.8, 4.8, dim)
                    candidates.append(cp)
                # Points near current best
                for _ in range(10):
                    cp = best_x + rng.normal(0, 0.3, dim)
                    cp = np.clip(cp, -5.0, 5.0)
                    candidates.append(cp)
                
                # Score candidates by IDW-predicted value + uncertainty
                best_candidate = None
                best_score = float('inf')
                
                for cp in candidates:
                    dists = np.linalg.norm(ref_points - cp, axis=1)
                    weights = 1.0 / (dists**2 + 0.1)
                    
                    # Predicted value
                    pred_val = np.sum(weights * ref_values) / np.sum(weights)
                    
                    # Uncertainty: high when far from reference points
                    # Use inverse of sum of weights as uncertainty proxy
                    uncertainty = 1.0 / (np.sum(weights) + 1e-6)
                    
                    # Combined score: prefer low predicted value and high uncertainty
                    # But don't go too far from known good regions
                    score = pred_val + 0.5 * uncertainty * (best_f - pred_val)
                    
                    if score < best_score:
                        best_score = score
                        best_candidate = cp
                
                if best_candidate is not None:
                    f_val = wrapped_func(best_candidate)
                    update_best(best_candidate, f_val)
                    
                    # Add to reference set if it's informative
                    if f_val < best_f + 5.0:
                        ref_points = np.vstack([ref_points, best_candidate])
                        ref_values = np.append(ref_values, f_val)
        
        # ================================================================
        # Phase 4: Final high-precision polish
        # ================================================================
        if eval_count < budget - 5 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # ================================================================
        # Phase 5: Fill remaining budget with fine-grained search
        # ================================================================
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.03, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
