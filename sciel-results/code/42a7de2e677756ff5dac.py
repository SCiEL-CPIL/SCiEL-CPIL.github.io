import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Global best tracking
        global_best_x = None
        global_best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, global_best_x, global_best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Clip to domain
            x_clipped = np.clip(x, -5.0, 5.0)
            f_val = func(x_clipped)
            calls += 1
            
            if f_val < global_best_f:
                global_best_f = f_val
                global_best_x = x_clipped.copy()
            
            return f_val

        rng = np.random.default_rng(42)

        # Phase 1: Lattice Search
        # Rastrigin minima are near integer coordinates.
        # The hidden shift is in [-4, 4] per dimension? No, "minimizer lies in [-4, 4]^5".
        # The function is f(x) = 10*A + sum((x_i - s_i)^2 - 10*cos(2*pi*(x_i - s_i)))
        # Wait, the standard Rastrigin is f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)).
        # With shift s, it becomes f(x) = 10*n + sum((x_i - s_i)^2 - 10*cos(2*pi*(x_i - s_i))).
        # The global minimum is at x = s.
        # Local minima are at x = s + k (integer k) approximately, but the quadratic term pushes them.
        # Actually, the local minima are near x = s + k for integers k.
        # Since s is in [-4, 4] and x is in [-5, 5], the relevant integers k are such that s+k is in [-5, 5].
        # If s is near 0, k can be -4..4. If s is near 4, k can be -8..1.
        # A dense grid of integer points in [-5, 5]^5 would be 11^5 = 161051 points. Too many.
        # However, we only have 500 calls.
        
        # Alternative: The "lattice" inspiration suggests 3^5 = 243 points.
        # Why 3? Maybe the shift is quantized? Or maybe we just sample a coarse grid?
        # Let's try a grid of points with spacing 1.0, but centered around 0?
        # Or maybe the shift is from a specific set?
        # "hidden seeded shift, rotation and offset"
        # Rotation complicates the lattice idea. If the space is rotated, the minima are NOT at integer coordinates in the input space.
        # The rotation matrix R is applied: f(x) = f_rastrigin(R(x - c) + offset)?
        # Or f(x) = f_rastrigin(Rx + shift)?
        # If there is a rotation, the periodicity is in the rotated coordinates.
        # The lattice of minima in the input space is R^T (Z^d + shift).
        # This is a general lattice, not axis-aligned.
        
        # Therefore, a simple axis-aligned grid will NOT find the minima efficiently if the rotation is arbitrary.
        # The current CMA-ES approach is more robust to rotation.
        
        # However, the inspiration says "Model the Rastrigin landscape as a lattice... perform a dense grid search over the 3^5 = 243 integer-aligned candidate positions".
        # This implies that maybe the rotation is small or aligned? Or maybe the "integer-aligned" refers to the shifted/rotated frame?
        # But we don't know the rotation.
        
        # Let's stick to a robust global optimizer but improve the initial exploration.
        # The current program uses 50 uniform points.
        # Maybe use a Sobol sequence or a better stratified sample?
        # Or use a two-stage approach:
        # 1. Coarse grid to find the basin.
        # 2. CMA-ES to refine.
        
        # Since rotation is unknown, we cannot assume axis-aligned minima.
        # But Rastrigin is separable in its native coordinates.
        # If we assume the rotation is the identity (or close to it), a grid works.
        # If the rotation is arbitrary, we need a global search.
        
        # Let's try a hybrid:
        # 1. Evaluate a set of points that cover the box well.
        # 2. Use the best point to initialize CMA-ES with a larger sigma to ensure it covers the basin.
        
        # Improvement: Use a larger initial sigma for CMA-ES if the first few evaluations don't improve much.
        # Or, restart CMA-ES if it stagnates.
        
        # Let's implement a multi-start CMA-ES.
        # 1. Generate K random starting points.
        # 2. Run a short CMA-ES from each, or just pick the best one and run a long CMA-ES.
        # Given the budget of 500, and CMA-ES lambda=10, we can run ~45 generations.
        # This is not a lot.
        
        # Strategy:
        # 1. Spend 20% of budget (100 calls) on global search (e.g., random sampling + local refinement).
        # 2. Spend 80% of budget (400 calls) on CMA-ES from the best point.
        
        # To improve global search:
        # Use a grid of 3x3x3x3x3 = 243 points? That's more than 100.
        # Use 2x2x2x2x2 = 32 points? Too coarse.
        # Use 100 random points.
        
        # Let's try to exploit the structure better.
        # If we assume the rotation is not too large, we can try a grid in the input space.
        # But to be safe, let's use a combination of:
        # 1. Center point.
        # 2. Vertices of the box.
        # 3. Random points.
        
        # Actually, the most effective improvement for CMA-ES on multimodal functions is often a better initial sigma and maybe a restart mechanism.
        
        # Let's try this:
        # 1. Evaluate 10 random points.
        # 2. Pick the best, run CMA-ES for 100 calls.
        # 3. If no improvement in last 20 calls, restart with a new random point and run CMA-ES for 100 calls.
        # 4. Repeat until budget is exhausted.
        
        # This "restart CMA-ES" strategy is robust.
        
        n_restarts = 5
        budget_per_restart = budget // n_restarts  # 100 calls each
        
        best_f = float('inf')
        best_x = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Pick a starting point
            # For the first restart, use a good global search.
            # For subsequent restarts, pick a random point far from the current best.
            
            if restart == 0:
                # Global search: 20 random points + center
                start_candidates = []
                start_candidates.append(np.zeros(dim))
                for _ in range(20):
                    start_candidates.append(rng.uniform(-5, 5, dim))
                
                vals = []
                for p in start_candidates:
                    v = evaluate(p)
                    vals.append(v)
                
                best_idx = np.argmin(vals)
                x_start = start_candidates[best_idx].copy()
                f_start = vals[best_idx]
            else:
                # Random restart
                x_start = rng.uniform(-5, 5, dim)
                f_start = evaluate(x_start)
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()
            
            # Run CMA-ES from x_start
            # Use a fixed number of evaluations for this phase
            max_evals_this_phase = budget_per_restart
            evals_used = 0
            
            # CMA-ES Parameters
            lam = 10
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)
            
            x_mean = x_start.copy()
            sigma = 1.0  # Initial sigma
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            c_sigma = (mu_eff + 2) / (dim + 4)
            c_c = (4 - mu_eff + c_sigma) / (dim + 4.3)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            best_f_phase = f_start
            best_x_phase = x_start.copy()
            
            # CMA-ES Loop
            while evals_used < max_evals_this_phase and calls < budget:
                if time.time() > deadline:
                    break
                
                # Check for convergence
                if sigma < 1e-8:
                    break
                
                # Generate population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                samples = []
                f_vals = []
                
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * L @ z
                    samples.append(y)
                
                # Evaluate all samples
                for y in samples:
                    f_val = evaluate(y)
                    f_vals.append(f_val)
                    evals_used += 1
                    
                if calls >= budget:
                    break
                    
                # Sort by fitness
                indices = np.argsort(f_vals)
                sorted_samples = [samples[i] for i in indices]
                sorted_f_vals = [f_vals[i] for i in indices]
                
                # Update best
                if sorted_f_vals[0] < best_f_phase:
                    best_f_phase = sorted_f_vals[0]
                    best_x_phase = sorted_samples[0].copy()
                    if best_f_phase < best_f:
                        best_f = best_f_phase
                        best_x = best_x_phase.copy()
                
                # Update mean
                new_x_mean = np.sum(weights[:, np.newaxis] * np.array(sorted_samples[:mu]), axis=0)
                
                # Update step size sigma
                y_sigma = (new_x_mean - x_mean) / (sigma * np.linalg.norm(L))
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls - lam + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)))
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update covariance matrix
                y_c = (new_x_mean - x_mean) / sigma
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
                
                # Rank-one update
                C = (1 + c_1 * (2 - h_sigma) - c_1) * C + c_1 * np.outer(y_c, y_c)
                
                # Rank-mu update
                delta = h_sigma * c_c * (2 - c_c)
                rank_mu_term = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (sorted_samples[i] - x_mean) / sigma
                    rank_mu_term += weights[i] * np.outer(y_i, y_i)
                
                C = C + c_mu * (delta * np.outer(p_c, p_c) + (1 - delta) * np.eye(dim) - rank_mu_term)
                
                # Symmetrize C
                C = (C + C.T) / 2
                
                # Update x_mean
                x_mean = new_x_mean
                
                # Check if C is still valid
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C = np.eye(dim) * np.max(eigvals) if np.max(eigvals) > 0 else np.eye(dim)
            
            # End of CMA-ES phase
            
        return best_x, best_f
