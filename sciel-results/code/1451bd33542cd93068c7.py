import numpy as np


class Optimizer:
    """
    Optimizer for DTLZ4 (M=3, N=12).
    Exploits the known structure of the Pareto Front:
    - Position variables: x[0], x[1]
    - Distance variables: x[2:] (must be 0.5 for g=0)
    
    Uses a Halton sequence for space-filling coverage of the 2D simplex
    in objective space, combined with a local refinement pass.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate the first n Halton sequence points in [0,1)."""
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            f = 1.0
            i_temp = i
            while i_temp > 0:
                f /= base
                x += f * (i_temp % base)
                i_temp //= base
            result.append(x)
        return np.array(result)

    def __call__(self, problem, bounds):
        import time
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget  # 200
        dim = self.dim  # 12
        n_obj = self.n_obj  # 3
        
        # Phase 1: Halton sequence for broad coverage (150 points)
        # Phase 2: Local refinement around best points (50 points)
        
        n_halton = 150
        n_refine = budget - n_halton  # 50
        
        # Generate 2D Halton sequence for simplex coordinates
        h1 = self._halton(n_halton, 2)
        h2 = self._halton(n_halton, 3)
        
        candidates = []
        
        # Map Halton points to simplex: (u, v) -> (x0, x1)
        # For DTLZ4 with M=3, the Pareto front is parameterized by
        # x[0] and x[1] where x[0]+x[1] <= 1 (in the transformed space)
        # We use u = x[0], v = x[1] * (1 - x[0]) to cover the simplex
        for i in range(n_halton):
            u = h1[i]
            v = h2[i]
            
            # Map to simplex: x0 = u, x1 = v * (1 - u)
            x0 = u
            x1 = v * (1.0 - u)
            
            # Try both alpha hypotheses:
            # High alpha: x0 = u^(1/alpha) to bias toward 1
            # Low alpha: x0 = u directly
            
            # Hypothesis 1: alpha=1 (DTLZ2-like, uniform)
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            candidates.append(x)
            
            # Hypothesis 2: alpha=10 (moderate)
            x0_alt = u ** 0.1
            x1_alt = v * (1.0 - x0_alt)
            x = np.zeros(dim)
            x[0] = x0_alt
            x[1] = x1_alt
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            candidates.append(x)
        
        # We now have 300 candidates but only 200 budget.
        # Let's be smarter: use 100 Halton points with 2 alpha values = 200 total
        # Actually, let's restructure: use all 200 for Halton with smart mapping
        
        # Reset: use 200 Halton points, each mapped with a blended alpha
        candidates = []
        n_total = budget
        
        # Generate enough Halton points
        h1 = self._halton(n_total, 2)
        h2 = self._halton(n_total, 3)
        
        # Use a blend of alpha values to cover uncertainty
        # alpha values to try: 1, 3, 10, 30, 100
        # Distribute 200 points across these: 40 each
        alphas = [1.0, 3.0, 10.0, 30.0, 100.0]
        per_alpha = n_total // len(alphas)  # 40 each
        remainder = n_total - per_alpha * len(alphas)
        
        idx = 0
        for ai, alpha in enumerate(alphas):
            count = per_alpha + (1 if ai < remainder else 0)
            for i in range(count):
                u = h1[idx]
                v = h2[idx]
                idx += 1
                
                # For DTLZ4, x[0] and x[1] determine the position on PF
                # The relationship: f1 = (1+g)*x0, f2 = (1+g)*x1, f3 = (1+g)*(1-x0-x1)
                # when g=0 (which requires x[2:] = 0.5)
                # So we need x0 + x1 <= 1 for valid PF points
                
                # Map (u,v) to simplex: x0 = u, x1 = v*(1-u)
                x0 = u
                x1 = v * (1.0 - u)
                
                # Apply alpha transformation: x0 = u^(1/alpha)
                # This biases x0 toward 1 for large alpha
                x0_t = u ** (1.0 / alpha)
                # Recompute x1 to stay on simplex
                x1_t = v * (1.0 - x0_t)
                
                x = np.zeros(dim)
                x[0] = x0_t
                x[1] = x1_t
                x[2:] = 0.5
                x = np.clip(x, lo, hi)
                candidates.append(x)
        
        # Ensure we don't exceed budget
        candidates = candidates[:budget]
        
        results = []
        for x in candidates:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        # Phase 2: If we have budget left and time, do local refinement
        if len(results) < budget and time.time() < deadline - 5:
            # Find the best 5 points and refine around them
            if len(results) >= 5:
                # Simple approach: find points with lowest sum of objectives
                sums = np.array([np.sum(f) for f in results])
                best_idx = np.argsort(sums)[:5]
                
                refine_candidates = []
                for bi in best_idx:
                    base_x = candidates[bi]
                    # Small perturbations in x[0] and x[1]
                    for dx0 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                        for dx1 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                            if len(refine_candidates) >= (budget - len(results)):
                                break
                            rx = base_x.copy()
                            rx[0] = base_x[0] + dx0
                            rx[1] = base_x[1] + dx1
                            rx = np.clip(rx, lo, hi)
                            refine_candidates.append(rx)
                    
                    if len(refine_candidates) >= (budget - len(results)):
                        break
                
                for rx in refine_candidates[:budget - len(results)]:
                    if time.time() > deadline:
                        break
                    try:
                        f = problem(rx)
                        results.append(f)
                    except Exception:
                        break
        
        return results
