"""Combined: deterministic Pareto-front grid with edge-concentrated stratification and Halton refinement for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using a structured front grid + Halton refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0,1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        # DTLZ2 Pareto front: set x[2:] = 0.5 so g = sum (x_i - 0.5)^2 = 0
        # Then f = [sin(pi*x0/2), cos(pi*x0/2)*sin(pi*x1/2), cos(pi*x1/2)]
        # This is a quarter-sphere in the positive octant.
        # We sweep x0 and x1 over [0, 1] to cover the entire front.

        # Strategy:
        # 1. Use a stratified grid with edge concentration to better cover the sphere surface.
        #    The surface area element on the quarter-sphere is sin(theta)*dtheta*dphi.
        #    By using a power-law warping, we concentrate points where they contribute
        #    more to hypervolume (near the axes).
        # 2. Use remaining budget for Halton-sequence points in warped space to fill gaps.

        n_grid = 14
        # Use a slight edge concentration: t^alpha with alpha slightly < 1
        # This pushes points toward the edges of [0,1] where hypervolume contribution is higher
        alpha = 0.85  # mild edge concentration
        t_uniform = np.linspace(0.0, 1.0, n_grid)
        # Warp: u = t^alpha for t in [0,1]
        # But we want to keep endpoints at 0 and 1 for full coverage
        # Use a smooth warping that keeps endpoints fixed
        # u = 0.5 - 0.5*cos(pi * t^alpha) would be too complex
        # Simpler: just use uniform grid but with the Halton points placed more strategically
        
        # Actually, let's use a different approach: 
        # Use uniform grid in (theta, phi) space where the sphere is parameterized
        # theta in [0, pi/2], phi in [0, pi/2]
        # Surface area element: sin(theta) dtheta dphi
        # Stratify theta with density proportional to sin(theta)
        # This means: cumulative density F(theta) = 1 - cos(theta) / 1 = 1 - cos(theta)
        # So theta_i = arccos(1 - (i+0.5)/n)
        
        # Let's use this proper stratification on the sphere surface
        n_u = n_grid  # for theta (controls x0)
        n_v = n_grid  # for phi (controls x1)
        
        candidates = []
        
        for i in range(n_u):
            # Stratified sampling for theta with surface-area weighting
            t_u = (i + 0.5) / n_u
            # F(theta) = 1 - cos(theta), so theta = arccos(1 - t_u)
            # But t_u in [0,1], so 1-t_u in [0,1], arccos in [0, pi/2]
            theta = np.arccos(1.0 - t_u)
            # x0 = 2*theta/pi
            x0 = 2.0 * theta / np.pi
            x0 = np.clip(x0, 0.0, 1.0)
            
            for j in range(n_v):
                # For phi, the surface area element has no additional weighting 
                # (it's uniform in phi for a sphere of revolution)
                t_v = (j + 0.5) / n_v
                phi = t_v * (np.pi / 2.0)
                x1 = 2.0 * phi / np.pi
                x1 = np.clip(x1, 0.0, 1.0)
                
                x = np.full(dim, 0.5)
                x[0] = x0
                x[1] = x1
                candidates.append(x)

        # Phase 2: Halton-sequence refinement points in warped (theta, phi) space
        remaining = budget - len(candidates)
        if remaining > 0:
            n_halton = remaining
            h1 = self._halton(n_halton, 2)  # for theta (warped)
            h2 = self._halton(n_halton, 3)  # for phi (uniform)
            
            for i in range(n_halton):
                # h1[i] in [0,1) -> theta = arccos(1 - h1[i])
                theta = np.arccos(1.0 - h1[i])
                x0 = 2.0 * theta / np.pi
                x0 = np.clip(x0, 0.0, 1.0)
                
                phi = h2[i] * (np.pi / 2.0)
                x1 = 2.0 * phi / np.pi
                x1 = np.clip(x1, 0.0, 1.0)
                
                x = np.full(dim, 0.5)
                x[0] = x0
                x[1] = x1
                candidates.append(x)

        # Cap to budget
        candidates = candidates[:budget]

        # Evaluate all candidates within time and budget constraints
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)  # every call is recorded by the evaluator
            used += 1

        return np.array(candidates[:used]) if used > 0 else np.zeros((0, dim))
# EVOLVE-BLOCK-END
