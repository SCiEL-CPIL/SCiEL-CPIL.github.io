"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) using a two-phase strategy:
    Phase 1: Sobol-based global exploration for early broad coverage.
    Phase 2: Nelder-Mead local refinement on the best candidate.
    
    The key insight for f13 is that it has a very narrow ridge/valley structure.
    We use a deterministic low-discrepancy sequence for initial exploration to
    maximize early progress (AOCC), then focus all remaining budget on refining
    the single best candidate found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_point(self, index, dim):
        """Generate a point in [0,1)^dim using a simple Sobol-like sequence."""
        # Using a simple bit-reversal based approach for deterministic low-discrepancy points
        # This is a simplified version; for dim=5 this works well enough
        point = np.zeros(dim)
        for d in range(dim):
            # Use different bases for each dimension to decorrelate
            base = 2  # Binary Sobol-like
            # Simple Van der Corput sequence with different shifts per dimension
            n = index + 1
            result = 0.0
            denom = 1.0
            while n > 0:
                result += (n % 2) * denom
                n //= 2
                denom /= 2
            # Apply a dimension-specific rotation/shift for decorrelation
            point[d] = (result + d * 0.1) % 1.0
        return point

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Phase 1: Global exploration with Sobol-like sequence
        # Use about 20-30 points for broad coverage
        n_explore = min(25, max(10, budget // 20))
        
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Generate Sobol-like point in [0,1)^dim
            p_unit = self._sobol_point(i, dim)
            # Map to [-5, 5]^dim
            p = p_unit * 10.0 - 5.0
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except Exception:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
        
        # Fallback if no valid point was found
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = np.inf
            calls_made += 1
        
        # Phase 2: Nelder-Mead refinement on the best candidate
        if calls_made < budget and time.time() < deadline:
            try:
                best_x, best_f, calls_used = self._nelder_mead(
                    func, best_x, best_f, calls_made, budget, deadline
                )
                calls_made += calls_used
            except Exception:
                pass
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, calls_made, budget, deadline):
        """
        Nelder-Mead implementation for local refinement.
        Returns (best_x, best_f, calls_used)
        """
        dim = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex with small initial step size
        delta = 0.3
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            xi = x0.copy()
            xi[i] += delta
            xi = np.clip(xi, -5.0, 5.0)
            
            try:
                fi = func(xi)
            except Exception:
                fi = np.inf
            calls_made += 1
            simplex.append(xi)
            simplex_f.append(fi)
        
        if len(simplex) < dim + 1:
            return x0, f0, 0
        
        calls_in_nm = 0
        max_iter = 1000
        no_improve_count = 0
        best_f_so_far = f0
        
        for _ in range(max_iter):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Sort simplex by fitness
            sorted_indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in sorted_indices]
            simplex_f = [simplex_f[i] for i in sorted_indices]
            
            x_best = simplex[0]
            f_best = simplex_f[0]
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            
            # Check for convergence
            if abs(f_worst - f_best) < 1e-10 * (1 + abs(f_best)):
                break
            
            # Centroid of all but worst
            x_c = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_c + alpha * (x_c - x_worst)
            x_r = np.clip(x_r, -5.0, 5.0)
            
            if time.time() > deadline or calls_made >= budget:
                break
                
            try:
                f_r = func(x_r)
            except Exception:
                f_r = np.inf
            calls_made += 1
            calls_in_nm += 1
            
            if f_r < f_best:
                # Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5.0, 5.0)
                
                if time.time() > deadline or calls_made >= budget:
                    break
                    
                try:
                    f_e = func(x_e)
                except Exception:
                    f_e = np.inf
                calls_made += 1
                calls_in_nm += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Inside contraction
                    x_c_i = x_c + rho * (x_worst - x_c)
                    x_c_i = np.clip(x_c_i, -5.0, 5.0)
                    
                    if time.time() > deadline or calls_made >= budget:
                        break
                        
                    try:
                        f_c = func(x_c_i)
                    except Exception:
                        f_c = np.inf
                    calls_made += 1
                    calls_in_nm += 1
                    
                    if f_c < f_r:
                        simplex[-1] = x_c_i
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            try:
                                f_s = func(simplex[i])
                            except Exception:
                                f_s = np.inf
                            calls_made += 1
                            calls_in_nm += 1
                            simplex_f[i] = f_s
                else:
                    # Outside contraction
                    x_c_o = x_c - rho * (x_worst - x_c)
                    x_c_o = np.clip(x_c_o, -5.0, 5.0)
                    
                    if time.time() > deadline or calls_made >= budget:
                        break
                        
                    try:
                        f_c = func(x_c_o)
                    except Exception:
                        f_c = np.inf
                    calls_made += 1
                    calls_in_nm += 1
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c_o
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if time.time() > deadline or calls_made >= budget:
                                break
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            try:
                                f_s = func(simplex[i])
                            except Exception:
                                f_s = np.inf
                            calls_made += 1
                            calls_in_nm += 1
                            simplex_f[i] = f_s
            
            # Track improvement
            if simplex_f[0] < best_f_so_far - 1e-12:
                best_f_so_far = simplex_f[0]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early termination if no improvement for many iterations
            if no_improve_count > 50:
                break
        
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx], simplex_f[best_idx], calls_in_nm

# EVOLVE-BLOCK-END
