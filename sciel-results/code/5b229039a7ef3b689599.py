"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _halton_point(dim, n, i):
    """
    Generate a deterministic point in [0, 1)^dim using the Halton sequence.
    This ensures diverse, broad coverage without needing SciPy's QMC.
    Uses prime bases for each dimension.
    """
    # First 10 primes for up to 10 dimensions
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    point = np.zeros(dim)
    for d in range(dim):
        base = primes[d % len(primes)]
        val = 0.0
        f = 1.0 / base
        n_val = i + 1  # 1-based index for sequence
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """
    Global-to-Local Cascade Optimizer.
    Phase 1: Broad coverage using Halton sequence points to find good basins early (AOCC boost).
    Phase 2: Refine the best candidates using a custom Nelder-Mead implementation.
    Nelder-Mead is chosen over BFGS for its robustness on non-smooth or "sharp" functions
    and because it does not require gradient estimates, saving function calls.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # We will use a portion of the budget for global exploration (Phase 1)
        # and the rest for local exploitation (Phase 2).
        # For dim=5, Nelder-Mead is efficient. Let's use ~30 points for exploration
        # to save budget for the refinement phase which is crucial for final score.
        n_explore = 30
        # Ensure we don't exceed budget
        n_explore = min(n_explore, budget - 1)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = [] # List of (f, x) for top candidates

        # Phase 1: Global Exploration with Halton Sequence
        # Map [0, 1)^dim to [-5, 5]^dim
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Generate Halton point
            p_unit = _halton_point(dim, n_explore, i)
            p = p_unit * 10.0 - 5.0
            
            # Clip to box
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            # Keep track of top candidates for multi-start refinement
            candidates.append((f, p))
        
        if best_x is None:
            # Fallback if we ran out of time immediately (unlikely)
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf
            calls_made += 1
            candidates.append((best_f, best_x))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with Custom Nelder-Mead
        # Select top K candidates to refine.
        # Nelder-Mead is memory efficient and doesn't need gradients.
        # For dim=5, it's very fast.
        
        max_starts = 3
        starts_to_try = candidates[:max_starts]
        
        # To avoid redundant work, we can skip starts that are very close to each other.
        # Simple check: if distance < 0.1, skip.
        unique_starts = []
        for f_init, x_init in starts_to_try:
            if time.time() > deadline or calls_made >= budget:
                break
            
            is_unique = True
            for _, x_prev in unique_starts:
                if np.linalg.norm(x_init - x_prev) < 0.5:
                    is_unique = False
                    break
            
            if is_unique:
                unique_starts.append((f_init, x_init))
        
        for f_init, x_init in unique_starts:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Run Nelder-Mead
            try:
                x_opt, f_opt, calls_used = self._nelder_mead(func, x_init, f_init, calls_made, budget, deadline)
                calls_made += calls_used
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt
            except Exception:
                pass

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, calls_made, budget, deadline):
        """
        Custom Nelder-Mead implementation.
        Returns (best_x, best_f, calls_used)
        """
        dim = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        # x0 is the best point.
        # Other vertices are x0 +/- delta * e_i
        delta = 0.5
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            
            if time.time() > deadline or calls_made >= budget:
                break
                
            try:
                fi = func(xi)
            except:
                fi = np.inf
            calls_made += 1
            simplex.append(xi)
            simplex_f.append(fi)
        
        if len(simplex) < dim + 1:
            return x0, f0, 0

        calls_in_nm = 0
        max_iter = 500 # Safety cap on iterations
        
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
            
            # Centroid of all but worst
            x_c = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = x_c + alpha * (x_c - x_worst)
            x_r = np.clip(x_r, -5, 5)
            
            if time.time() > deadline or calls_made >= budget:
                break
                
            try:
                f_r = func(x_r)
            except:
                f_r = np.inf
            calls_made += 1
            calls_in_nm += 1
            
            if f_r < f_best:
                # 2. Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                
                if time.time() > deadline or calls_made >= budget:
                    break
                    
                try:
                    f_e = func(x_e)
                except:
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
                # 3. Contraction
                if f_r < f_worst:
                    # Inside contraction
                    x_c_i = x_c + rho * (x_worst - x_c)
                    x_c_i = np.clip(x_c_i, -5, 5)
                    
                    if time.time() > deadline or calls_made >= budget:
                        break
                        
                    try:
                        f_c = func(x_c_i)
                    except:
                        f_c = np.inf
                    calls_made += 1
                    calls_in_nm += 1
                    
                    if f_c < f_r:
                        simplex[-1] = x_c_i
                        simplex_f[-1] = f_c
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if time.time() > deadline or calls_made >= budget:
                                break
                                
                            try:
                                f_s = func(simplex[i])
                            except:
                                f_s = np.inf
                            calls_made += 1
                            calls_in_nm += 1
                            simplex_f[i] = f_s
                else:
                    # Outside contraction
                    x_c_o = x_c - rho * (x_worst - x_c)
                    x_c_o = np.clip(x_c_o, -5, 5)
                    
                    if time.time() > deadline or calls_made >= budget:
                        break
                        
                    try:
                        f_c = func(x_c_o)
                    except:
                        f_c = np.inf
                    calls_made += 1
                    calls_in_nm += 1
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c_o
                        simplex_f[-1] = f_c
                    else:
                        # 4. Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if time.time() > deadline or calls_made >= budget:
                                break
                                
                            try:
                                f_s = func(simplex[i])
                            except:
                                f_s = np.inf
                            calls_made += 1
                            calls_in_nm += 1
                            simplex_f[i] = f_s
        
        # Return best point in simplex
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx], simplex_f[best_idx], calls_in_nm

# EVOLVE-BLOCK-END
