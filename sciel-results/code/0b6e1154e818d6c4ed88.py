# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Direct, structure-exploiting DTLZ4 optimizer.

    DTLZ4 (3 objectives) structure:
      f_i(x) = g(x) * h_i(x, z(x)),  i = 1..M
      g(x) = 100 * (n_g + sum_{i=n_g+1}^{n} ((x_i - 0.5)^alpha - 0.5^alpha))
      n_g = M - 1 = 2  (for n_obj=3)
      alpha = 100
      z(x) = [f_1/(f_1+...+f_M), ..., f_{M-1}/(...)]  (from first M variables via h)
      h_i(z, g) = z_i^(1/alpha) * (1 + g)   for the 1-objective DTLZ form... 
      Actually for DTLZ4: f_i = g * z_i^alpha? Let's use the canonical: 
      For DTLZ4: f_i(x) = g(x) * h_i(z(x), g(x)) where z comes from first M-1 vars,
      and h_i(z,g) = z_i^(1/alpha) for i in 1..M-1? No.

    Canonical DTLZ4 (Deb et al.):
      M objectives, N variables.
      g(x) = 100 * ( n_g + sum_{i=n_g+1}^{N} ((x_i - 0.5)^alpha - 0.5^alpha) ), n_g = M - 1
      f_i(x) = g(x) * h_i( z(x), g(x) ),  i = 1..M
      z(x) = [ x_1, x_2, ..., x_{M-1} ]   (M-1 variables)
      h_1(z,g) = (1 - z_{M-1}) * (1 + g)
      h_i(z,g) = (1 - z_{M-1}) * z_{M-1} * z_{M-2} * ... for i=2..M-1? 

    Let me use the explicit standard form for M=3 (n_g=2, N=12, alpha=100):
      g = 100*(x_10 + x_11)   [since (x-0.5)^alpha for x in [0,1] with alpha=100, the sum term is 
        actually ((x_10-0.5)^100 - 0.5^100) + ((x_11-0.5)^100 - 0.5^100) which ~ 0 for x near 0.5,
        but for the front we need g=0 -> x_10=x_11=0.5]
      Wait: g = 100*( n_g + sum ) with n_g=2, so g = 100*(2 + (term10 + term11)). That can't be 0.
      
    Correction: the standard DTLZ4 g is:
      g(x) = 100 * ( n_g + sum_{i=n_g+1}^{N} ( (x_i - 0.5)^alpha - 0.5^alpha ) )
    With n_g = M - 1, and the sum is over variables M..N (indices M-1..N-1 in 0-based).
    For the front, we need g = 0. But with the +n_g term, the minimum of the sum part is -n_g*0.5^alpha
    only if each term reaches its minimum of -0.5^alpha. Actually each (x_i-0.5)^alpha - 0.5^alpha
    has minimum -0.5^alpha (at x_i=0 or 1? No, at x_i where (x_i-0.5)^alpha is min = 0, giving -0.5^alpha).
    Hmm, (x-0.5)^alpha for alpha even (100) is >= 0, min 0 at x=0.5. So min of each term is -0.5^alpha.
    Sum of n_g terms at min = -n_g * 0.5^alpha. Then g = 100*(n_g - n_g*0.5^alpha) which is > 0, not 0.
    
    I'm misremembering. Let me use the form where g=0 is achievable: the common DTLZ4 definition has
    g(x) = 100*( n_g + sum_{i=n_g+1}^{N} (x_i - 0.5)^alpha * 2 - 0.5^alpha )? 

    To avoid confusion, I will NOT rely on the closed form for g. Instead I use the structural fact that
    the Pareto front of DTLZ4 corresponds to g=0, and for the front the objective values are determined
    only by the first M variables. I'll generate candidate points on the known unit-simplex-like front
    and set the trailing variables to 0.5 (which makes the g-sum term zero in the standard form where
    (0.5-0.5)^alpha = 0 and the constant is chosen so g=0). If the actual g has an offset, the evaluator's
    problem(x) will tell us, and my local refinement will correct the trailing variables.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 270  # hard safety margin (< 300s wall-clock)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        n = self.dim
        M = self.n_obj
        
        # DTLZ4: first n_g = M-1 variables feed g, remaining n - (M-1) determine front shape
        n_g = M - 1  # 2 for M=3
        n_z = n - n_g  # 10 variables for the front shape

        rng = np.random.default_rng(0)

        # g-variables (first n_g) should be at 0.5 to minimize g
        g_center = (lo[:n_g] + hi[:n_g]) / 2.0  # 0.5 vector of length n_g

        # z-variables (last n_z) determine the Pareto front shape
        z_lo = lo[n_g:]
        z_hi = hi[n_g:]

        candidates = []

        # Phase 1: Structured grid over z-block with g at center
        # Use a dense grid to capture the front shape
        grid_size = 6  # 6^10 is too many, so use sparse grid + random
        # Actually, 6^10 = 60M which is too many. Use random + structured sampling.
        
        # Random samples in z-space
        n_rand = 80
        for _ in range(n_rand):
            z = rng.random(n_z) * (z_hi - z_lo) + z_lo
            x = np.concatenate([g_center, z])
            candidates.append(x)

        # Structured: sample along edges and faces of the hypercube
        # Sample points where most z-vars are at 0 or 1, and 1-2 vary
        for i in range(n_z):
            for j in range(i+1, n_z):
                # Vary z[i] and z[j], keep others at 0 or 1
                for val_i in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    for val_j in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        z = np.zeros(n_z)
                        # Set most to 0.5 for a central slice
                        z[:] = 0.5
                        z[i] = val_i
                        z[j] = val_j
                        z = z * (z_hi - z_lo) + z_lo
                        x = np.concatenate([g_center, z])
                        candidates.append(x)

        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for x in candidates:
            key = tuple(np.round(x, 6))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(x)
        candidates = unique_candidates

        # Evaluate candidates (respect budget)
        F_list = []
        X_list = []
        used = 0
        for x in candidates:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = np.asarray(problem(x), dtype=float)
            X_list.append(x)
            F_list.append(f)
            used += 1

        X = np.array(X_list)
        F = np.array(F_list)

        # Phase 2: Refinement - focus on non-dominated points
        if len(F) > 0:
            # Find non-dominated points
            nd_idx = []
            for i in range(len(F)):
                dom = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dom = True
                        break
                if not dom:
                    nd_idx.append(i)
            if not nd_idx:
                nd_idx = list(range(len(F)))
            
            # Refine all non-dominated points
            base_X = X[nd_idx].copy()
            base_F = F[nd_idx].copy()
            refine_indices = nd_idx
        else:
            base_X = X.copy()
            base_F = F.copy()
            refine_indices = list(range(len(X)))

        # Adaptive step size refinement
        base_step = span * 0.03
        n_refine_rounds = min(10, (self.budget - used) // max(1, len(refine_indices)))
        
        for round_idx in range(n_refine_rounds):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            current_step = base_step * (0.5 ** round_idx)
            
            for idx in refine_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x0 = base_X[idx]
                # Try several perturbations and keep the best
                best_f = base_F[idx]
                best_x = x0
                
                for _ in range(3):  # 3 perturbations per point per round
                    if used >= self.budget:
                        break
                    d = current_step * rng.standard_normal(n)
                    x = np.clip(x0 + d, lo, hi)
                    f = np.asarray(problem(x), dtype=float)
                    used += 1
                    
                    # Accept if it improves any objective without worsening others too much
                    if np.all(f <= best_f + 1e-8) and np.any(f < best_f - 1e-8):
                        best_f = f
                        best_x = x
                
                base_X[idx] = best_x
                base_F[idx] = best_f

        # Merge: keep original evaluations + refined points
        # The evaluator records ALL returned objective vectors, so we need to be careful
        # Actually, looking at the contract: "Every returned objective vector is recorded"
        # But __call__ returns final_X which is ignored. The problem() calls are what matter.
        # So all evaluations made during problem() calls are recorded automatically.
        # We just need to ensure we don't exceed budget.
        
        return X  # Return value is ignored by evaluator
# EVOLVE-BLOCK-END
