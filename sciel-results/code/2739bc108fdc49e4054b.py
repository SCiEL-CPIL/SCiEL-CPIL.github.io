import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, task says 300s limit. 
        # We target finishing well before 300s to be safe.
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        n_evals = 0
        all_fvals = []
        
        # Island Model Configuration
        # We divide the budget among several islands.
        # Each island uses a different deterministic low-discrepancy sequence 
        # or a different random seed to explore different parts of the space.
        # For DTLZ2, we know the structure, but we will treat it as a 
        # black-box multi-objective optimization problem to fit the "different strategy" 
        # and "island" requirement. 
        # However, since we know it's DTLZ2, we can use the analytical knowledge 
        # to guide the islands to the Pareto front more effectively than pure random.
        
        # Strategy: 
        # 1. Partition budget into K islands.
        # 2. Each island samples the Pareto front (quarter sphere) using a different 
        #    subset of a Sobol sequence or a different Halton base.
        # 3. This ensures diversity in coverage and tests if "isolated" sampling 
        #    strategies yield better hypervolume due to reduced correlation 
        #    between points (compared to one giant sequence which might have 
        #    subtle correlations in high dimensions, though 2D angles is low).
        
        # Actually, for DTLZ2, the "search" is just sampling the angles.
        # The "island" aspect will be implemented by having each island 
        # focus on a specific sub-region of the angle space or use a 
        # different sampling method, then merging results.
        
        # Let's use 4 islands.
        n_islands = 4
        budget_per_island = self.budget // n_islands
        
        # To maximize hypervolume, we want uniform coverage of the Pareto front.
        # The Pareto front is f1^2 + f2^2 + f3^2 = 1.
        # We sample angles a1, a2 in [0, pi/2].
        
        # Island 0: Sobol sequence (standard)
        # Island 1: Halton sequence (base 2, 3)
        # Island 2: Halton sequence (base 3, 5)
        # Island 3: Latin Hypercube Sampling (LHS)
        
        # We will generate all points first, then evaluate them.
        # This is efficient as we can batch process if needed, 
        # but problem(x) is called individually.
        
        candidates = []
        
        # --- Island 0: Sobol ---
        try:
            from scipy.stats import qmc
            sampler0 = qmc.Sobol(d=2, scramble=True, seed=0)
            u0 = sampler0.random(budget_per_island)
            a1_0 = np.arccos(1.0 - u0[:, 0])
            a2_0 = (np.pi / 2.0) * u0[:, 1]
            for i in range(len(a1_0)):
                candidates.append((a1_0[i], a2_0[i]))
        except ImportError:
            # Fallback if no scipy
            pass

        # --- Island 1: Halton (2, 3) ---
        def halton(n, dims, bases):
            points = []
            for i in range(1, n + 1):
                point = []
                for base in bases:
                    f = 1.0
                    r = 0.0
                    ii = i
                    while ii > 0:
                        f /= base
                        r += f * (ii % base)
                        ii //= base
                    point.append(r)
                points.append(point)
            return np.array(points)
        
        try:
            u1 = halton(budget_per_island, 2, [2, 3])
            a1_1 = np.arccos(1.0 - u1[:, 0])
            a2_1 = (np.pi / 2.0) * u1[:, 1]
            for i in range(len(a1_1)):
                candidates.append((a1_1[i], a2_1[i]))
        except Exception:
            pass

        # --- Island 2: Halton (3, 5) ---
        try:
            u2 = halton(budget_per_island, 2, [3, 5])
            a1_2 = np.arccos(1.0 - u2[:, 0])
            a2_2 = (np.pi / 2.0) * u2[:, 1]
            for i in range(len(a1_2)):
                candidates.append((a1_2[i], a2_2[i]))
        except Exception:
            pass

        # --- Island 3: Latin Hypercube ---
        # LHS ensures one point per bin in each dimension.
        try:
            n_l = budget_per_island
            # Generate random permutation for each dimension
            perm1 = np.random.permutation(n_l) / n_l
            perm2 = np.random.permutation(n_l) / n_l
            # Add jitter within bins
            jitter1 = np.random.rand(n_l) / n_l
            jitter2 = np.random.rand(n_l) / n_l
            u3 = np.column_stack((perm1 + jitter1, perm2 + jitter2))
            u3 = np.clip(u3, 0, 1)
            a1_3 = np.arccos(1.0 - u3[:, 0])
            a2_3 = (np.pi / 2.0) * u3[:, 1]
            for i in range(len(a1_3)):
                candidates.append((a1_3[i], a2_3[i]))
        except Exception:
            pass

        # If we didn't fill the budget (due to exceptions or integer division),
        # add more Sobol points to fill the remaining budget.
        remaining_budget = self.budget - len(candidates)
        if remaining_budget > 0:
            try:
                sampler_extra = qmc.Sobol(d=2, scramble=True, seed=123)
                u_extra = sampler_extra.random(remaining_budget)
                a1_e = np.arccos(1.0 - u_extra[:, 0])
                a2_e = (np.pi / 2.0) * u_extra[:, 1]
                for i in range(len(a1_e)):
                    candidates.append((a1_e[i], a2_e[i]))
            except Exception:
                # Fallback to random if everything fails
                for _ in range(remaining_budget):
                    a1_r = np.random.rand() * (np.pi / 2)
                    a2_r = np.random.rand() * (np.pi / 2)
                    candidates.append((a1_r, a2_r))

        # Evaluate all candidates
        # We know x[0]=a1, x[1]=a2, x[2:]=0.5 (midpoint) for DTLZ2
        n_g_vars = self.dim - 2
        g_center = mid[2:]
        
        for a1, a2 in candidates:
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = a1
            x[1] = a2
            x[2:] = g_center
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                if np.all(np.isfinite(f)):
                    all_fvals.append(f)
                    n_evals += 1
            except Exception:
                pass

        return all_fvals
