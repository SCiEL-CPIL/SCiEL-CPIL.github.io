"""Differential Evolution + Nelder-Mead hybrid optimizer for BBOB f19.

Uses DE for global exploration with LHS initialization, then local refinement
with Nelder-Mead. Different optimization family than CMA-ES; DE is robust to
rotation and avoids covariance matrix issues.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """DE + Nelder-Mead hybrid optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs_sample(self, n_points, n_dims, rng, bounds=(-5.0, 5.0)):
        """Latin Hypercube Sampling for diverse initialization."""
        samples = np.zeros((n_points, n_dims))
        for d in range(n_dims):
            perm = rng.permutation(n_points)
            for i in range(n_points):
                frac = (perm[i] + rng.random()) / n_points
                samples[i, d] = bounds[0] + frac * (bounds[1] - bounds[0])
        return samples

    def _nelder_mead(self, x0, f0, budget, deadline, initial_simplex=None):
        """Nelder-Mead local refinement."""
        n = self.dim
        f = self._f
        
        # Build initial simplex
        if initial_simplex is None:
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] += 0.1
        else:
            simplex = initial_simplex.copy()
        
        # Evaluate initial simplex
        fvals = np.empty(n + 1)
        fvals[0] = f0
        for i in range(1, n + 1):
            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
            fvals[i] = f(simplex[i])
        
        calls = n  # n additional evaluations for simplex
        
        alpha, gamma, rho = 1.0, 2.0, 0.5
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = fvals.min()
        
        it = 0
        max_it = budget
        while it < max_it and time.time() < deadline:
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Track best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
            
            # Centroid of all but worst
            centroid = simplex[:-1].mean(axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = f(x_ref)
            calls += 1
            
            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = f(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = f(x_con)
                calls += 1
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = f(simplex[i])
                    calls += n
            
            it += 1
            if calls >= budget:
                break
        
        return best_x, best_f, calls

    def _de_optimize(self, x0, f0, budget, deadline, pop_size=20, F=0.8, CR=0.9):
        """Differential Evolution optimizer."""
        n = self.dim
        f = self._f
        
        # Initialize population around x0 with some spread
        pop = np.zeros((pop_size, n))
        pop_fvals = np.empty(pop_size)
        
        # First individual is x0
        pop[0] = x0
        pop_fvals[0] = f0
        
        # Generate rest with LHS-like diversity
        remaining = pop_size - 1
        for i in range(1, pop_size):
            # Mix of local perturbations and global exploration
            if i < remaining // 2:
                # Local: small perturbation around x0
                pop[i] = x0 + 1.0 * rng.standard_normal(n)
            else:
                # Global: random point in box
                pop[i] = rng.uniform(-5.0, 5.0, n)
            pop[i] = np.clip(pop[i], -5.0, 5.0)
            pop_fvals[i] = f(pop[i])
        
        calls = pop_size - 1  # already evaluated x0
        
        best_x = x0.copy()
        best_f = f0
        
        # Track best in population
        for i in range(pop_size):
            if pop_fvals[i] < best_f:
                best_f = pop_fvals[i]
                best_x = pop[i].copy()
        
        gen = 0
        max_gen = budget
        no_improve = 0
        
        while gen < max_gen and time.time() < deadline:
            # Sort population
            idx = np.argsort(pop_fvals)
            pop = pop[idx]
            pop_fvals = pop_fvals[idx]
            
            # Update best
            if pop_fvals[0] < best_f:
                best_f = pop_fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # Generate new population
            new_pop = [pop[0].copy()]  # elitism
            new_fvals = [pop_fvals[0]]
            
            for i in range(1, pop_size):
                # DE/rand/1
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                while r1 == r2 or r1 == r3 or r2 == r3:
                    r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                v = np.clip(v, -5.0, 5.0)
                
                # Crossover
                jrand = rng.integers(n)
                u = rng.random(n)
                trial = np.where(u <= CR, v, pop[i])
                trial[jrand] = v[jrand]
                trial = np.clip(trial, -5.0, 5.0)
                
                f_trial = f(trial)
                calls += 1
                
                if f_trial <= pop_fvals[i]:
                    new_pop.append(trial)
                    new_fvals.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_fvals.append(pop_fvals[i])
            
            pop = np.array(new_pop)
            pop_fvals = np.array(new_fvals)
            
            gen += 1
            
            # Early stopping
            if no_improve > 10:
                break
            
            # Check budget
            if calls >= budget:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(42)
        self._f = func
        n = self.dim
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Phase 1: LHS-based global initialization (20 points)
        n_init = 20
        lhs_samples = self._lhs_sample(n_init, n, rng)
        
        for i in range(n_init):
            if time.time() > deadline - 30:
                break
            xp = lhs_samples[i]
            fp = float(func(xp))
            calls += 1
            if fp < best_f:
                best_f = fp
                best_x = xp.copy()
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))
            calls += 1
        
        # Phase 2: DE for global optimization
        de_budget = min(300, self.budget - calls)
        if de_budget > 10 and time.time() < deadline - 30:
            x_de, f_de, de_calls = self._de_optimize(
                best_x, best_f, de_budget, deadline, pop_size=15, F=0.8, CR=0.9
            )
            if f_de < best_f:
                best_f = f_de
                best_x = x_de.copy()
            calls += de_calls
        
        # Phase 3: Nelder-Mead local refinement
        nm_budget = self.budget - calls
        if nm_budget > 5 and time.time() < deadline - 10:
            x_nm, f_nm, nm_calls = self._nelder_mead(best_x, best_f, nm_budget, deadline)
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
            calls += nm_calls
        
        return best_x, best_f

# EVOLVE-BLOCK-END
