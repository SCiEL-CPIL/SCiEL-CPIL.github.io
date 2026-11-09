"""Multi-objective optimizer on RE21 using scatter-and-polish with Sobol sampling and multi-start local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    """Scatter-and-polish multi-objective optimizer with Sobol initialization and multi-start local optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        archive = []  # list of (x, f) tuples
        used = 0

        def is_valid(f):
            return np.all(f < ref) and np.all(np.isfinite(f))

        def add_to_archive(x, f):
            if not is_valid(f):
                return
            # Check if dominated by existing archive
            for _, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove dominated points
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            archive[:] = new_archive

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            used += 1
            if is_valid(f):
                add_to_archive(x, f)
            return f

        # Phase 1: Sobol sequence initialization for uniform coverage
        n_init = min(64, self.budget)
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=self.dim, scramble=True)
            sobol_samples = sampler.random(n_init)
            for i in range(n_init):
                if used >= self.budget or time.time() > deadline:
                    break
                x = np.array([lo[d] + (hi[d] - lo[d]) * sobol_samples[i, d] for d in range(self.dim)])
                evaluate(x)
        except ImportError:
            # Fallback to Latin Hypercube if Sobol not available
            for i in range(n_init):
                if used >= self.budget or time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                for d in range(self.dim):
                    stratum = np.random.randint(0, n_init)
                    frac = (stratum + np.random.random()) / n_init
                    x[d] = lo[d] + (hi[d] - lo[d]) * frac
                evaluate(x)

        # Phase 2: Multi-start local optimization with diverse weight vectors
        # Use a set of weight vectors to cover the Pareto front
        n_weights = 12
        weights = np.linspace(0.05, 0.95, n_weights)
        
        # Select diverse starting points from archive
        if len(archive) > 0:
            # Sort archive by first objective to get diverse starting points
            sorted_archive = sorted(archive, key=lambda item: item[1][0])
            # Pick starting points spread across the archive
            n_starts = min(6, len(sorted_archive))
            if n_starts >= 2:
                start_indices = np.linspace(0, len(sorted_archive) - 1, n_starts, dtype=int)
                start_points = [sorted_archive[i] for i in start_indices]
            else:
                start_points = sorted_archive[:n_starts]
        else:
            # If no valid points, use random starts
            start_points = []
            for _ in range(6):
                x = np.random.uniform(lo, hi)
                start_points.append((x, None))

        # For each weight vector and starting point, run local optimization
        for w1 in weights:
            if used >= self.budget or time.time() > deadline - 30.0:
                break
            w2 = 1.0 - w1
            
            for x_init, _ in start_points:
                if used >= self.budget or time.time() > deadline - 30.0:
                    break
                
                if x_init is None:
                    x_init = np.random.uniform(lo, hi)
                
                def objective(x):
                    f = problem(x)
                    return w1 * f[0] + w2 * f[1]
                
                try:
                    result = minimize(
                        objective,
                        x_init,
                        method='L-BFGS-B',
                        bounds=list(bounds),
                        options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10}
                    )
                    if used < self.budget and time.time() < deadline - 10.0:
                        f_new = problem(result.x)
                        used += 1
                        if is_valid(f_new):
                            add_to_archive(result.x, f_new)
                except Exception:
                    pass
        
        # Phase 3: Use remaining budget for differential evolution on weighted sums
        remaining_budget = self.budget - used
        if remaining_budget > 20 and time.time() < deadline - 60.0:
            # Use a few more weight vectors with DE
            de_weights = [0.2, 0.4, 0.6, 0.8]
            for w1 in de_weights:
                if used >= self.budget or time.time() > deadline - 30.0:
                    break
                w2 = 1.0 - w1
                
                def de_objective(x):
                    f = problem(x)
                    return w1 * f[0] + w2 * f[1]
                
                try:
                    result = differential_evolution(
                        de_objective,
                        bounds=list(bounds),
                        maxiter=50,
                        tol=1e-8,
                        seed=42,
                        polish=True
                    )
                    if used < self.budget and time.time() < deadline - 10.0:
                        f_new = problem(result.x)
                        used += 1
                        if is_valid(f_new):
                            add_to_archive(result.x, f_new)
                except Exception:
                    pass
        
        return archive

# EVOLVE-BLOCK-END
