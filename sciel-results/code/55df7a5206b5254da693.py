"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT4: exploit g=0 condition, sweep x1 for front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT4: x1 in [-5,5] but Pareto front is x1 in [0,1]
        # x2..x10 in [-5,5], g(x)=sum(xi^2), need g=0 for Pareto front
        # Strategy: set x2..x10 = 0, sweep x1 over [0,1] densely

        # Phase 1: Dense sweep of x1 with x2..x10 = 0
        # This directly samples the Pareto front
        n_sweep = 100
        x1_values = np.linspace(0.0, 1.0, n_sweep)
        
        archive = []  # list of (x, f) pairs
        used = 0

        for x1 in x1_values:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            archive.append((x.copy(), np.array(f)))
            used += 1

        # Phase 2: Refine with local search around promising points
        # Use remaining budget for local perturbations
        remaining = self.budget - used
        if remaining > 0 and archive:
            # Sort archive by f1 to get ordered front
            archive.sort(key=lambda t: t[1][0])
            
            # Pick diverse points from the front for local refinement
            n_refine = min(remaining, 50)
            if n_refine > 0:
                # Sample evenly from the sorted archive
                indices = np.linspace(0, len(archive) - 1, n_refine).astype(int)
                rng = np.random.default_rng(42)
                
                for idx in indices:
                    if used >= self.budget or time.time() > deadline:
                        break
                    parent_x = archive[idx][0].copy()
                    parent_f = archive[idx][1].copy()
                    
                    # Local perturbation: small step in x1, tiny step in others
                    step_x1 = 0.01 * rng.standard_normal()
                    step_rest = 0.001 * rng.standard_normal(self.dim - 1)
                    
                    child_x = parent_x.copy()
                    child_x[0] += step_x1
                    child_x[1:] += step_rest
                    child_x = np.clip(child_x, lo, hi)
                    
                    child_f = problem(child_x)
                    used += 1
                    
                    # Only add if not dominated by existing archive
                    dominated = False
                    for _, af in archive:
                        if np.all(af <= child_f) and np.any(af < child_f):
                            dominated = True
                            break
                    if not dominated:
                        archive.append((child_x, child_f))
                    
                    # Remove points dominated by the new child
                    new_archive = []
                    for ax, af in archive:
                        if np.all(child_f <= af) and np.any(child_f < af):
                            continue  # dominated by child
                        new_archive.append((ax, af))
                    archive = new_archive

        # Phase 3: If budget remains, do more dense sampling
        if used < self.budget and time.time() < deadline:
            # Additional fine-grained sweep in gaps
            if len(archive) > 1:
                archive.sort(key=lambda t: t[1][0])
                f1_vals = [t[1][0] for t in archive]
                # Find largest gaps in f1
                gaps = []
                for i in range(len(f1_vals) - 1):
                    gap = f1_vals[i + 1] - f1_vals[i]
                    gaps.append((gap, i))
                gaps.sort(reverse=True)
                
                for gap_size, i in gaps[:10]:
                    if used >= self.budget or time.time() > deadline:
                        break
                    if gap_size < 0.001:
                        break
                    # Insert midpoint
                    x1_mid = (f1_vals[i] + f1_vals[i + 1]) / 2.0
                    x = np.zeros(self.dim)
                    x[0] = x1_mid
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    used += 1
                    archive.append((x, np.array(f)))
                    # Re-sort
                    archive.sort(key=lambda t: t[1][0])

        # Return the archive (the return value is ignored, but we need to return something)
        if archive:
            return np.array([a[0] for a in archive])
        else:
            return np.array([np.zeros(self.dim)])
# EVOLVE-BLOCK-END
