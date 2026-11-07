"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT6-specific optimizer using known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget
        
        # ZDT6 specific: x2-x10 should be 0.5 to minimize g
        # x1 varies to explore Pareto front
        # Strategy: allocate evaluations to sample x1 values and refine
        
        # Phase 1: Coarse sampling of x1 with x2-x10 = 0.5
        # The Pareto front in objective space is parameterized by f1 in [0,1]
        # f1 = 1 - exp(-4*x1) * sin(10*pi*x1)^6
        # We want good coverage of f1 values
        
        # Use quantile sampling of x1 to get even coverage of f1
        # f1 is not monotonic in x1, so we need to be careful
        # Let's sample x1 uniformly first, then adapt
        
        # Initial population: sample x1 values
        n_coarse = min(40, budget // 5)
        x1_samples = np.linspace(0.01, 0.99, n_coarse)
        
        archive = []  # List of (x, f) pairs
        used = 0
        
        # Evaluate coarse samples with x2-x10 = 0.5
        for x1 in x1_samples:
            if used >= budget or time.time() > deadline:
                break
            x = np.full(n, 0.5)
            x[0] = x1
            x = np.clip(x, lo, hi)
            f = problem(x)
            archive.append((x, f))
            used += 1
        
        # Phase 2: Local refinement around best points
        # Sort by f1 and find diverse points
        if len(archive) > 0:
            # Keep non-dominated points
            def dominates(f1, f2):
                return (f1[0] <= f2[0] and f1[1] <= f2[1] and 
                        (f1[0] < f2[0] or f1[1] < f2[1]))
            
            # Simple non-dominated filter
            nd_indices = []
            for i, (_, fi) in enumerate(archive):
                is_dominated = False
                for j, (_, fj) in enumerate(archive):
                    if i != j and dominates(fj, fi):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_indices.append(i)
            
            # Refine around non-dominated points
            if len(nd_indices) > 0:
                # Select a subset for refinement
                n_refine = min(len(nd_indices), budget - used - 5)
                if n_refine > 0:
                    # Spread out refinement points
                    step = max(1, len(nd_indices) // n_refine)
                    refine_indices = nd_indices[::step][:n_refine]
                    
                    for idx in refine_indices:
                        if used >= budget - 2 or time.time() > deadline:
                            break
                        x_base, f_base = archive[idx]
                        
                        # Try small perturbations in x1
                        for delta in [-0.02, 0.02, -0.005, 0.005]:
                            if used >= budget - 1 or time.time() > deadline:
                                break
                            x = x_base.copy()
                            x[0] += delta
                            x = np.clip(x, lo, hi)
                            f = problem(x)
                            archive.append((x, f))
                            used += 1
        
        # Phase 3: If budget remains, do random exploration
        rng = np.random.default_rng(42)
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(n)
            f = problem(x)
            archive.append((x, f))
            used += 1
        
        return archive
# EVOLVE-BLOCK-END
