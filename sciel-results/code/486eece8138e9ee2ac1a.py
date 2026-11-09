"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic Sobol-based global search with progressive local refinement.
    
    Strategy:
    1. Use a scrambled Sobol sequence mapped to [-4, 4]^5 for global coverage.
    2. After the first 50 evaluations, identify the best point and begin
       interleaving local refinements around it while continuing global search.
    3. This provides early progress (scoring favors early improvements) while
       maintaining global coverage to avoid missing the true minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        from scipy.stats import qmc
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        # Generate global Sobol points mapped to [-4, 4]^5
        # We need up to budget points for global search
        sobol_global = qmc.Sobol(d=dim, scramble=True, seed=42)
        points_unit_global = sobol_global.random(budget)
        points_global = points_unit_global * 8.0 - 4.0
        
        # We'll interleave: first 50 are pure global, then alternate
        # 1 global + 1 local refinement per pair after that
        # This ensures early progress while maintaining coverage
        
        # Pre-generate local Sobol points around origin (we'll recenter)
        # We'll generate them on the fly as we find better best_x
        
        local_seed = 43
        local_points_used = 0
        local_sobol = qmc.Sobol(d=dim, scramble=True, seed=local_seed)
        # Generate a large batch of local unit points
        max_local = budget
        points_unit_local = local_sobol.random(max_local)
        
        # Phase 1: First 50 calls - pure global exploration
        n_phase1 = min(50, budget)
        for i in range(n_phase1):
            if calls_made >= budget:
                break
            x = points_global[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Interleave global and local
        # After finding a good point, do local refinement
        # Strategy: every 2 calls, 1 is local refinement around current best
        # Local search radius shrinks over time
        
        global_idx = n_phase1  # next global point to use
        local_idx = 0  # next local point to use
        
        while calls_made < budget:
            if best_x is None:
                # Should not happen, but safety
                x = points_global[global_idx]
                fx = func(x)
                calls_made += 1
                global_idx += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                continue
            
            # Determine local search radius based on how many local points used
            # Start with radius 2.0, shrink to 0.1
            # Use a smooth decay
            progress = local_idx / max(1, max_local)
            radius = 2.0 * (1.0 - 0.9 * progress) + 0.1 * progress
            radius = max(radius, 0.1)
            
            # Get next local point
            if local_idx < max_local:
                local_unit = points_unit_local[local_idx]
                local_x = best_x + (local_unit - 0.5) * 2.0 * radius
                local_x = np.clip(local_x, -4.0, 4.0)
                local_idx += 1
            else:
                # Fallback to global
                local_x = None
            
            # Alternate: 1 local, then 1 global
            if local_idx % 2 == 1 and local_x is not None:
                # Local refinement
                x = local_x
            else:
                # Global exploration
                if global_idx < budget:
                    x = points_global[global_idx]
                    global_idx += 1
                else:
                    # No more global points, do local
                    if local_x is not None:
                        x = local_x
                    else:
                        break
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
