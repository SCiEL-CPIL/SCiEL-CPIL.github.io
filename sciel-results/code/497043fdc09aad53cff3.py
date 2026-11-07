"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma, lattice-aware restarts, and sustained convergence."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Allocate budget: most for main search, small portion for final polish
        polish_budget = max(20, budget // 25)
        search_budget = budget - polish_budget

        # Precompute a coarse lattice for restarts to exploit Rastrigin periodicity
        # The minimizer is in [-4, 4]^5. A grid of step 2.0 covers this well.
        lattice_vals = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])

        # Phase 1: Main (1+1)-ES with lattice-aware restarts
        # Start at origin
        x = np.zeros(d)
        fx = func(x)
        total_calls += 1
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        sigma = 2.5  # Start with a larger step to escape local minima quickly
        no_improve = 0
        max_no_improve = 40  # Restart threshold: balance exploration vs convergence
        restart_count = 0
        max_restarts = 8  # Allow a few restarts, but prioritize sustained search
        
        # Track if we are in a "converged" state to reduce restarts later
        converged = False

        while total_calls < search_budget and not check_deadline():
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                # Aggressive expansion on success
                sigma *= 1.4 
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                # If we've been improving for a while, we might be in the right basin
                if no_improve == 0 and sigma < 0.5:
                    converged = True
            else:
                # Contract on failure
                sigma *= 0.85
                no_improve += 1

            # Restart Logic
            # Only restart if we are stuck AND we haven't converged recently
            # If converged, we allow more failures before restarting to let it grind down
            effective_max_no_improve = max_no_improve if not converged else max_no_improve * 2
            
            if no_improve >= effective_max_no_improve and restart_count < max_restarts:
                restart_count += 1
                # Lattice-aware restart: Pick a random point from the coarse grid
                # This ensures we jump to a different "periodic" region
                x = np.array([lattice_vals[rng.integers(0, len(lattice_vals))] for _ in range(d)])
                fx = func(x)
                total_calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                
                # Reset sigma to a medium value to explore the new basin
                sigma = 1.5
                no_improve = 0
                converged = False # Reset convergence flag

            # If we have used all restarts, we just keep going with current sigma
            # The sigma will shrink to near zero, effectively doing a local polish

        # Phase 2: Final polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2  # Start polish with a small step

            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)

            for _ in range(polish_steps):
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    # Slow expansion in polish phase to avoid jumping out
                    sigma *= 1.1 
                else:
                    sigma *= 0.9

                # Stop if sigma is very small
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
