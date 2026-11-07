"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with stagnation-triggered restarts for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        # Multi-start initialization: spread across the domain
        # Use a fixed set of diverse starting points
        rng = np.random.default_rng(42)

        # Strategy parameters
        mu = 8   # number of parents
        lam = 16  # number of offspring
        # CMA-like step size adaptation via (1/5) rule on population
        sigma = 2.0

        # Track best
        best_x = None
        best_f = np.inf

        # Multi-start: try several different basins
        # We'll use a population-based ES with restarts
        # Initial population: spread across the box
        # For Rastrigin, the global minimum is at some shifted point
        # We want to explore broadly first

        # Phase 1: Broad exploration with multiple random starts
        # Phase 2: Focused exploitation with ES

        # Let's do a (mu+lambda)-ES with periodic restarts when stagnating

        # Initial population: diverse points
        def init_pop(n):
            """Generate n diverse initial points."""
            # Mix of uniform random and grid-like points
            pts = []
            for i in range(n):
                if i < n // 2:
                    # Uniform random
                    pts.append(rng.uniform(-5, 5, dim))
                else:
                    # Structured: use different center offsets
                    center = rng.uniform(-4, 4, dim)
                    pts.append(np.clip(center + rng.uniform(-1, 1, dim), -5, 5))
            return np.array(pts)

        # Main loop with restarts
        eval_count = 0
        max_evals = budget

        # Restart strategy: if no improvement for a window, restart from a new random point
        # but keep the best found so far
        stagnation_window = 30  # number of generations without improvement to trigger restart
        no_improve = 0

        # Initial population
        pop = init_pop(lam)
        pop_f = np.array([func(x) for x in pop])
        eval_count += lam

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # Sort population by fitness
        order = np.argsort(pop_f)
        pop = pop[order]
        pop_f = pop_f[order]

        gen = 0
        while eval_count < max_evals:
            if time.time() > deadline:
                break

            gen += 1

            # Select parents (top mu)
            parents = pop[:mu]
            parent_f = pop_f[:mu]

            # Compute mean and sigma adaptation
            # Simple sigma adaptation: (1/5) rule
            # Count successes in last generation
            # We'll use a simpler approach: adapt sigma based on improvement rate

            # Generate offspring
            offspring = np.zeros((lam, dim))
            offspring_f = np.zeros(lam)

            for i in range(lam):
                if eval_count >= max_evals:
                    break
                if time.time() > deadline:
                    break

                # Select a parent with weighted selection
                # Use tournament or weighted
                if rng.random() < 0.3:
                    # Explore: pick a random parent and large step
                    p = parents[rng.integers(mu)]
                    step = sigma * 2.0
                else:
                    # Exploit: pick best parent
                    p = parents[0]
                    step = sigma

                y = p + step * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                offspring[i] = y
                offspring_f[i] = func(y)
                eval_count += 1

            # Handle case where we didn't generate all offspring
            if eval_count < max_evals:
                # Fill remaining with copies of best (shouldn't happen)
                pass

            # Combine population and offspring, select best mu+lam -> keep top (mu+lam)
            # Actually, keep top lam for next generation (truncation)
            all_x = np.vstack([pop, offspring[:lam]])
            all_f = np.concatenate([pop_f, offspring_f])

            # Find new best
            global_best_idx = np.argmin(all_f)
            if all_f[global_best_idx] < best_f:
                best_f = all_f[global_best_idx]
                best_x = all_x[global_best_idx].copy()
                no_improve = 0
            else:
                no_improve += 1

            # Adapt sigma
            # Use (1/5) rule: fraction of successful steps
            # Approximate: compare offspring to their parents
            # Simpler: if best improved, increase sigma slightly; else decrease
            if no_improve == 0:
                sigma = min(sigma * 1.1, 5.0)
            else:
                sigma = max(sigma * 0.95, 0.1)

            # Stagnation restart
            if no_improve >= stagnation_window:
                # Restart: replace some of the population with new random points
                # Keep the best few, replace the rest
                n_replace = lam - mu
                for i in range(mu, lam):
                    pop[i] = np.clip(best_x + rng.normal(0, sigma * 1.5, dim), -5, 5)
                    pop_f[i] = func(pop[i])
                    eval_count += 1
                    if eval_count >= max_evals:
                        break
                    if time.time() > deadline:
                        break

                # Also try a completely new random point
                new_pt = rng.uniform(-5, 5, dim)
                new_f = func(new_pt)
                eval_count += 1
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_pt.copy()

                # Re-sort
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]
                no_improve = 0
                sigma = 2.0  # Reset step size

                if eval_count >= max_evals or time.time() > deadline:
                    break

            else:
                # Normal selection: keep top lam from combined
                combined_order = np.argsort(all_f)
                pop = all_x[combined_order[:lam]].copy()
                pop_f = all_f[combined_order[:lam]].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
