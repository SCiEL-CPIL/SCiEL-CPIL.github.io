"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restart (1+1)-ES with global scan for Lunacek."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin (330s wall clock limit)
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if not np.all(np.isfinite(x)):
                return np.inf
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Structured global scan + random sampling
        # Lunacek f24 has two basins; structured points help locate them.
        global_budget = max(50, total_budget // 4)
        
        # Build candidate list: structured + random
        candidates = []
        # Origin
        candidates.append(np.zeros(n))
        # Axis-aligned offsets (helps find shifted basins)
        for i in range(n):
            for s in (-3.0, -1.5, 1.5, 3.0):
                p = np.zeros(n)
                p[i] = s
                candidates.append(p)
        # Diagonal points
        for s in (-2.0, -1.0, 1.0, 2.0):
            candidates.append(np.full(n, s))
        # Random points
        for _ in range(30):
            candidates.append(rng.uniform(-4.5, 4.5, n))
        
        # Evaluate structured candidates first
        for x in candidates:
            if time.time() > deadline or calls_made >= global_budget:
                break
            evaluate(x)
        
        # Fill remaining global budget with random sampling
        while calls_made < global_budget and time.time() < deadline:
            x = rng.uniform(-5.0, 5.0, n)
            evaluate(x)

        # Phase 2: Local refinement using (1+1)-ES from best point
        if best_x is None:
            best_x = np.zeros(n)
            best_f = evaluate(best_x)

        # (1+1)-ES parameters
        sigma = 1.5  # Initial step size for local refinement
        no_improve = 0
        max_no_improve = 40  # Restart if no improvement for this many steps
        exploration_every = 25  # Periodic re-exploration interval
        steps_since_exploration = 0

        while calls_made < total_budget and time.time() < deadline:
            # Generate candidate
            z = rng.standard_normal(n)
            x_candidate = best_x + sigma * z
            x_candidate = np.clip(x_candidate, -5.0, 5.0)

            if not np.all(np.isfinite(x_candidate)):
                no_improve += 1
                if no_improve > max_no_improve:
                    # Random restart
                    best_x = rng.uniform(-5.0, 5.0, n)
                    best_f = evaluate(best_x)
                    sigma = 1.5
                    no_improve = 0
                    steps_since_exploration = 0
                continue

            f_candidate = evaluate(x_candidate)
            steps_since_exploration += 1

            if f_candidate < best_f:
                # Accept
                best_x = x_candidate.copy()
                best_f = f_candidate
                no_improve = 0
                # Slightly increase step size on success (cautious)
                sigma *= 1.08
                if sigma > 5.0:
                    sigma = 5.0
            else:
                # Reject
                no_improve += 1
                # Decrease step size on failure
                sigma *= 0.92
                if sigma < 0.01:
                    sigma = 0.01

            # Periodic re-exploration: try a few random points to find other basins
            if steps_since_exploration >= exploration_every and no_improve > 5:
                steps_since_exploration = 0
                found_better = False
                for _ in range(3):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    xr = rng.uniform(-4.5, 4.5, n)
                    fr = evaluate(xr)
                    if fr < best_f:
                        best_x = xr.copy()
                        best_f = fr
                        sigma = 1.5
                        no_improve = 0
                        found_better = True
                        break
                if found_better:
                    steps_since_exploration = 0

            # If stuck for too long, do a small random perturbation to escape
            if no_improve > max_no_improve:
                # Small random jump
                jump_size = 0.5
                perturbation = rng.uniform(-jump_size, jump_size, n)
                x_perturbed = best_x + perturbation
                x_perturbed = np.clip(x_perturbed, -5.0, 5.0)
                f_perturbed = evaluate(x_perturbed)
                if f_perturbed < best_f:
                    best_x = x_perturbed.copy()
                    best_f = f_perturbed
                sigma = 0.5
                no_improve = 0
                steps_since_exploration = 0

        if best_x is None:
            best_x = np.zeros(n)
            if calls_made < total_budget:
                best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
