"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search: random init, mutate non-dominated parents."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        pop = int(min(20, self.budget // 4))
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Local improvement helper
        def local_refine(x, f, max_iters=50):
            """Local hill-climbing with small steps."""
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                # Try small perturbations
                best_x, best_f = x, f
                for _ in range(10):
                    step = 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    new_x = np.clip(x + step, lo, hi)
                    new_f = problem(new_x)
                    used_local = 1  # Track evaluations in outer scope via nonlocal? No, just count.
                    # Since we can't easily track nonlocal int in nested func without nonlocal, 
                    # we'll just do it inline in the main loop or pass a mutable list.
                    # For simplicity, let's just do a few steps inline if we have time.
                    # Actually, let's stick to the main loop structure to be safe with budget.
                    break 
            return x, f

        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Find non-dominated set
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Select parent
            p = parents[rng.integers(len(parents))]
            
            # Adaptive mutation: start with larger steps, shrink over time
            # Or just use a reasonable fixed step that works for WFG4
            # WFG4 has a curved front. 0.1 was used before. Let's try 0.05 for finer search later?
            # Let's keep 0.1 but maybe add a second stage?
            # Let's just stick to the simple loop but ensure we don't time out.
            
            child = p + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replacement: replace the worst point in terms of sum of objectives
            # Or use crowding distance? Sum is simpler.
            worst = int(np.argmax(np.sum(F, axis=1)))
            
            # Only replace if child is better or non-dominated?
            # The original just replaced the worst. Let's keep that but maybe check if child is better.
            # If child is dominated by the worst point, replacing is bad?
            # Actually, standard (1+1) EA replaces if better.
            # Let's check if child dominates worst or is non-dominated with worst.
            # If child is better than worst in sum, replace.
            if np.sum(fc) < np.sum(F[worst]):
                X[worst], F[worst] = child, fc
            elif not np.all(F[worst] <= fc) and not np.all(F[worst] >= fc):
                # Non-dominated, maybe keep both? But we have fixed size.
                # Let's just replace if it's "better" in some sense or random.
                # For now, keep the simple sum rule.
                pass

        return X
# EVOLVE-BLOCK-END
