import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0
        self.start_time = 0
        self.deadline = 320

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # 1. Initialization with Latin Hypercube
        n_init = min(50, self.budget // 4)
        perm = np.random.permutation(n_init)
        X_init = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            X_init[:, d] = np.linspace(lo[d], hi[d], n_init, endpoint=False)[perm] + span[d] * np.random.random() * (1/n_init)
            
        F_init = np.array([self._evaluate(problem, x) for x in X_init])
        
        for i in range(n_init):
            self._update_archive(X_init[i], F_init[i])
            
        # 2. Niche-based Local Search (Crowding)
        # Strategy: Instead of a single adaptive step size, we maintain a population of "niches".
        # Each niche has its own step size and center.
        # We select a niche based on its contribution to the hypervolume (proxy for importance).
        # This allows simultaneous exploration of different regions of the Pareto front.
        
        # Initialize niches from the best points in the initial archive
        # We'll create up to 10 niches
        n_niches = 10
        
        # Sort archive by f0 to spread niches along the front
        self.archive.sort(key=lambda x: x[1][0])
        
        niches = []
        # Pick points evenly spaced from the sorted archive
        if len(self.archive) >= n_niches:
            indices = np.linspace(0, len(self.archive) - 1, n_niches, dtype=int)
            for idx in indices:
                x, f = self.archive[idx]
                niches.append({
                    'center': x.copy(),
                    'step': 0.2 * span, # Initial step size
                    'age': 0
                })
        else:
            # If archive is small, just use what we have
            for x, f in self.archive:
                niches.append({
                    'center': x.copy(),
                    'step': 0.2 * span,
                    'age': 0
                })
        
        # If we still don't have enough niches, add random ones
        while len(niches) < n_niches:
            x = lo + span * np.random.random(self.dim)
            niches.append({
                'center': x,
                'step': 0.2 * span,
                'age': 0
            })
        
        # Track best hypervolume contribution per niche (simplified: just use age and success)
        niche_success = [0] * len(niches)
        
        while self.eval_count < self.budget:
            if time.time() > self.deadline:
                break
            
            # Select a niche using weighted random selection based on recent success
            # Higher success -> higher weight
            weights = [max(1, s) for s in niche_success]
            total_weight = sum(weights)
            r = np.random.random() * total_weight
            cumulative = 0
            niche_idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    niche_idx = i
                    break
            
            niche = niches[niche_idx]
            
            # Generate child around niche center
            # Adaptive step: shrink if no recent success, grow slightly if success
            child_x = niche['center'] + niche['step'] * np.random.randn(self.dim)
            child_x = np.clip(child_x, lo, hi)
            
            child_f = self._evaluate(problem, child_x)
            
            # Check dominance
            dominated = False
            for j in range(len(self.archive)):
                if np.all(child_f < self.archive[j][1]) and np.any(child_f < self.archive[j][1]):
                    dominated = True
                    break
            
            if not dominated:
                # Remove existing points dominated by child
                new_archive = []
                improved = False
                for j in range(len(self.archive)):
                    ax, af = self.archive[j]
                    if not (np.all(child_f <= af) and np.any(child_f < af)):
                        new_archive.append((ax, af))
                    else:
                        improved = True
                self.archive = new_archive
                self.archive.append((child_x, child_f))
                
                if improved:
                    # Update niche
                    niche['center'] = child_x.copy()
                    niche['step'] *= 0.9 # Shrink step to exploit
                    niche_success[niche_idx] += 1
                else:
                    # Child was non-dominated but didn't improve archive (rare, but possible if equal)
                    # Or if it was dominated, we penalize
                    pass
            else:
                # Child was dominated, penalize niche
                niche_success[niche_idx] = max(0, niche_success[niche_idx] - 1)
                # Shrink step more aggressively if failing
                niche['step'] *= 0.8
            
            # Decay success scores to allow niches to recover
            for i in range(len(niche_success)):
                niche_success[i] *= 0.95
        
        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for ax, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
