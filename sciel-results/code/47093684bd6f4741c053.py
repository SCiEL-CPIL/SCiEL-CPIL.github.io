"""Multi-objective optimizer for RE37 using distance-based acquisition in objective space."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Halton initialization and iterative
    farthest-point sampling in objective space to cover the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def _halton(self, n, dim):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2
                f = 1.0
                result = 0.0
                j = i
                while j > 0:
                    f /= base
                    result += f * (j % base)
                    j //= base
                points[i - 1, d] = result
        return points

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        self.X = []
        self.F = []

        def add_eval(x, f):
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), lo, hi)
            f = np.asarray(f, dtype=float).reshape(-1)
            self.X.append(x.copy())
            self.F.append(f.copy())

        # --- Phase 1: Halton initialization ---
        n_init = min(30, self.budget // 6)
        halton_pts = self._halton(n_init, dim)
        X_init = lo + halton_pts * span

        for x in X_init:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                x = np.clip(x, lo, hi)
                f = eval_real(x)
                add_eval(x, f)
            except Exception:
                break

        if len(self.X) == 0:
            return None

        # --- Phase 2: Iterative farthest-point sampling in objective space ---
        # We maintain a list of "seed" points in objective space and repeatedly
        # pick the seed that maximizes the minimum distance to all existing
        # evaluated points in objective space, then perturb it in decision space.
        
        # Normalize objective space for distance computation
        F_arr = np.array(self.F)
        f_min = F_arr.min(axis=0)
        f_max = F_arr.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-15] = 1.0

        # Keep a set of candidate "anchor" points in objective space
        # Start with all currently evaluated points as anchors
        anchors_F = [f.copy() for f in self.F]
        
        # We'll also keep a list of (x, f) pairs for anchors to know where to perturb
        anchors_X = [x.copy() for x in self.X]
        
        max_iterations = self.budget - self.eval_count
        for it in range(max_iterations):
            if time.time() > deadline:
                break
            if self.eval_count >= self.budget:
                break
            
            if len(anchors_F) == 0:
                break
            
            # Find the anchor with the maximum minimum distance to all evaluated points
            F_eval = np.array(self.F)
            F_norm = (F_eval - f_min) / f_range
            
            best_anchor_idx = 0
            best_min_dist = -1.0
            
            for ai in range(len(anchors_F)):
                anchor_norm = (anchors_F[ai] - f_min) / f_range
                # Compute distances to all evaluated points
                dists = np.linalg.norm(F_norm - anchor_norm, axis=1)
                min_dist = np.min(dists)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_anchor_idx = ai
            
            # Perturb the best anchor in decision space
            x_anchor = anchors_X[best_anchor_idx].copy()
            
            # Try several perturbations and pick the best one based on objective distance
            n_perturb = 5
            best_x = x_anchor
            best_f = None
            best_dist_to_existing = -1.0
            
            for p in range(n_perturb):
                # Gaussian perturbation with decreasing scale
                scale = 0.1 * (0.7 ** p)
                x_trial = x_anchor + rng.normal(0, scale, dim)
                x_trial = np.clip(x_trial, lo, hi)
                
                try:
                    f_trial = eval_real(x_trial)
                except Exception:
                    continue
                
                # Compute distance to all existing evaluated points in normalized objective space
                f_trial_norm = (f_trial - f_min) / f_range
                dists = np.linalg.norm(F_norm - f_trial_norm, axis=1)
                min_dist = np.min(dists)
                
                if min_dist > best_dist_to_existing:
                    best_dist_to_existing = min_dist
                    best_x = x_trial
                    best_f = f_trial
            
            if best_f is not None:
                add_eval(best_x, best_f)
                anchors_F.append(best_f.copy())
                anchors_X.append(best_x.copy())
                
                # Update normalization if needed (keep it simple, recompute occasionally)
                if it % 10 == 0:
                    F_arr = np.array(self.F)
                    f_min = F_arr.min(axis=0)
                    f_max = F_arr.max(axis=0)
                    f_range = f_max - f_min
                    f_range[f_range < 1e-15] = 1.0
        
        return None
# EVOLVE-BLOCK-END
