"""Multi-objective optimizer for RE21 with hypervolume-guided DE approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21 (2 objectives, 4 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f

        def get_non_dominated(F_list):
            """Return indices of non-dominated points in F_list."""
            if len(F_list) == 0:
                return []
            F_arr = np.array(F_list, dtype=float)
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]

        def compute_hv_contribution(front_pts, ref_pt):
            """Compute 2D hypervolume of front_pts w.r.t. ref_pt (only points strictly better than ref)."""
            if len(front_pts) == 0:
                return 0.0
            pts = np.array(front_pts, dtype=float)
            # Filter strictly better than ref
            mask = np.all(pts < ref_pt, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            # For 2D, use the standard sweep-line algorithm
            # Sort by first objective ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            
            hv = 0.0
            max_obj1 = 0.0  # Track the maximum (worst) obj1 seen so far from the right
            
            # Process from right to left (highest obj0 to lowest)
            for i in range(len(pts) - 1, -1, -1):
                # The width is determined by the next point's obj0 (or ref if last)
                if i < len(pts) - 1:
                    w0 = pts[i + 1, 0] - pts[i, 0]
                else:
                    w0 = ref_pt[0] - pts[i, 0]
                
                # The height is ref[1] - max_obj1_so_far
                h1 = ref_pt[1] - max_obj1
                
                if h1 > 0 and w0 > 0:
                    hv += w0 * h1
                
                # Update max_obj1
                max_obj1 = max(max_obj1, pts[i, 1])
            
            return hv

        # Phase 1: Reference-point-aligned deterministic grid initialization
        # Focus on the asymmetric space: obj0 range [0, 3000], obj1 range [0, 0.06]
        # We need to explore the decision space to find points that achieve low obj0 and low obj1
        
        n_init = 30
        # Use a stratified random approach with emphasis on different regions
        rng = np.random.RandomState(42)
        
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            # Stratified sampling: divide each dimension into segments
            x = np.zeros(self.dim)
            for d in range(self.dim):
                segment = int(i * self.dim / n_init) % self.dim
                if d == segment:
                    # Spread across the dimension
                    frac = (i % (n_init // self.dim + 1)) / (n_init // self.dim + 1)
                    x[d] = lo[d] + frac * (hi[d] - lo[d])
                else:
                    x[d] = (lo[d] + hi[d]) / 2.0
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # Also add some random points
        for i in range(10):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            evaluate(x)

        # Phase 2: Hypervolume-guided Adaptive Differential Evolution
        # Population-based DE with shrinking mutation factor
        
        pop_size = min(20, self.budget - used)
        if pop_size < 4:
            # Not enough budget for DE, just do random search
            while used < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x)
            return all_X

        # Initialize population from best non-dominated points + random
        front_idx = get_non_dominated(all_F)
        pop = []
        pop_F = []
        
        # Add non-dominated points to population
        for idx in front_idx[:pop_size // 2]:
            pop.append(all_X[idx].copy())
            pop_F.append(all_F[idx].copy())
        
        # Fill remaining with random points
        while len(pop) < pop_size and used < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = evaluate(x)
            pop.append(x)
            pop_F.append(f)
        
        if len(pop) == 0:
            while used < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x)
            return all_X

        # DE loop with shrinking mutation factor
        max_generations = max(5, (self.budget - used) // (pop_size + 1))
        
        for gen in range(max_generations):
            if used >= self.budget or time.time() > deadline:
                break
            
            # Dynamically shrinking mutation factor
            F = 0.8 * (1.0 - gen / max_generations) + 0.2
            CR = 0.5
            
            # Find non-dominated in population
            pop_front = get_non_dominated(pop_F)
            
            for i in range(len(pop)):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Differential evolution mutation
                if len(pop) >= 3:
                    # Select three distinct random indices
                    candidates = [j for j in range(len(pop)) if j != i]
                    a, b, c = np.random.choice(candidates, size=min(3, len(candidates)), replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    mutant = pop[i] + (hi - lo) * 0.1 * np.random.randn(self.dim)
                
                # Crossover
                trial = pop[i].copy()
                for d in range(self.dim):
                    if np.random.random() < CR or d == np.random.randint(self.dim):
                        trial[d] = mutant[d]
                
                trial = np.clip(trial, lo, hi)
                
                # Evaluate trial
                f_trial = evaluate(trial)
                
                # Selection: keep if it improves hypervolume or is non-dominated
                current_front = get_non_dominated(all_F)
                if len(current_front) == 0:
                    pop[i] = trial
                    pop_F[i] = f_trial
                else:
                    # Check if trial is non-dominated w.r.t. current front
                    is_dominated = False
                    for idx in current_front:
                        if np.all(all_F[idx] <= f_trial) and np.any(all_F[idx] < f_trial):
                            is_dominated = True
                            break
                    if not is_dominated:
                        pop[i] = trial
                        pop_F[i] = f_trial
            
            # Periodically refresh population with new random points
            if gen % 3 == 2 and used < self.budget:
                for _ in range(min(2, len(pop))):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = lo + (hi - lo) * np.random.random(self.dim)
                    f = evaluate(x)
                    idx_refresh = np.random.randint(len(pop))
                    pop[idx_refresh] = x
                    pop_F[idx_refresh] = f

        # Phase 3: Hypervolume-guided local refinement
        front_idx = get_non_dominated(all_F)
        if len(front_idx) > 0:
            # Compute hypervolume contribution for each front point
            front_F = [all_F[i] for i in front_idx]
            front_X = [all_X[i] for i in front_idx]
            
            # Sort by hypervolume contribution (descending)
            hv_contribs = []
            for i in range(len(front_F)):
                # Compute HV with and without this point
                hv_with = compute_hv_contribution(front_F, ref)
                without = [f for j, f in enumerate(front_F) if j != i]
                hv_without = compute_hv_contribution(without, ref)
                hv_contribs.append(hv_with - hv_without)
            
            order = np.argsort(hv_contribs)[::-1]
            
            # Refine top points by HV contribution
            for idx_order in order[:min(8, len(order))]:
                if used >= self.budget or time.time() > deadline:
                    break
                base_idx = front_idx[idx_order]
                base_x = all_X[base_idx].copy()
                base_f = all_F[base_idx].copy()
                
                # Coordinate-wise local search with adaptive step
                for dim_i in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    step_frac = 0.05
                    for _ in range(4):
                        if used >= self.budget or time.time() > deadline:
                            break
                        step_size = (hi[dim_i] - lo[dim_i]) * step_frac
                        improved = False
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_new = base_x.copy()
                            x_new[dim_i] += direction * step_size
                            x_new = np.clip(x_new, lo, hi)
                            f_new = evaluate(x_new)
                            # Accept if it improves at least one objective without worsening the other
                            if (f_new[0] < base_f[0] and f_new[1] <= base_f[1]) or \
                               (f_new[1] < base_f[1] and f_new[0] <= base_f[1]):
                                base_x = x_new.copy()
                                base_f = f_new.copy()
                                improved = True
                                break
                        if not improved:
                            step_frac *= 0.5
                        if step_frac < 0.003:
                            break

        # Phase 4: Final hypervolume-guided random exploration
        while used < self.budget:
            if time.time() > deadline:
                break
            # Bias random sampling toward regions that may improve HV
            # Sample in a box around the current best front points
            front_idx = get_non_dominated(all_F)
            if len(front_idx) > 0 and np.random.random() < 0.7:
                # Pick a random front point and perturb
                base_idx = front_idx[np.random.randint(len(front_idx))]
                base_x = all_X[base_idx]
                scale = 0.1 + 0.1 * np.random.random()
                x = base_x + scale * (hi - lo) * np.random.randn(self.dim)
            else:
                x = lo + (hi - lo) * np.random.random(self.dim)
            x = np.clip(x, lo, hi)
            evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
