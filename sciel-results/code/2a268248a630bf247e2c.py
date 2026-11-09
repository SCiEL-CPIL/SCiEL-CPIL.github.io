"""Multi-objective optimizer on RE21 using HypE-style hypervolume contribution selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """HypE-style multi-objective optimizer with hypervolume contribution selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref = np.array([3000.0, 0.06], dtype=np.float64)

        def is_valid(f):
            return np.all(f < ref)

        # Archive stored as sorted list by f[0] for efficient HV computation
        archive = []  # list of (x, f) sorted by f[0]
        used = 0

        def add_to_archive(x, f):
            nonlocal used
            if not is_valid(f):
                return
            # Check if dominated by any point in archive
            for ax, af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return  # dominated, don't add
            # Remove dominated points from archive
            new_archive = []
            for ax, af in archive:
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append((ax, af))
            new_archive.append((x, f))
            # Sort by first objective for efficient HV
            new_archive.sort(key=lambda item: item[1][0])
            archive[:] = new_archive

        def hv_2d_sorted(sorted_pts, ref_pt):
            """Compute hypervolume in 2D for points already sorted by f[0]."""
            if not sorted_pts:
                return 0.0
            hv = 0.0
            prev_y = ref_pt[1]
            for _, p in sorted_pts:
                if p[1] >= ref_pt[1]:
                    continue
                width = ref_pt[0] - p[0]
                height = prev_y - p[1]
                if width > 0 and height > 0:
                    hv += width * height
                prev_y = min(prev_y, p[1])
            return hv

        def marginal_contributions_fast(sorted_archive, ref_pt):
            """Compute marginal HV contribution in O(n) for 2D Pareto set sorted by f1 ascending."""
            n = len(sorted_archive)
            if n == 0:
                return np.array([])
            
            f1_vals = np.array([p[1][0] for p in sorted_archive])
            f2_vals = np.array([p[1][1] for p in sorted_archive])
            
            # For a Pareto set sorted by f1 ascending, the marginal contribution of point i
            # is the area of the rectangle it uniquely covers:
            # width = (f1[i+1] if i < n-1 else ref[0]) - f1[i]
            # height = (min f2 of points with index > i, or ref[1] if none) - f2[i]
            # But we must also ensure f2[i] < ref[1] and f1[i] < ref[0]
            
            # Suffix minimum of f2: min_y_from[i] = min(f2[i+1], ..., f2[n-1], ref[1])
            # Actually, for the height, we need the minimum y among points to the RIGHT of i
            # (strictly greater f1), capped at ref[1]. If no points to the right, use ref[1].
            
            suffix_min_y = np.full(n + 1, ref_pt[1])
            suffix_min_y[n] = ref_pt[1]
            for i in range(n - 1, -1, -1):
                suffix_min_y[i] = min(suffix_min_y[i + 1], f2_vals[i])
            
            # For point i, the "next" x boundary is f1[i+1] if i < n-1, else ref[0]
            # The height is: min(suffix_min_y[i+1], ref_pt[1]) - f2[i]
            # Wait, suffix_min_y[i+1] already includes ref[1] as the base.
            # Actually suffix_min_y[i] = min(f2[i], f2[i+1], ..., f2[n-1], ref[1])
            # We want the min y of points STRICTLY to the right: min(f2[i+1], ..., f2[n-1], ref[1])
            # That's suffix_min_y[i+1] if we define suffix_min_y[n] = ref[1]
            
            # Let me redefine: right_min_y[i] = min(f2[i+1], f2[i+2], ..., f2[n-1], ref[1])
            right_min_y = np.full(n, ref_pt[1])
            right_min_y[n-1] = ref_pt[1]
            for i in range(n - 2, -1, -1):
                right_min_y[i] = min(right_min_y[i + 1], f2_vals[i + 1])
            
            # Width for point i: next_x - f1[i], where next_x = f1[i+1] if i < n-1 else ref[0]
            next_x = np.full(n, ref_pt[0])
            next_x[:n-1] = f1_vals[1:]
            widths = next_x - f1_vals
            
            # Height for point i: right_min_y[i] - f2[i], but only if positive
            heights = right_min_y - f2_vals
            
            # Clip negative values to 0
            widths = np.maximum(widths, 0.0)
            heights = np.maximum(heights, 0.0)
            
            contribs = widths * heights
            
            # Only valid points contribute (f1 < ref[0] and f2 < ref[1])
            valid_mask = (f1_vals < ref_pt[0]) & (f2_vals < ref_pt[1])
            contribs[~valid_mask] = 0.0
            
            return contribs

        def sample_uniform():
            return lo + (hi - lo) * np.random.random(self.dim)

        def sample_gaussian(parent, sigma_scale):
            sigma = sigma_scale * (hi - lo)
            child = parent + sigma * np.random.standard_normal(self.dim)
            return np.clip(child, lo, hi)

        # Phase 1: Latin Hypercube Sampling for better initial coverage
        n_init = min(50, self.budget // 3)
        for i in range(n_init):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # LHS: each dimension gets a unique stratified bin
                bin_idx = (i + d) % n_init
                bin_start = bin_idx / n_init
                bin_end = (bin_idx + 1) / n_init
                x[d] = lo[d] + (hi[d] - lo[d]) * (bin_start + np.random.random() * (bin_end - bin_start))
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        # Phase 2: HypE-style selection and mutation
        mutation_sigma = 0.2
        min_sigma = 0.003
        max_sigma = 0.35

        while used < self.budget:
            if time.time() > deadline:
                break

            if len(archive) >= 2:
                contribs = marginal_contributions_fast(archive, ref)
                contribs = np.maximum(contribs, 1e-12)
                probs = contribs / contribs.sum()
                idx = np.random.choice(len(archive), p=probs)
            elif len(archive) == 1:
                idx = 0
            else:
                x = sample_uniform()
                f = problem(x)
                used += 1
                add_to_archive(x, f)
                continue

            parent_x, parent_f = archive[idx]

            r = np.random.random()
            if r < 0.15:
                x = sample_uniform()
            else:
                x = sample_gaussian(parent_x, mutation_sigma)

            f = problem(x)
            used += 1
            add_to_archive(x, f)

            # Adapt sigma: shrink slowly, but expand if archive is small
            if len(archive) < 10:
                mutation_sigma = min(max_sigma, mutation_sigma * 1.01)
            else:
                mutation_sigma = max(min_sigma, mutation_sigma * 0.995)

        # Phase 3: Local refinement on top archive points using scipy minimize
        if used < self.budget and len(archive) > 0 and time.time() < deadline:
            # Pick top-k points by marginal contribution for local search
            if len(archive) >= 2:
                contribs = marginal_contributions_fast(archive, ref)
                top_k = np.argsort(contribs)[::-1][:min(8, len(archive))]
            else:
                top_k = [0]
            
            for idx in top_k:
                if used >= self.budget or time.time() > deadline:
                    break
                parent_x, parent_f = archive[idx]
                
                # Try a few local perturbations with decreasing step size
                for trial in range(5):
                    if used >= self.budget or time.time() > deadline:
                        break
                    step = 0.05 * (0.5 ** trial)
                    x = parent_x + step * (hi - lo) * np.random.standard_normal(self.dim)
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    used += 1
                    add_to_archive(x, f)
                
                # Also try scipy local optimization on a weighted scalarization
                if used < self.budget and time.time() < deadline:
                    weights = np.array([0.5, 0.5])
                    def scalarize(x):
                        fx = problem(x)
                        return weights[0] * fx[0] + weights[1] * fx[1]
                    
                    try:
                        res = minimize(scalarize, parent_x, method='Nelder-Mead',
                                     bounds=list(zip(lo, hi)),
                                     options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4})
                        if used < self.budget:
                            f = problem(res.x)
                            used += 1
                            add_to_archive(res.x, f)
                    except:
                        pass

        return archive
# EVOLVE-BLOCK-END
