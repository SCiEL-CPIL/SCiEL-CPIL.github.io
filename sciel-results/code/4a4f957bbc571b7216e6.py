"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided local search with archive-based parent selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Reference point for hypervolume calculation
        ref = np.array([3.0, 5.0, 7.0])
        
        # Archive of non-dominated solutions
        archive = []  # list of (x, f)
        
        def hv_contribution(f_new, archive_f):
            """Calculate hypervolume contribution of f_new given existing archive points."""
            # For 3 objectives, contribution is the volume of the region dominated by f_new
            # but not dominated by any point in the archive.
            # Simple approximation: volume of box from f_new to ref, minus overlaps.
            # For efficiency with small archives, we use a simple box volume as proxy.
            if any(f_new >= ref):
                return 0.0
            # Volume of box from f_new to ref
            vol = 1.0
            for i in range(self.n_obj):
                if f_new[i] < ref[i]:
                    vol *= (ref[i] - f_new[i])
                else:
                    return 0.0
            return vol
        
        def is_dominated(f_new, archive_f):
            """Check if f_new is dominated by any point in archive."""
            for f in archive_f:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False
        
        def remove_dominated(archive, new_idx):
            """Remove points dominated by the new point at new_idx."""
            f_new = archive[new_idx][1]
            to_remove = []
            for i in range(len(archive)):
                if i == new_idx:
                    continue
                f_i = archive[i][1]
                if np.all(f_new <= f_i) and np.any(f_new < f_i):
                    to_remove.append(i)
            # Remove in reverse order to maintain indices
            for i in sorted(to_remove, reverse=True):
                archive.pop(i)
        
        # Initialize with random points
        init_size = min(20, self.budget // 5)
        for _ in range(init_size):
            x = lo + (hi - lo) * rng.random(self.dim)
            f = np.array(problem(x))
            if not any(f >= ref):  # Only add if potentially useful
                archive.append((x, f))
        
        used = init_size
        
        # Sort initial archive by hypervolume contribution (descending)
        # We'll maintain a simple archive and select parents based on their "value"
        
        while used < self.budget:
            # Time guard
            if time.time() > deadline:
                break
            
            if len(archive) == 0:
                # Fallback: random restart
                x = lo + (hi - lo) * rng.random(self.dim)
                f = np.array(problem(x))
                archive.append((x, f))
                used += 1
                continue
            
            # Select parent: choose from archive with probability proportional to hypervolume contribution
            # Approximate contribution as box volume
            contributions = []
            for i, (x, f) in enumerate(archive):
                if any(f >= ref):
                    contributions.append(0.0)
                else:
                    vol = 1.0
                    for j in range(self.n_obj):
                        vol *= max(0, ref[j] - f[j])
                    contributions.append(vol)
            
            total = sum(contributions)
            if total <= 0:
                # All points are beyond reference; pick randomly
                parent_idx = rng.integers(len(archive))
            else:
                probs = np.array(contributions) / total
                parent_idx = rng.choice(len(archive), p=probs)
            
            parent_x, parent_f = archive[parent_idx]
            
            # Adaptive mutation: step size based on how "good" the parent is
            # Better parents (lower objective values) get smaller steps for exploitation
            # Use normalized distance to ideal point
            ideal = np.min([f for _, f in archive], axis=0)
            worst = np.max([f for _, f in archive], axis=0)
            span = worst - ideal
            span[span < 1e-10] = 1.0  # Avoid division by zero
            
            # Normalized position: 0 = best, 1 = worst
            norm_pos = np.sum((parent_f - ideal) / span) / self.n_obj
            norm_pos = np.clip(norm_pos, 0, 1)
            
            # Step size: larger for worse points (exploration), smaller for better (exploitation)
            base_step = 0.15 * (1.0 - norm_pos * 0.7)
            step_size = max(0.02, base_step)
            
            # Gaussian mutation
            mutation = step_size * (hi - lo) * rng.standard_normal(self.dim)
            child_x = parent_x + mutation
            child_x = np.clip(child_x, lo, hi)
            
            child_f = np.array(problem(child_x))
            used += 1
            
            # Check if child is useful (not dominated by archive and not beyond ref)
            if any(child_f >= ref):
                continue
            
            if is_dominated(child_f, [f for _, f in archive]):
                continue
            
            # Add to archive and remove dominated points
            archive.append((child_x, child_f))
            remove_dominated(archive, len(archive) - 1)
            
            # Limit archive size to prevent memory issues
            if len(archive) > 50:
                # Keep top 50 by hypervolume contribution
                scored = []
                for i, (x, f) in enumerate(archive):
                    if any(f >= ref):
                        scored.append((0.0, i))
                    else:
                        vol = 1.0
                        for j in range(self.n_obj):
                            vol *= max(0, ref[j] - f[j])
                        scored.append((vol, i))
                scored.sort(reverse=True)
                keep_indices = [i for _, i in scored[:50]]
                archive = [archive[i] for i in keep_indices]
        
        return archive
# EVOLVE-BLOCK-END
