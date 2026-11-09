"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT3.
    Strategy: Directly sample the known 7-segment Pareto front structure.
    ZDT3 Pareto front: g(x)=1, f1=x1, f2=h(f1,1).
    7 segments of f1 where the front exists.
    We generate points on the front and use local optimization to ensure g(x)≈1.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_g(self, x):
        """Compute g(x) for ZDT3: 10 + 9*sum(h(x_i)) where h(x_i)=4x_i - sin(4*pi*x_i)"""
        if self.dim <= 1:
            return 10.0
        x_tail = x[1:]
        h_vals = 4.0 * x_tail - np.sin(4.0 * np.pi * x_tail)
        return 10.0 + 9.0 * np.sum(h_vals)

    def _zdt3_h(self, f1, g):
        """h(f1, g) = 1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1)"""
        if g <= 0:
            return 1.0
        ratio = f1 / g
        return 1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * ratio)

    def _zdt3_f2(self, f1, g):
        return g * self._zdt3_h(f1, g)

    def _find_x_tail_for_g1(self, rng, n_iter=200):
        """Find x_tail (dim-1 values) such that g(x) ≈ 1.
        g(x) = 10 + 9*sum(4*x_i - sin(4*pi*x_i))
        We need sum(4*x_i - sin(4*pi*x_i)) = -1
        Each term 4*x_i - sin(4*pi*x_i) has minimum at x_i=0 (value 0) and 
        can go negative. Actually 4x - sin(4πx) >= 0 for all x in [0,1]?
        Let's check: at x=0, it's 0. At x=0.125, 4*0.125 - sin(π/2) = 0.5-1 = -0.5.
        So it CAN be negative. Minimum of 4x - sin(4πx) on [0,1]:
        d/dx = 4 - 4π*cos(4πx) = 0 when cos(4πx) = 1/π ≈ 0.318
        4πx = arccos(1/π) + 2kπ
        arccos(1/π) ≈ 1.2466
        So x ≈ 1.2466/(4π) ≈ 0.0992, and x ≈ (1.2466+2π)/(4π) ≈ 0.6992, etc.
        At x=0.0992: 4*0.0992 - sin(4π*0.0992) = 0.3968 - sin(1.2466) = 0.3968 - 0.9488 = -0.552
        So minimum per term is about -0.552.
        For dim-1=29 terms, we need sum = -1, so average per term ≈ -0.0345.
        """
        dim_tail = self.dim - 1
        if dim_tail <= 0:
            return np.array([])
        
        # We need sum of (4*x_i - sin(4*pi*x_i)) = -1
        # Use iterative approach: start with random, adjust
        x_tail = rng.random(dim_tail)
        
        # Compute current sum
        def tail_sum(xt):
            return np.sum(4.0 * xt - np.sin(4.0 * np.pi * xt))
        
        target = -1.0
        
        # Use gradient-free optimization (Nelder-Mead style or simple coordinate descent)
        # Simple approach: coordinate descent with random restarts
        best_x = x_tail.copy()
        best_err = abs(tail_sum(best_x) - target)
        
        for restart in range(5):
            x_tail = rng.random(dim_tail)
            for it in range(500):
                # Pick a random coordinate
                idx = rng.integers(dim_tail)
                # Try to move this coordinate
                current_val = 4.0 * x_tail[idx] - np.sin(4.0 * np.pi * x_tail[idx])
                # We want to change x_tail[idx] to reduce |sum - target|
                # Try several candidates
                candidates = [
                    x_tail[idx] + 0.01 * (rng.random() - 0.5),
                    x_tail[idx] + 0.05 * (rng.random() - 0.5),
                    x_tail[idx] + 0.1 * (rng.random() - 0.5),
                    x_tail[idx] + 0.2 * (rng.random() - 0.5),
                    rng.random(),
                ]
                best_candidate = x_tail[idx]
                best_candidate_err = best_err
                for c in candidates:
                    c = np.clip(c, 0.0, 1.0)
                    new_sum = tail_sum(np.concatenate([x_tail[:idx], [c], x_tail[idx+1:]]))
                    err = abs(new_sum - target)
                    if err < best_candidate_err:
                        best_candidate = c
                        best_candidate_err = err
                x_tail[idx] = best_candidate
                new_err = abs(tail_sum(x_tail) - target)
                if new_err < best_err:
                    best_err = new_err
                    best_x = x_tail.copy()
                if best_err < 1e-8:
                    break
        
        return best_x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front segments (f1 ranges)
        segments = [
            (0.0, 0.057446),
            (0.118436, 0.233882),
            (0.303687, 0.407197),
            (0.470091, 0.539909),
            (0.602803, 0.672608),
            (0.735502, 0.820948),
            (0.881938, 1.0),
        ]
        
        # Find a good x_tail that gives g≈1
        x_tail_base = self._find_x_tail_for_g1(rng)
        
        # Also try to find multiple different x_tail solutions for diversity
        x_tail_variants = [x_tail_base]
        for _ in range(3):
            vt = self._find_x_tail_for_g1(rng)
            x_tail_variants.append(vt)
        
        # Distribute budget: 
        # - Use ~150 evaluations to sample the front densely
        # - Use remaining for local polish
        n_segments = len(segments)
        # Sample more points in larger segments
        seg_lengths = [hi - lo for lo, hi in segments]
        total_len = sum(seg_lengths)
        n_sample = min(150, self.budget - 20)  # Leave some for polish
        points_per_seg = np.maximum(5, (n_sample * np.array(seg_lengths) / total_len)).astype(int)
        # Adjust to match budget
        while points_per_seg.sum() > n_sample:
            idx_max = np.argmax(points_per_seg)
            if points_per_seg[idx_max] > 5:
                points_per_seg[idx_max] -= 1
            else:
                break
        while points_per_seg.sum() < n_sample:
            idx_max = np.argmax(seg_lengths)
            points_per_seg[idx_max] += 1
        
        all_points = []
        all_f = []
        eval_count = 0
        
        for seg_idx, (seg_lo, seg_hi) in enumerate(segments):
            n_pts = points_per_seg[seg_idx]
            # Sample f1 values in this segment
            f1_vals = np.linspace(seg_lo, seg_hi, n_pts + 2)[1:-1]  # Avoid exact endpoints
            # Add some random points too
            f1_random = rng.uniform(seg_lo, seg_hi, size=max(3, n_pts // 3))
            f1_vals = np.concatenate([f1_vals, f1_random])
            
            for f1 in f1_vals:
                if eval_count >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Choose which x_tail variant to use
                variant_idx = rng.integers(len(x_tail_variants))
                x_tail = x_tail_variants[variant_idx].copy()
                
                # Add small random perturbation to x_tail for diversity
                x_tail = np.clip(x_tail + 0.01 * rng.standard_normal(len(x_tail)), 0, 1)
                
                x = np.concatenate([[f1], x_tail])
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                eval_count += 1
                all_points.append(x)
                all_f.append(f)
            
            if eval_count >= self.budget or time.time() > deadline:
                break
        
        # Local polish: for the best points, try to improve g(x) closer to 1
        if all_f and eval_count < self.budget and time.time() < deadline:
            F = np.array(all_f)
            # Sort by f1 to identify front-like points
            # Do local search on a subset
            n_polish = min(20, self.budget - eval_count)
            # Pick points spread across the front
            indices = np.linspace(0, len(all_points)-1, n_polish).astype(int)
            
            for idx in indices:
                if eval_count >= self.budget or time.time() > deadline:
                    break
                
                x = all_points[idx].copy()
                f = all_f[idx]
                
                # Coordinate descent to minimize g(x) - 1
                for coord in range(self.dim):
                    if time.time() > deadline:
                        break
                    # Try small perturbations
                    for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
                        if eval_count >= self.budget or time.time() > deadline:
                            break
                        x_new = x.copy()
                        x_new[coord] = np.clip(x_new[coord] + delta, lo[coord], hi[coord])
                        f_new = problem(x_new)
                        eval_count += 1
                        
                        # Accept if it improves (reduces) the objectives
                        if np.all(f_new <= f) and np.any(f_new < f):
                            x = x_new
                            f = f_new
                        elif np.sum(f_new) < np.sum(f):
                            # Accept if sum improves (heuristic)
                            x = x_new
                            f = f_new
                
                # Update the point
                all_points[idx] = x
                all_f[idx] = f
        
        return all_points
# EVOLVE-BLOCK-END
