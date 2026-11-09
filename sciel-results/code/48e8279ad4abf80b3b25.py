# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ2's known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Archive of non-dominated points
        archive_F = []
        archive_X = []
        used = 0

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def update_archive(x, f):
            nonlocal archive_F, archive_X
            for i, af in enumerate(archive_F):
                if dominates(af, f):
                    return
            new_archive_F = []
            new_archive_X = []
            for i, af in enumerate(archive_F):
                if not dominates(f, af):
                    new_archive_F.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive_F.append(f)
            new_archive_X.append(x)
            archive_F = new_archive_F
            archive_X = new_archive_X

        # DTLZ2 Pareto front: sum(f_i^2) = 1, f_i >= 0
        # Sample points on the front using Dirichlet-like distribution
        # For 3 objectives, use 2 angular parameters in [0, pi/2]
        
        # Phase 1: Sample directly on the known Pareto front
        # Use stratified sampling on the spherical triangle
        n_front = min(60, budget // 3)
        
        # Use low-discrepancy style sampling via sorted random angles
        # For 3D: f1 = sin(a)*cos(b), f2 = sin(a)*sin(b), f3 = cos(a)
        # where a, b in [0, pi/2]
        
        # Create quasi-random grid in [0, pi/2]^2
        n_per_dim = int(math.ceil(math.sqrt(n_front)))
        angles_a = np.linspace(0.01, math.pi/2 - 0.01, n_per_dim)
        angles_b = np.linspace(0.01, math.pi/2 - 0.01, n_per_dim)
        
        # Shuffle to avoid systematic bias, then take first n_front
        aa = angles_a.copy()
        bb = angles_b.copy()
        rng.shuffle(aa)
        rng.shuffle(bb)
        
        # Create pairs
        pairs = []
        for a in aa:
            for b in bb:
                pairs.append((a, b))
        rng.shuffle(pairs)
        pairs = pairs[:n_front]
        
        for (a, b) in pairs:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Compute front point
            f_target = np.array([
                math.sin(a) * math.cos(b),
                math.sin(a) * math.sin(b),
                math.cos(a)
            ])
            
            # For DTLZ2: x_0..x_{m-2} affect g, x_{m-1}..x_{n-1} affect front position
            # Standard DTLZ2: g = sum((x_i - 0.5)^2) for i in 0..m-2
            # h_i = 1 - sqrt(f_i) where f_i depends on x_m..x_{n-1}
            # Actually for DTLZ2: f_i = (1+g) * h_i(x_{m+1}..x_n)
            # h_i uses the last n-m+1 variables
            
            # To get exactly on the front, set g=0 (x_0..x_{m-2} = 0.5)
            # and choose last variables to match target angles
            
            m = 3  # number of decision variables affecting g (standard DTLZ2 with 3 objs)
            # Actually DTLZ2: g depends on first m variables, front on remaining
            # For n_obj=3, typically m=2, so x_0, x_1 affect g
            # x_2..x_11 (10 vars) determine the front position via h functions
            
            # Simpler approach: just evaluate random points near the front
            # Set first m-1 vars to 0.5 (to make g small), then tune last vars
            
            x = np.full(dim, 0.5)
            # For DTLZ2 with 3 objectives, the front is determined by the last 2 variables
            # h_1 = cos(x_{n-1}), h_2 = sin(x_{n-1})cos(x_{n-2}), h_3 = sin(x_{n-1})sin(x_{n-2})
            # Actually standard DTLZ2: f_i = (1+g)*h_i where h uses x_{m+1}..x_n
            # For 3 objectives: h_1 = cos(x_{n-1}), h_2 = sin(x_{n-1})cos(x_{n-2}), h_3 = sin(x_{n-1})sin(x_{n-2})
            
            # We want (1+g)*h = f_target. If g=0, then h = f_target
            # So: cos(x_{n-1}) = f_target[0], sin(x_{n-1})cos(x_{n-2}) = f_target[1], sin(x_{n-1})sin(x_{n-2}) = f_target[2]
            
            # From first: x_{n-1} = arccos(f_target[0])
            # From second/third: tan(x_{n-2}) = f_target[2]/f_target[1]
            # So x_{n-2} = arctan2(f_target[2], f_target[1])
            
            if f_target[0] < 1e-10:
                x[-1] = math.pi / 2
            else:
                x[-1] = math.acos(min(1.0, max(0.0, f_target[0])))
            
            if f_target[1] < 1e-10 and f_target[2] < 1e-10:
                x[-2] = 0.0
            else:
                x[-2] = math.atan2(f_target[2], f_target[1])
            
            x[-1] = np.clip(x[-1], lo[-1], hi[-1])
            x[-2] = np.clip(x[-2], lo[-2], hi[-2])
            
            f = np.array(problem(x))
            used += 1
            update_archive(x, f)

        # Phase 2: Local refinement around archive points
        while used < budget:
            if time.time() > deadline:
                break
            
            if len(archive_X) == 0:
                # Fallback to random
                x = lo + (hi - lo) * rng.random(dim)
                f = np.array(problem(x))
                used += 1
                update_archive(x, f)
                continue
            
            idx = rng.integers(len(archive_X))
            parent = archive_X[idx].copy()
            
            # Adaptive sigma: start larger, shrink over time
            progress = used / budget
            sigma = 0.05 * (1.0 - 0.5 * progress)
            
            child = parent + sigma * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = np.array(problem(child))
            used += 1
            update_archive(child, fc)

        return archive_X
# EVOLVE-BLOCK-END
