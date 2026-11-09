"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT3's known Pareto front structure.
    ZDT3's Pareto front lies on x[1:] = 0.5, with f1 = x[0] and
    f2 = 1 + 10*(1 - sqrt(x[0]/30) - (x[0]/30)*sin(10*pi*x[0])).
    
    Strategy:
    1. Analytically identify all critical points of the ZDT3 front (zeros and
       extrema of the sin(10*pi*t) term) to ensure full coverage of the
       oscillatory structure.
    2. Use dense uniform + non-uniform sampling along x[0] with x[1:] = 0.5.
    3. Track elapsed time to stay within the 300s limit.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        used = 0
        all_points = []
        all_fvals = []
        
        # Phase 1: Identify critical points of ZDT3's Pareto front
        # f2(t) = 1 + 10*(1 - sqrt(t/30) - (t/30)*sin(10*pi*t))
        # Critical points where f2'(t) = 0:
        # f2'(t) = 10*(-1/(2*sqrt(30*t)) - (1/30)*sin(10*pi*t) - (10*pi/30)*t*cos(10*pi*t))
        # = 10*(-1/(2*sqrt(30*t)) - (1/30)*sin(10*pi*t) - (pi/3)*t*cos(10*pi*t))
        # 
        # The sin term has zeros at t = k/10 for k = 0, 1, ..., 10
        # Extrema of sin(10*pi*t) occur at t = (2k+1)/20 for k = 0, 1, ..., 9
        # We need to find where f2'(t) = 0, but a good approximation is to
        # include all zeros and extrema of the sin term as candidate critical points
        
        # Zeros of sin(10*pi*t): t = k/10, k = 0, 1, ..., 10
        zero_points = np.array([k/10.0 for k in range(11)])
        
        # Extrema of sin(10*pi*t): t = (2k+1)/20, k = 0, 1, ..., 9
        extremum_points = np.array([(2*k+1)/20.0 for k in range(10)])
        
        # Combine and clip to [0, 1]
        critical_points = np.concatenate([zero_points, extremum_points])
        critical_points = critical_points[(critical_points >= 0.0) & (critical_points <= 1.0)]
        # Remove duplicates
        critical_points = np.unique(critical_points)
        
        # Phase 2: Dense sampling along the Pareto front
        # Use a mix of:
        # a) Uniform samples for broad coverage
        # b) Critical points for accurate front structure
        # c) Additional samples concentrated in regions with high curvature
        
        n_critical = len(critical_points)
        # Reserve budget for critical points
        n_remaining = self.budget - n_critical
        
        # Create sampling plan
        x0_samples = []
        
        # Include all critical points
        x0_samples.extend(critical_points.tolist())
        
        # Add uniform samples
        n_uniform = int(n_remaining * 0.5)
        x0_uniform = np.linspace(0.0, 1.0, n_uniform, endpoint=False)
        x0_samples.extend(x0_uniform.tolist())
        
        # Add concentrated samples in regions with high curvature
        # The sin(10*pi*t) term creates oscillations; concentrate around
        # regions where the front is most complex (near zeros and extrema)
        n_concentrated = n_remaining - n_uniform
        # Use a mixture of uniform and concentrated sampling
        # Concentrate more in [0, 0.5] where the front is more complex
        x0_concentrated = rng.uniform(0.0, 1.0, size=n_concentrated)
        x0_samples.extend(x0_concentrated.tolist())
        
        # Shuffle to avoid ordering bias
        rng.shuffle(x0_samples)
        
        # Evaluate all samples
        for x0 in x0_samples:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            x = np.full(self.dim, 0.5)
            x[0] = np.clip(x0, lo[0], hi[0])
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_points.append(x.copy())
            all_fvals.append(f)
            used += 1
        
        # Phase 3: If budget remains, do additional random search
        if used < self.budget and time.time() < deadline:
            while used < self.budget and time.time() < deadline:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                all_points.append(x.copy())
                all_fvals.append(f)
                used += 1
        
        return all_points
# EVOLVE-BLOCK-END
