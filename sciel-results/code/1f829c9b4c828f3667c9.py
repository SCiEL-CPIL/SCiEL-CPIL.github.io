"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Structure-exploiting optimizer for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse random sampling to detect hidden shift
        # Büche-Rastrigin: f(x) = 10*n + sum((x_i - 1)^2 - 10*cos(2*pi*(x_i-1)))
        # Local minima at x_i = 1 + 2*pi*k for integers k
        # With hidden shift s and rotation R: minima at R^T * (1 + 2*pi*k) + s
        # We sample to find the shift by looking at where periodicity aligns
        
        n_phase1 = min(50, max_calls // 10)
        samples = []
        for i in range(n_phase1):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)
            samples.append((f, x.copy()))
        
        samples.sort(key=lambda t: t[0])
        
        # Phase 2: Generate candidate minima using detected structure
        # The Büche-Rastrigin function has minima at integer multiples of 2*pi
        # shifted by the hidden offset. We generate candidates at the lattice
        # points and evaluate the best ones.
        
        # First, try to detect the shift by evaluating at known lattice points
        # The unshifted minima are at x_i = 1 + 2*pi*k
        # With shift s, they're at x_i = s_i + 1 + 2*pi*k
        
        # Generate a grid of candidate minima
        # For each dimension, possible k values such that s_i + 1 + 2*pi*k is in [-5, 5]
        # Without knowing s exactly, we try multiple hypotheses
        
        candidates = []
        
        # Strategy: Use the best samples to estimate the shift
        # For each best sample, check which lattice point it's closest to
        # This gives us an estimate of the shift
        
        # First, let's try a simpler approach: generate candidates at
        # regular grid points that are likely to be near minima
        
        # The period is 2*pi ≈ 6.283
        # In [-5, 5], we have about 10/6.283 ≈ 1.6 periods per dimension
        # So k can be -1, 0, 1 for each dimension (with shift)
        
        # Generate all combinations of lattice points
        # x_i = base_i + 2*pi*k_i where base_i is the shift component
        # We don't know the shift, so we try to infer it from samples
        
        # Better approach: For each sample, compute the "phase" in each dimension
        # phase_i = (x_i - 1) % (2*pi) normalized to [-pi, pi]
        # If the sample is near a minimum, phase should be near 0
        
        # Let's try a different structural approach:
        # 1. Use random sampling to find good regions
        # 2. For each good region, generate nearby lattice points
        # 3. Evaluate and refine the best candidates
        
        # Phase 2a: Generate candidate points near the best samples
        n_top = min(10, len(samples))
        for f_init, x_init in samples[:n_top]:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            # For each dimension, generate nearby lattice points
            # Lattice points are at 1 + 2*pi*k (unshifted)
            # With shift s, they're at s + 1 + 2*pi*k
            # We estimate s from the sample: s ≈ x_init - (1 + 2*pi*k_est)
            # where k_est = round((x_init - 1) / (2*pi))
            
            for d in range(dim):
                # Estimate the shift in this dimension from the sample
                k_est = round((x_init[d] - 1.0) / (2 * np.pi))
                s_est = x_init[d] - (1.0 + 2 * np.pi * k_est)
                
                # Generate candidate points with this estimated shift
                for k in range(-2, 3):  # k = -2, -1, 0, 1, 2
                    x_candidate = x_init.copy()
                    x_candidate[d] = s_est + 1.0 + 2 * np.pi * k
                    if -5 <= x_candidate[d] <= 5:
                        candidates.append(x_candidate.copy())
        
        # Also add pure lattice points assuming zero shift as baseline
        for k_combo in np.ndindex(*[(2 * np.pi * np.arange(-2, 3)) for _ in range(dim)]):
            # This is too many, so just add a few
            pass
        
        # Phase 2b: Add random candidates for diversity
        n_random_extra = min(30, max(10, (max_calls - call_count) // 10))
        for i in range(n_random_extra):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            candidates.append(x)
        
        # Deduplicate candidates (keep unique points)
        unique_candidates = []
        seen = set()
        for x in candidates:
            key = tuple(np.round(x, 3))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(x)
        
        # Phase 3: Evaluate candidates and keep the best
        eval_results = []
        for x in unique_candidates:
            if call_count >= max_calls or time.time() > deadline:
                break
            f = counted_func(x)
            eval_results.append((f, x.copy()))
        
        eval_results.sort(key=lambda t: t[0])
        
        # Phase 4: Local refinement using line search on the best candidates
        n_refine = min(15, len(eval_results))
        for f_init, x_init in eval_results[:n_refine]:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            x = x_init.copy()
            f_curr = f_init
            max_iter_local = 30
            
            for iter_local in range(max_iter_local):
                if call_count >= max_calls or time.time() > deadline:
                    break
                
                improved = False
                best_dim_f = f_curr
                best_dim_x = None
                best_dim = None
                
                # Try coordinate-wise moves with adaptive steps
                for d in range(dim):
                    # Try multiple step sizes, including ridge-jumping steps
                    step_sizes = [0.5, 1.0, 2.0, 3.0, 5.0, 2*np.pi]
                    for step in step_sizes:
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < best_dim_f:
                                best_dim_f = f_trial
                                best_dim_x = x_trial.copy()
                                best_dim = d
                                improved = True
                    
                    if call_count >= max_calls or time.time() > deadline:
                        break
                
                if not improved:
                    # Try smaller steps
                    for d in range(dim):
                        for step in [0.1, 0.2, 0.05]:
                            for direction in [-1, 1]:
                                x_trial = x.copy()
                                x_trial[d] += direction * step
                                if x_trial[d] < -5 or x_trial[d] > 5:
                                    continue
                                f_trial = counted_func(x_trial)
                                if f_trial < best_dim_f:
                                    best_dim_f = f_trial
                                    best_dim_x = x_trial.copy()
                                    best_dim = d
                                    improved = True
                            if call_count >= max_calls or time.time() > deadline:
                                break
                        if call_count >= max_calls or time.time() > deadline:
                            break
                    if not improved:
                        break
                
                x = best_dim_x
                f_curr = best_dim_f
                
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
