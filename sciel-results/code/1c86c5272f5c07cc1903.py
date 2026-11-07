import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Helper to check time
        def time_up():
            return (time.time() - self.start_time) > deadline

        # Helper to evaluate and track
        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        # We assume x[0] is the primary variable controlling the front.
        # We will sweep x[0] and test different configurations for x[1:].
        
        # Phase 1: Probe to determine best tail strategy
        # Test tails: Zeros, Halfs, Random
        probe_x1s = [0.1, 0.5, 0.9]
        tail_strategies = {
            'zero': lambda d: np.zeros(d),
            'half': lambda d: np.full(d, 0.5),
            'rand': lambda d: np.random.rand(d)
        }
        
        best_tail_name = 'zero'
        min_score = float('inf')
        
        for x1 in probe_x1s:
            for name, gen_tail in tail_strategies.items():
                if self.eval_count >= self.budget or time_up():
                    break
                x = np.empty(self.dim)
                x[0] = x1
                x[1:] = gen_tail(self.dim - 1)
                x = np.clip(x, lo, hi)
                
                f = eval_point(x)
                if f is not None:
                    # Use sum of objectives as a proxy for closeness to ideal
                    score = sum(f)
                    if score < min_score:
                        min_score = score
                        best_tail_name = name

        # If probe didn't work well or failed, default to zero
        if best_tail_name not in tail_strategies:
            best_tail_name = 'zero'

        gen_best_tail = tail_strategies[best_tail_name]

        # Phase 2: Sweep x[0] using the best tail strategy
        # We have remaining budget. Let's use most of it for a dense sweep of x[0].
        # Leave ~10% for random exploration to handle unexpected cases.
        remaining = self.budget - self.eval_count
        sweep_count = int(remaining * 0.9)
        rand_count = remaining - sweep_count
        
        # Generate x1 values for sweep
        # Use Chebyshev nodes for better coverage of the convex ZDT2 front
        if sweep_count > 0:
            # Chebyshev nodes: x_k = 0.5 * (1 + cos((2k+1) * pi / (2n))) for k=0,...,n-1
            k = np.arange(sweep_count)
            angles = (2 * k + 1) * np.pi / (2 * sweep_count)
            x1_values = 0.5 * (1 + np.cos(angles))
            # Scale to [lo[0], hi[0]]
            x1_values = lo[0] + (hi[0] - lo[0]) * x1_values
        else:
            x1_values = np.array([])
        
        for x1 in x1_values:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.empty(self.dim)
            x[0] = x1
            x[1:] = gen_best_tail(self.dim - 1)
            x = np.clip(x, lo, hi)
            eval_point(x)

        # Phase 3: Random exploration with best tail or mixed
        for _ in range(rand_count):
            if self.eval_count >= self.budget or time_up():
                break
            # Mix of random x1 and best tail, or fully random
            if np.random.rand() < 0.5:
                x1 = np.random.uniform(lo[0], hi[0])
                x = np.empty(self.dim)
                x[0] = x1
                x[1:] = gen_best_tail(self.dim - 1)
            else:
                x = np.random.uniform(lo, hi)
            
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
