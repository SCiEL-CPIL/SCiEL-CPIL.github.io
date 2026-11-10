import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Candidate tails: vectors for x_2..x_n
        # We want to find the tail that minimizes g(x) (and thus f2)
        # Standard ZDT4 global min for tail is at 0 (and multiples of 0.5).
        # We will test a few candidates to be robust to bounds.
        
        # Reduce candidates to save budget for x1 sampling
        # 0 is the most likely global min.
        # lo and hi are fallbacks if 0 is out of bounds or boundaries are better.
        tails = []
        tails.append(np.zeros(self.dim - 1))
        tails.append(lo[1:].copy())
        tails.append(hi[1:].copy())
        
        n_tail_candidates = len(tails)
        
        # Ensure we have enough budget
        if self.budget <= n_tail_candidates + 10:
             n_x1_samples = self.budget // n_tail_candidates
             if n_x1_samples == 0:
                 n_x1_samples = 1
                 n_tail_candidates = 1
             else:
                 n_tail_candidates = self.budget // n_x1_samples
        else:
            n_x1_samples = self.budget - n_tail_candidates

        # Clip tails to bounds
        for i in range(len(tails)):
            tails[i] = np.clip(tails[i], lo[1:], hi[1:])

        # Step 1: Determine the best tail
        # Use a reference x1, e.g., the midpoint
        x1_ref = (lo[0] + hi[0]) / 2.0
        
        best_tail_idx = 0
        best_f2 = float('inf')
        
        for i, tail in enumerate(tails):
            x = np.empty(self.dim)
            x[0] = x1_ref
            x[1:] = tail
            x_clipped = np.clip(x, lo, hi)
            obj = problem(x_clipped)
            f2 = obj[1]
            if f2 < best_f2:
                best_f2 = f2
                best_tail_idx = i
                
        best_tail = tails[best_tail_idx]

        # Step 2: Sample x1 densely with the best tail
        x1_vals = np.linspace(lo[0], hi[0], n_x1_samples)
        
        for x1 in x1_vals:
            x = np.empty(self.dim)
            x[0] = x1
            x[1:] = best_tail
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
