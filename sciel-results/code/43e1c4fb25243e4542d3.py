import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 280  # Safety margin
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            span = hi - lo
            
            # 1. LHS Initialization
            n_init = min(self.budget // 2, 100) # Use half budget for init, cap at 100
            # Generate LHS
            lhs = np.zeros((n_init, self.dim))
            for i in range(self.dim):
                # Create n_init intervals
                edges = np.linspace(lo[i], hi[i], n_init + 1)
                # Random points in each interval
                midpoints = (edges[:-1] + edges[1:]) / 2
                # Uniform random offset within interval
                offsets = (edges[1:] - edges[:-1]) * np.random.random(n_init)
                lhs[:, i] = midpoints + offsets
                # Shuffle
                np.random.shuffle(lhs[:, i])
            
            # Evaluate Init
            front_x = []
            front_f = []
            for x in lhs:
                if time.time() > deadline: break
                f = np.array(problem(x))
                # Check dominance against current front
                dominated = False
                for i in range(len(front_x)):
                    if np.all(front_f[i] <= f) and np.any(front_f[i] < f):
                        dominated = True
                        break
                if not dominated:
                    # Check if it dominates existing points
                    new_idx = []
                    for i in range(len(front_x)):
                        if np.all(f <= front_f[i]) and np.any(f < front_f[i]):
                            new_idx.append(i)
                    for i in reversed(new_idx):
                        del front_x[i]
                        del front_f[i]
                    front_x.append(x)
                    front_f.append(f)
            
            used = n_init
            
            # 2. Refinement
            # Adaptive step size
            step_size = 0.1 # 10% of range
            
            while used < self.budget:
                if time.time() > deadline: break
                if not front_x:
                    # Fallback to random if front is empty (shouldn't happen)
                    x = lo + span * np.random.random(self.dim)
                else:
                    # Pick a random parent from the front
                    idx = np.random.randint(len(front_x))
                    parent = front_x[idx]
                    # Gaussian mutation
                    x = parent + np.random.normal(0, step_size, self.dim)
                    x = np.clip(x, lo, hi)
                
                f = np.array(problem(x))
                used += 1
                
                # Update Front
                dominated = False
                for i in range(len(front_x)):
                    if np.all(front_f[i] <= f) and np.any(front_f[i] < f):
                        dominated = True
                        break
                
                if not dominated:
                    new_idx = []
                    for i in range(len(front_x)):
                        if np.all(f <= front_f[i]) and np.any(f < front_f[i]):
                            new_idx.append(i)
                    for i in reversed(new_idx):
                        del front_x[i]
                        del front_f[i]
                    front_x.append(x)
                    front_f.append(f)
                    
                    # Optionally decrease step size as we converge?
                    # Let's keep it simple first.
            
            return front_x
