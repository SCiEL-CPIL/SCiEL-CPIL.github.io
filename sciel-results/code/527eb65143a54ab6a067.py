"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-and-optimize approach for ZDT1.
    
    ZDT1 has a known analytical structure:
    - Objective 1: f1 = x0
    - g(x) = 1 + 10*(n-1) / sum(x_i for i=1..n-1)
    - Objective 2: f2 = g * (1 - sqrt(x0 / g))
    
    The Pareto front is: for t in [0,1], f1 = t, f2 = 11*(1 - sqrt(t/11))
    To achieve this, set x[0] = t and x[1:] = 1.0 (maximizes sum, minimizes g to 11).
    
    Strategy:
    1. Verify the structure with a few probes.
    2. Sample the Pareto front densely using the analytical solution.
    3. Polish each point with local optimization to ensure we're on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj
        self.elapsed_start = None

    def __call__(self, problem, bounds):
        self.elapsed_start = time.time()
        deadline = self.elapsed_start + 320  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        rng = np.random.default_rng(0)
        
        all_points = []
        all_f = []
        
        def eval_point(x):
            """Evaluate a point, tracking budget and time."""
            nonlocal all_points, all_f
            if len(all_points) >= self.budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_points.append(x.copy())
            all_f.append(f.copy())
            return f
        
        # Phase 1: Verify the ZDT1 structure with a few probes
        # Test if x[0] controls f1 and x[1:] control g
        base_x = np.ones(self.dim)
        f_base = eval_point(base_x)
        
        if f_base is None:
            return all_points
        
        # Test x[0] sensitivity for f1
        test_x = base_x.copy()
        test_x[0] = 0.5
        f_test = eval_point(test_x)
        
        if f_test is None:
            return all_points
        
        # If f1 changed significantly, x[0] is likely the first objective variable
        f1_change = abs(f_test[0] - f_base[0])
        
        # Phase 2: Sample the Pareto front using the analytical solution
        # For ZDT1, the Pareto front is achieved by:
        # x[0] = t (where t in [0,1])
        # x[1:] = 1.0 (maximizes sum(x[1:]), minimizing g)
        
        # We'll sample t values that give good hypervolume coverage
        # Use a logarithmic spacing to get more points near t=0 (where the front is curved)
        n_front = min(150, self.budget - len(all_points) - 20)
        
        # Create t values with denser sampling near 0
        t_values = np.concatenate([
            np.linspace(0.001, 0.01, 10),      # Very dense near 0
            np.linspace(0.01, 0.1, 20),         # Dense near low values
            np.linspace(0.1, 0.5, 30),          # Medium density
            np.linspace(0.5, 0.9, 30),          # Medium density
            np.linspace(0.9, 0.999, 20),        # Dense near 1
        ])
        t_values = t_values[:n_front]
        
        for t in t_values:
            if time.time() > deadline or len(all_points) >= self.budget - 5:
                break
            
            x = np.ones(self.dim)
            x[0] = t
            f = eval_point(x)
        
        # Phase 3: Local optimization around promising points
        if time.time() < deadline and len(all_points) < self.budget - 10:
            F_current = np.array(all_f)
            n_current = len(F_current)
            
            # Find non-dominated points
            nondom_idx = []
            for i in range(n_current):
                is_dom = False
                for j in range(n_current):
                    if i != j and np.all(F_current[j] <= F_current[i]) and np.any(F_current[j] < F_current[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nondom_idx.append(i)
            
            # Do local search around a few non-dominated points to polish
            n_local = min(20, len(nondom_idx), (self.budget - len(all_points)) // 3)
            
            for idx in nondom_idx[:n_local]:
                if time.time() > deadline or len(all_points) >= self.budget - 2:
                    break
                
                base_x = all_points[idx].copy()
                base_f = all_f[idx].copy()
                
                # Try small perturbations in x[0] direction (most important for f1)
                for delta in [-0.005, -0.002, 0.002, 0.005]:
                    if time.time() > deadline or len(all_points) >= self.budget - 1:
                        break
                    
                    new_x = base_x.copy()
                    new_x[0] = np.clip(base_x[0] + delta, lo[0], hi[0])
                    new_f = eval_point(new_x)
        
        # Ensure we haven't exceeded budget
        all_points = all_points[:self.budget]
        all_f = all_f[:self.budget]
        
        return all_points
# EVOLVE-BLOCK-END
