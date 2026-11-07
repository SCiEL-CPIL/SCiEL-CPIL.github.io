"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start coordinate descent with golden-section 1D refinement,
    interleaved with random restarts to handle the 21-peak landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Golden-section search in 1D
        def golden_section_1d(func_1d, lo, hi, max_iter=20):
            """Minimize func_1d on [lo, hi] using golden-section search."""
            if time.time() > deadline or calls >= budget:
                return (lo + hi) / 2, float('inf')
            
            gr = (np.sqrt(5) - 1) / 2  # golden ratio
            a = lo
            b = hi
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            if calls >= budget or time.time() > deadline:
                return (lo + hi) / 2, float('inf')
            
            fc = func_1d(c)
            fd = func_1d(d)
            
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    if abs(c - d) < 1e-10:
                        break
                    fc = func_1d(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    if abs(c - d) < 1e-10:
                        break
                    fd = func_1d(d)
            
            # Return the better of c and d
            if fc < fd:
                return c, fc
            else:
                return d, fd

        # Coordinate descent with golden-section refinement on each coordinate
        def coordinate_descent(x0, max_rounds=5):
            nonlocal calls, best_f, best_x
            
            if calls >= budget or time.time() > deadline:
                return x0, best_f
            
            x = x0.copy()
            f = best_f
            rng_local = np.random.default_rng(12345)
            
            for round_idx in range(max_rounds):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                
                # Shuffle coordinate order to escape symmetry
                coord_order = rng_local.permutation(dim)
                improved = False
                
                for d in coord_order:
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    
                    # Define 1D function along coordinate d
                    def func_1d(val, d=d, x=x):
                        if calls >= budget or time.time() > deadline:
                            return f
                        x_temp = x.copy()
                        x_temp[d] = val
                        return eval_and_track(x_temp)
                    
                    # Golden-section search on [-5, 5]
                    opt_val, opt_f = golden_section_1d(func_1d, -5.0, 5.0, max_iter=15)
                    
                    if opt_f < f:
                        x[d] = opt_val
                        f = opt_f
                        improved = True
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            return x, f

        # Phase 1: Initial diverse sampling using a quasi-random sequence
        rng = np.random.default_rng(42)
        
        # Use a simple LHS (Latin Hypercube) style sampling for diversity
        n_scan = min(30, budget // 15)
        for i in range(n_scan):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            eval_and_track(x)
        
        # Also try some structured points
        structured_points = [
            np.zeros(dim),
            np.array([5.0] * dim),
            np.array([-5.0] * dim),
        ]
        for sp in structured_points:
            if calls >= budget or time.time() > deadline:
                break
            eval_and_track(sp)

        # Phase 2: Multi-start coordinate descent
        # Start from the best point found, then from a few other promising points
        starts = [best_x.copy()]
        
        # Add a few random starts near the best point
        for _ in range(3):
            if calls >= budget - 50 or time.time() > deadline:
                break
            perturbation = rng.normal(0, 1.0, dim)
            x_start = np.clip(best_x + perturbation, -5, 5)
            eval_and_track(x_start)
            starts.append(x_start.copy())
        
        # Add a few random starts elsewhere
        for _ in range(3):
            if calls >= budget - 50 or time.time() > deadline:
                break
            x_start = rng.uniform(-5, 5, dim)
            eval_and_track(x_start)
            starts.append(x_start.copy())
        
        for x_start in starts:
            if calls >= budget - 20 or time.time() > deadline:
                break
            coordinate_descent(x_start, max_rounds=4)
        
        # Phase 3: Fine-tuning with smaller steps
        if calls < budget - 10 and time.time() < deadline:
            # Do another round of coordinate descent with more iterations but smaller search range
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Try refining each coordinate with a narrower range
            for d in range(dim):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                
                center = x_curr[d]
                half_width = 2.0
                
                def func_1d(val, d=d, x=x_curr):
                    if calls >= budget or time.time() > deadline:
                        return f_curr
                    x_temp = x.copy()
                    x_temp[d] = val
                    return eval_and_track(x_temp)
                
                opt_val, opt_f = golden_section_1d(func_1d, center - half_width, center + half_width, max_iter=20)
                
                if opt_f < f_curr:
                    x_curr[d] = opt_val
                    f_curr = opt_f
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
        
        # Phase 4: If we have budget left, do a few more random restarts with coordinate descent
        while calls < budget - 30 and time.time() < deadline:
            x_start = rng.uniform(-5, 5, dim)
            eval_and_track(x_start)
            if best_f < float('inf'):
                coordinate_descent(x_start, max_rounds=2)
            # Also try refining the current best
            if calls < budget - 20:
                coordinate_descent(best_x.copy(), max_rounds=2)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
