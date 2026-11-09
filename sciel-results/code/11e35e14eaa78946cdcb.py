import numpy as np
import time

class Optimizer:
    """Nelder-Mead optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 290s
        deadline = start_time + 290
        
        # Initialize at the center of the box
        # The minimizer is in [-4, 4]^5, so 0 is a good deterministic start
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Initial simplex
        # x0 is the best point. We need D other points.
        # Standard initialization: x0 + delta * e_i
        delta = 2.0
        simplex = [x0.copy()]
        for i in range(self.dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            
        # Evaluate initial simplex
        values = [f0]
        for i in range(1, self.dim + 1):
            if calls >= self.budget or time.time() > deadline:
                break
            fi = func(simplex[i])
            calls += 1
            values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = simplex[i].copy()
        
        # If we didn't evaluate all initial points, we can't do NM properly.
        # But with budget 500 and dim 5, we always have enough for init.
        
        # NM Loop
        while calls < self.budget and time.time() < deadline:
            # Sort simplex by values
            idx = np.argsort(values)
            simplex = [simplex[i] for i in idx]
            values = [values[i] for i in idx]
            
            # Convergence check
            # If the range of values is very small, stop
            if abs(values[-1] - values[0]) < 1e-8:
                break
            
            # Centroid of all but the worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            
            if fr < values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                
                if fe < fr:
                    simplex[-1] = xe
                    values[-1] = fe
                else:
                    simplex[-1] = xr
                    values[-1] = fr
            elif fr < values[-2]:
                # Accept reflection
                simplex[-1] = xr
                values[-1] = fr
            else:
                # Contraction
                # Try outside contraction first (standard NM usually tries inside if reflection is worse than worst, but here reflection is worse than second worst)
                # Standard: if fr >= values[-2], try contraction.
                # Inside contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                
                if fc < fr:
                    simplex[-1] = xc
                    values[-1] = fc
                else:
                    # Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, self.dim + 1):
                        if calls >= self.budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fi = func(simplex[i])
                        values[i] = fi
                        calls += 1
                        if fi < best_f:
                            best_f = fi
                            best_x = simplex[i].copy()
                    
                    # Re-sort after shrink
                    idx = np.argsort(values)
                    simplex = [simplex[i] for i in idx]
                    values = [values[i] for i in idx]
                    
                    # If shrink didn't improve the worst point significantly, we might be stuck.
                    # But we continue until budget/time.
        
        return best_x, best_f
