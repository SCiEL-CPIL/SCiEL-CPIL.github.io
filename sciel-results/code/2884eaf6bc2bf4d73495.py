import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        d = self.dim
        rng = np.random.default_rng(42)
        
        # Initial point
        x0 = rng.uniform(-5, 5, d)
        
        # Build simplex
        x = [x0]
        fx = [func(x0)]
        
        # Initial step size
        # If we use 1.0, it might be too small.
        # Let's use 1.5
        step_size = 1.5
        
        for i in range(d):
            xi = x0.copy()
            xi[i] += step_size
            x.append(xi)
            fx.append(func(xi))
            
        calls = d + 1
        best_idx = np.argmin(fx)
        best_x = x[best_idx].copy()
        best_f = fx[best_idx]
        
        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Sort
            order = np.argsort(fx)
            x = [x[i] for i in order]
            fx = [fx[i] for i in order]
            
            # Centroid of best d
            centroid = np.mean(x[:-1], axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - x[-1])
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            calls += 1
            
            if fxr < fx[0]:
                # Expand
                xe = centroid + gamma * (centroid - x[-1])
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                calls += 1
                if fxe < fxr:
                    x[-1] = xe
                    fx[-1] = fxe
                else:
                    x[-1] = xr
                    fx[-1] = fxr
            elif fxr < fx[-2]:
                # Accept
                x[-1] = xr
                fx[-1] = fxr
            else:
                # Contract
                if fxr < fx[-1]:
                    # Outside
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    calls += 1
                    if fxc <= fxr:
                        x[-1] = xc
                        fx[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            x[i] = x[0] + sigma * (x[i] - x[0])
                            fx[i] = func(x[i])
                            calls += 1
                            if time.time() > deadline:
                                break
                else:
                    # Inside
                    xc = centroid + rho * (x[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    calls += 1
                    if fxc < fx[-1]:
                        x[-1] = xc
                        fx[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            x[i] = x[0] + sigma * (x[i] - x[0])
                            fx[i] = func(x[i])
                            calls += 1
                            if time.time() > deadline:
                                break
            
            # Update best
            current_best_idx = np.argmin(fx)
            if fx[current_best_idx] < best_f:
                best_f = fx[current_best_idx]
                best_x = x[current_best_idx].copy()
                
        return best_x, best_f
