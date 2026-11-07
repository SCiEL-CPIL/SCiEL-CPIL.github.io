import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        time_limit = 290.0

        best_x = None
        best_f = float('inf')
        calls = 0

        # 1. Initial Exploration
        np.random.seed(42)
        init_points = [np.zeros(n)]
        for _ in range(10):
            init_points.append(np.random.uniform(-5, 5, n))
        
        for x in init_points:
            if time.time() - start_time > time_limit or calls >= budget:
                break
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls += 1

        # 2. Nelder-Mead Simplex Method
        x = best_x.copy()
        f = best_f
        
        # Create initial simplex
        simplex = [x.copy()]
        for i in range(n):
            new_point = x.copy()
            new_point[i] += 0.25
            simplex.append(new_point)
        
        # Evaluate simplex points
        simplex_f = [f]
        for i in range(1, n + 1):
            if calls >= budget:
                break
            fx = func(simplex[i])
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = simplex[i].copy()
            simplex_f.append(fx)
        
        # Nelder-Mead parameters
        alpha_ = 1.0  # reflection
        gamma_ = 2.0  # expansion
        rho_ = 0.5    # contraction
        sigma_ = 0.5  # shrink
        max_iter = 1000
        
        for _ in range(max_iter):
            if calls >= budget or time.time() - start_time > time_limit:
                break
            
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Check convergence
            if abs(simplex_f[0] - simplex_f[-1]) < 1e-10:
                break
            
            # Compute centroid (excluding worst point)
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha_ * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            if calls < budget:
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma_ * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                if calls < budget:
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho_ * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls < budget:
                        fc = func(xc)
                        calls += 1
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    
                    if fc <= fr:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_ * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls < budget:
                                fi = func(simplex[i])
                                calls += 1
                                if fi < best_f:
                                    best_f = fi
                                    best_x = simplex[i].copy()
                            simplex_f[i] = fi
                else:
                    # Inside contraction
                    xc = centroid - rho_ * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    if calls < budget:
                        fc = func(xc)
                        calls += 1
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_ * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls < budget:
                                fi = func(simplex[i])
                                calls += 1
                                if fi < best_f:
                                    best_f = fi
                                    best_x = simplex[i].copy()
                            simplex_f[i] = fi
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
        
        return best_x, best_f
