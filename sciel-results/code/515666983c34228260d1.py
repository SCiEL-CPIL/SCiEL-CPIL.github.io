import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Global Phase: Deterministic Space-Filling Sampling
        num_global = 10
        points = self._halton_points(num_global, self.dim, lower=-5.0, upper=5.0)
        
        best_x = None
        best_f = float('inf')
        calls_used = 0
        
        for p in points:
            f_val = func(p)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = p.copy()
        
        if calls_used >= self.budget:
            return best_x, best_f

        # 2. Local Phase: Nelder-Mead
        # Initialize simplex around best_x
        delta = 0.5
        x0 = best_x
        simplex = [x0.copy()]
        simplex_f = [best_f]
        
        for i in range(self.dim):
            x_i = x0.copy()
            x_i[i] += delta
            simplex.append(x_i)
            if calls_used < self.budget:
                f_i = func(x_i)
                calls_used += 1
                simplex_f.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            else:
                # If we run out of budget here, just return what we have
                break

        if calls_used >= self.budget:
            return best_x, best_f

        # Nelder-Mead Loop
        while calls_used < self.budget:
            # Sort simplex by f value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            
            if calls_used >= self.budget:
                break
                
            f_ref = func(x_ref)
            calls_used += 1
            
            if f_ref < simplex_f[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                
                if calls_used < self.budget:
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Contraction
                rho = 0.5
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                
                if calls_used < self.budget:
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            if calls_used >= self.budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = func(simplex[i])
                            calls_used += 1
                else:
                    # No budget left for contraction, keep worst? Or just break.
                    pass

        return best_x, best_f

    def _halton_points(self, n, dim, lower, upper):
        points = []
        for i in range(1, n + 1):
            point = []
            for d in range(2, dim + 2): # Bases 2, 3, 4, 5, 6
                f = 1.0
                i_val = i
                h = 0.0
                while i_val > 0:
                    i_val, r = divmod(i_val, d)
                    h += r * f
                    f /= d
                point.append(h)
            # Map to [lower, upper]
            point_final = lower + np.array(point) * (upper - lower)
            points.append(point_final)
        return np.array(points)
