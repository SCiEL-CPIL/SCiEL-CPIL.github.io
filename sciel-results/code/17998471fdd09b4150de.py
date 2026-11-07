import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280
        
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        
        # Phase 1: Nelder-Mead
        # Initialize simplex
        simplex = [x.copy()]
        for i in range(dim):
            v = x.copy()
            v[i] += 0.5
            simplex.append(v)
        
        f_simplex = [fx] + [func(np.clip(v, -5, 5)) for v in simplex[1:]]
        calls += dim
        
        # Sort simplex
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]
        
        nm_limit = int(budget * 0.6)
        
        while calls < nm_limit and time.time() < deadline:
            if abs(f_simplex[-1] - f_simplex[0]) < 1e-10:
                break
                
            x_c = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_c + 1.0 * (x_c - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_simplex[0] <= f_r < f_simplex[-2]:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            elif f_r < f_simplex[0]:
                # Expansion
                x_e = x_c + 2.0 * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_simplex[-1]:
                    x_c2 = x_c + 0.5 * (x_r - x_c)
                    x_c2 = np.clip(x_c2, -5, 5)
                    f_c2 = func(x_c2)
                    calls += 1
                    if f_c2 <= f_r:
                        simplex[-1] = x_c2
                        f_simplex[-1] = f_c2
                    else:
                        break
                else:
                    x_c2 = x_c + 0.5 * (simplex[-1] - x_c)
                    x_c2 = np.clip(x_c2, -5, 5)
                    f_c2 = func(x_c2)
                    calls += 1
                    if f_c2 < f_simplex[-1]:
                        simplex[-1] = x_c2
                        f_simplex[-1] = f_c2
                    else:
                        break
                        
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
        x_best = simplex[0]
        fx_best = f_simplex[0]
        
        # Phase 2: Newton
        while calls < budget and time.time() < deadline:
            h = 1e-4
            grad = np.zeros(dim)
            hess = np.zeros((dim, dim))
            f0 = fx_best
            
            f_plus = np.zeros(dim)
            f_minus = np.zeros(dim)
            for i in range(dim):
                xp = x_best.copy(); xp[i] += h
                xm = x_best.copy(); xm[i] -= h
                xp = np.clip(xp, -5, 5)
                xm = np.clip(xm, -5, 5)
                f_plus[i] = func(xp)
                f_minus[i] = func(xm)
                calls += 2
                if calls >= budget: break
                
            if calls >= budget: break
            
            for i in range(dim):
                grad[i] = (f_plus[i] - f_minus[i]) / (2*h)
                hess[i, i] = (f_plus[i] - 2*f0 + f_minus[i]) / (h*h)
                
            for i in range(dim):
                for j in range(i+1, dim):
                    if calls + 4 > budget: break
                    x1 = x_best.copy(); x1[i] += h; x1[j] += h
                    x2 = x_best.copy(); x2[i] += h; x2[j] -= h
                    x3 = x_best.copy(); x3[i] -= h; x3[j] += h
                    x4 = x_best.copy(); x4[i] -= h; x4[j] -= h
                    x1 = np.clip(x1, -5, 5); x2 = np.clip(x2, -5, 5)
                    x3 = np.clip(x3, -5, 5); x4 = np.clip(x4, -5, 5)
                    
                    f1 = func(x1); f2 = func(x2)
                    f3 = func(x3); f4 = func(x4)
                    calls += 4
                    
                    val = (f1 - f2 - f3 + f4) / (4*h*h)
                    hess[i, j] = val
                    hess[j, i] = val
                    
            if calls >= budget: break
            
            try:
                eigvals = np.linalg.eigvalsh(hess)
                if np.min(eigvals) < 1e-8:
                    hess_reg = hess + 1e-8 * np.eye(dim)
                    dx = np.linalg.solve(hess_reg, -grad)
                else:
                    dx = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                break
                
            alpha = 1.0
            improved = False
            for _ in range(10):
                if calls >= budget: break
                x_new = np.clip(x_best + alpha * dx, -5, 5)
                f_new = func(x_new)
                calls += 1
                if f_new < fx_best:
                    x_best = x_new
                    fx_best = f_new
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
                
        return x_best, fx_best
