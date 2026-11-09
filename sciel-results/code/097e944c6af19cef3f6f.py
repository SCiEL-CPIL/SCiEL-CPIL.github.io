# After CMA-ES loop
if self.calls < self.budget:
    try:
        from scipy.optimize import minimize
        # Polish with Nelder-Mead
        res = minimize(eval_func, best_x, method='Nelder-Mead', options={'xatol': 1e-10, 'ftol': 1e-10, 'maxiter': 100})
        if res.fun < best_fx:
            best_fx = res.fun
            best_x = res.x
    except StopIteration:
        pass
    except Exception:
        pass
