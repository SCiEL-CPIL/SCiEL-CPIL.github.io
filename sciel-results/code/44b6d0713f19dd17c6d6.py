            try:
                result = minimize(func, x0, method='Powell', 
                                 options={'maxiter': remaining, 'xtol': 1e-8, 'ftol': 1e-10})
                if result.fun < best_f:
                    best_x, best_f = result.x, result.fun
            except:
                pass
