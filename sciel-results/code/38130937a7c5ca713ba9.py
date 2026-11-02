            prev_fun = objective(current_x, k, lam)
            res = minimize(...)
            if res.fun < prev_fun:
                current_x = res.x
