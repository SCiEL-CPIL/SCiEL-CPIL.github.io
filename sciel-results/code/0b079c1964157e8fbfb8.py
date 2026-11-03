<<<<<<< SEARCH
    def _contact_polish(circles, w, h):
        if time_left() < 10.0:
            return circles
        x0 = circles.flatten()
        r_max = min(w, h) / 2.0
        bounds = [(0, w), (0, h), (1e-4, r_max)] * N
        best_x = x0.copy(); best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))
        for beta, eq_w, iters in [(300.0, 0.005, 2000), (800.0, 0.002, 3000),
                                  (2000.0, 0.001, 5000), (5000.0, 0.0005, 5000)]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(_objective_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, eq_w),
                               options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 80, 'maxfun': 50000})
                if time_left() < 3.0:
                    break
                nc = res.x.reshape(-1, 3)
                if _strictly_valid(nc, w, h):
                    val = float(np.sum(nc[:, 2]))
                    if val > best_val:
                        best_val = val; best_x = res.x.copy()
                x0 = res.x
            except Exception:
                pass
        return best_x.reshape(-1, 3)
=======
    def _contact_polish(circles, w, h):
        if time_left() < 10.0:
            return circles
        x0 = circles.flatten()
        r_max = min(w, h) / 2.0
        bounds = [(0, w), (0, h), (1e-4, r_max)] * N
        best_x = x0.copy(); best_val = -np.inf
        if _strictly_valid(circles, w, h):
            best_val = float(np.sum(circles[:, 2]))
        for beta, eq_w, iters in [(300.0, 0.005, 2000), (800.0, 0.002, 3000),
                                  (2000.0, 0.001, 5000), (5000.0, 0.0005, 5000)]:
            if time_left() < 5.0:
                break
            try:
                res = minimize(_objective_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, beta, eq_w),
                               options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxls': 80, 'maxfun': 50000})
                if time_left() < 3.0:
                    break
                nc = res.x.reshape(-1, 3)
                if _strictly_valid(nc, w, h):
                    val = float(np.sum(nc[:, 2]))
                    if val > best_val:
                        best_val = val; best_x = res.x.copy()
                x0 = res.x
            except Exception:
                pass
        # Final high-precision polish pass
        if time_left() > 8.0:
            try:
                res = minimize(_objective_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                               jac=True, args=(w, h, 10000.0, 0.0001),
                               options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-14,
                                        'maxls': 100, 'maxfun': 100000})
                if time_left() < 2.0:
                    pass
                else:
                    nc = res.x.reshape(-1, 3)
                    if _strictly_valid(nc, w, h):
                        val = float(np.sum(nc[:, 2]))
                        if val > best_val:
                            best_val = val; best_x = res.x.copy()
            except Exception:
                pass
        return best_x.reshape(-1, 3)
>>>>>>> REPLACE

