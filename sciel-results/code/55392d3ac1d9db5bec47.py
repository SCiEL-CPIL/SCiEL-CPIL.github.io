<<<<<<< SEARCH
    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
=======
    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Final polish: use scipy L-BFGS-B with exact objective
    if time.time() < deadline - 10:
        try:
            # Define wrapper for scipy that handles constraints via penalty
            def scipy_objective(x):
                if time.time() > deadline - 5:
                    return 1e10
                a_t = x[:12]
                b_t = x[12:24]
                # Penalty for sum constraint
                penalty = 100.0 * (np.sum(a_t) - 1.0)**2 + 100.0 * (np.sum(b_t) - 1.0)**2
                return objective_exact(x) + penalty

            res = minimize(scipy_objective, best_x, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_obj:
                # Evaluate with exact objective
                exact_val = objective_exact(res.x)
                if exact_val < best_obj:
                    best_obj = exact_val
                    best_x = res.x
        except Exception:
            pass

    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
>>>>>>> REPLACE
