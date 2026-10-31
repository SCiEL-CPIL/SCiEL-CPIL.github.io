            if result.fun < best_score:
                best_score = result.fun
                best_solution = result.x
        except Exception as e:
            print(f"DE failed for config {idx}: {e}")
            
    if best_solution is None:
        # Fallback
        return [[0,0,0] for _ in range(N)], (0,0), 10.0, 0.0

    # Polishing step using Nelder-Mead
    try:
        from scipy.optimize import minimize
        res = minimize(
            objective,
            best_solution,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_solution = res.x
    except Exception as e:
        pass
    