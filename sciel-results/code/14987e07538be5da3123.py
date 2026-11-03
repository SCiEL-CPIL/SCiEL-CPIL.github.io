            res = minimize(
                fun,
                best_c.flatten(),
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15},
                callback=callback
            )
