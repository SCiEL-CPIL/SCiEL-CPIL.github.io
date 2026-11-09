        def minimizer(x, **kwargs):
            start_count = call_count[0]
            try:
                res = minimize(objective, x, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 20})
                end_count = call_count[0]
                return {'fun': res.fun, 'x': res.x, 'nfev': end_count - start_count}
            except Exception:
                return {'fun': np.inf, 'x': x, 'nfev': 0}
