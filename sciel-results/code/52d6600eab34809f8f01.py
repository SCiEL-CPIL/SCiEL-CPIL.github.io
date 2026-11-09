<<<<<<< SEARCH
            try:
                # Nelder-Mead is robust. 
                # maxiter 100 with ~5 evals/iter is ~500 evals.
                # We have ~350 calls left, so 70 maxiter is safe.
                res = minimize(safe_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': 70, 'xatol': 1e-4, 'fatol': 1e-4},
                               initial_simplex=simplex)
            except ValueError:
                # Budget or time exceeded, keep best found
                pass
            except Exception:
                pass
=======
            try:
                # Nelder-Mead is robust. 
                # Keep maxiter low to ensure we don't hang.
                res = minimize(safe_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3},
                               initial_simplex=simplex)
            except ValueError:
                pass
            except Exception:
                pass
>>>>>>> REPLACE
