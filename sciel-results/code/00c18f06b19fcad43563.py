            res = minimize(_objective, y0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
