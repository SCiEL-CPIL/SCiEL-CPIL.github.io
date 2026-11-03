<<<<<<< SEARCH
def _polish(params, theta_outer):
    """High-precision polish."""
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 300, 'ftol': 1e-15})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    return params
=======
def _polish(params, theta_outer):
    """High-precision polish."""
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 500, 'ftol': 1e-15})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 1200, 'ftol': 1e-15, 'gtol': 1e-14})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    return params


def _polish_joint(params):
    """High-precision joint polish (inner + outer angle)."""
    try:
        res = minimize(_objective_joint, params,
                       method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-14})
        if res.success or res.fun < _objective_joint(params):
            return res.x
    except Exception:
        pass
    return params
>>>>>>> REPLACE

