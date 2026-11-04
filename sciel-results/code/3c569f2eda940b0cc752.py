<<<<<<< SEARCH
def _local_optimize(circles, W, H, max_iter=2000, xatol=1e-10, fatol=1e-12,
                    boundary_weight=200.0, overlap_weight=200.0):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    def obj_grad(params):
        return _objective_and_grad(params, W, H, boundary_weight, overlap_weight)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-10, 'maxls': 100}
    )
    return result.x.reshape(N, 3)
=======
def _local_optimize(circles, W, H, max_iter=2500, xatol=1e-10, fatol=1e-12,
                    boundary_weight=200.0, overlap_weight=200.0):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-7, W - 1e-7))  # x
        bounds.append((1e-7, H - 1e-7))  # y
        bounds.append((1e-7, r_max))     # r

    def obj_grad(params):
        return _objective_and_grad(params, W, H, boundary_weight, overlap_weight)

    result = minimize(
        obj_grad, params,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': fatol, 'gtol': 1e-11, 'maxls': 150}
    )
    return result.x.reshape(N, 3)
>>>>>>> REPLACE

