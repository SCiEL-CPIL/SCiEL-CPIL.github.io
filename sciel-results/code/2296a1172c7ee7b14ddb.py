<<<<<<< SEARCH
def _refine(pts, deadline):
    """Local refinement maximizing a repulsive-potential objective (proxy for min dist)."""
    n = len(pts)
    pts = pts.astype(float).copy()

    def objective(x):
        p = x.reshape(n, 2)
        diff = p[:, None, :] - p[None, :, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        norms = np.hypot(p[:, 0], p[:, 1])
        r = 1.0 - norms.max()
        return float(np.mean(1.0 / (d + 1e-6)) + 1e6 * r * r)

    x = pts.reshape(-1)
    obj = objective(x)
    # Vectorized steepest descent with backtracking line search.
    it = 0
    while time.time() < deadline and it < 4000:
        it += 1
        p = x.reshape(n, 2)
        diff = p[:, None, :] - p[None, :, :]
        d2 = (diff ** 2).sum(-1)
        np.fill_diagonal(d2, 1e-12)
        d = np.sqrt(d2)
        inv = 1.0 / (d + 1e-6)
        # grad of mean(1/d): -mean_j (1/d_ij^2) * (p_i - p_j)/d_ij
        g = -np.einsum('ij,ijk->ik', inv * inv, diff) / n
        norms = np.hypot(p[:, 0], p[:, 1])
        i_max = int(np.argmax(norms))
        if norms[i_max] > 1e-9:
            g[i_max] += 2.0e6 * (1.0 - norms[i_max]) * p[i_max] / norms[i_max]
        g = g.reshape(-1)
        gn = np.linalg.norm(g)
        if gn < 1e-13:
            break
        g = g / gn
        # backtracking line search
        step = 0.05
        for _ in range(15):
            x2 = x - step * g
            if objective(x2) < obj:
                x = x2
                obj = objective(x)
                break
            step *= 0.5
        else:
            break
    return x.reshape(n, 2)
=======
def _soft_min_obj(x, n, k=5):
    """Soft-min objective using KDTree for speed. Lower is better."""
    p = x.reshape(n, 2)
    tree = cKDTree(p)
    dists, _ = tree.query(k=k+1)
    # dists[i,0] is 0 (self), dists[i,1:] are k nearest neighbors
    nn_dists = dists[:, 1:]
    # Soft minimum: -log(sum(exp(-d/t))) * t approximates min
    t = 0.01
    log_sum = np.log(np.sum(np.exp(-nn_dists / t), axis=1))
    soft_min = -t * np.mean(log_sum)
    
    # Boundary penalty: push points inward
    norms = np.hypot(p[:, 0], p[:, 1])
    boundary_violation = np.maximum(0, norms - 0.99)
    boundary_penalty = np.sum(boundary_violation ** 2) * 1e4
    
    return soft_min + boundary_penalty


def _soft_min_grad(x, n, k=5):
    """Gradient of the soft-min objective."""
    p = x.reshape(n, 2)
    tree = cKDTree(p)
    dists, idx = tree.query(k=k+1)
    nn_dists = dists[:, 1:]
    nn_idx = idx[:, 1:]
    
    t = 0.01
    # For each point i, grad of soft_min_i w.r.t. p_i
    # soft_min_i = -t * log(sum_j exp(-d_ij/t))
    # d/dp_i soft_min_i = t * sum_j (exp(-d_ij/t) / sum_k exp(-d_ik/t)) * (p_i - p_j) / d_ij * (-1/t)
    #                = -sum_j (w_ij * (p_i - p_j) / d_ij) where w_ij = exp(-d_ij/t) / sum_k exp(-d_ik/t)
    
    weights = np.exp(-nn_dists / t)
    weights /= weights.sum(axis=1, keepdims=True)
    
    grad = np.zeros_like(p)
    for i in range(n):
        for j in range(k):
            ni = nn_idx[i, j]
            di = nn_dists[i, j]
            if di > 1e-10:
                grad[i] += weights[i, j] * (p[i] - p[ni]) / di
                grad[ni] -= weights[i, j] * (p[i] - p[ni]) / di
    
    grad /= n  # because we're averaging
    
    # Boundary gradient
    norms = np.hypot(p[:, 0], p[:, 1])
    mask = norms > 0.99
    if np.any(mask):
        grad[mask] += 2e4 * (norms[mask] - 0.99) * p[mask] / norms[mask, None]
    
    return grad.reshape(-1)


def _refine(pts, deadline):
    """Local refinement using L-BFGS-B with soft-min objective."""
    n = len(pts)
    x0 = pts.astype(float).copy().reshape(-1)
    
    def fun(x):
        return _soft_min_obj(x, n)
    
    def jac(x):
        return _soft_min_grad(x, n)
    
    # Use L-BFGS-B which is efficient for large n
    result = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxfun': 2000, 'disp': False})
    
    if time.time() > deadline:
        return pts  # Return original if timeout
    
    x_opt = result.x.reshape(n, 2)
    
    # Verify improvement and clamp to unit disk
    norms = np.hypot(x_opt[:, 0], x_opt[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        x_opt[mask] /= norms[mask, None]
    
    return x_opt
>>>>>>> REPLACE

