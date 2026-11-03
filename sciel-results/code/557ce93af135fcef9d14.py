def energy_spherical(params):
    theta = params[0::2]
    phi = params[1::2]
    pts = spherical_to_cartesian(theta, phi)
    return compute_energy_cartesian(pts)

def grad_spherical(params):
    theta = params[0::2]
    phi = params[1::2]
    pts = spherical_to_cartesian(theta, phi)
    n = len(theta)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    if np.min(dist[np.triu_indices(n, k=1)]) < 1e-6:
        return np.zeros_like(params)
        
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    G = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    t = np.column_stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    
    p = np.column_stack([
        -np.sin(phi) * np.sin(theta),
        np.cos(phi) * np.sin(theta),
        np.zeros_like(theta)
    ])
    
    grad_theta = -np.sum(G * t, axis=1)
    grad_phi = -np.sum(G * p, axis=1)
    
    grad = np.zeros_like(params)
    grad[0::2] = grad_theta
    grad[1::2] = grad_phi
    return grad
