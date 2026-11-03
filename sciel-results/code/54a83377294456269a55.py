"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time
import scipy.optimize


def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3)."""
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    d = np.sqrt(d2)
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    grad = M @ points
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - grad
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _project_gradient(points, grad):
    """Project gradient onto tangent space of unit sphere."""
    dots = (points * grad).sum(axis=1, keepdims=True)
    return grad - dots * points


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(N, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _gradient_descent(points, lr, n_iters, momentum=0.9, lr_decay=1.0, min_dist=1e-9):
    """Gradient descent on the sphere with momentum and adaptive learning rate."""
    points = points.copy()
    points = _normalize(points)
    v = np.zeros_like(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iters):
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        v = momentum * v - lr * lr_decay * grad / (grad_norm + 1e-15)
        points = points + v
        points = _normalize(points)
        
        d = _pairwise_distances(points)
        if d.min() < min_dist:
            iu = np.triu_indices(N, k=1)
            close = d[iu] < min_dist * 10
            for idx in range(len(close)):
                if close[idx]:
                    i, j = iu[0][idx], iu[1][idx]
                    mid = (points[i] + points[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-15:
                        mid = mid / mid_norm
                    perp = points[i] - points[j]
                    perp = perp - np.dot(perp, mid) * mid
                    perp_norm = np.linalg.norm(perp)
                    if perp_norm < 1e-15:
                        perp = np.array([1.0, 0.0, 0.0])
                        perp = perp - np.dot(perp, mid) * mid
                        perp_norm = np.linalg.norm(perp)
                    perp = perp / perp_norm
                    points[i] = _normalize(points[i] + 0.01 * perp)
                    points[j] = _normalize(points[j] - 0.01 * perp)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        lr_decay = max(0.999, 1.0 - it / (2.0 * n_iters))
    
    return best_points, best_energy


def _anneal(points, n_restarts=5, lr_start=0.01, lr_end=0.0001, n_iters=2000, seed=0):
    """Multi-restart gradient descent with simulated annealing-like schedule."""
    rng = np.random.default_rng(seed)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for restart in range(n_restarts):
        if restart == 0:
            pts = points.copy()
        else:
            noise = rng.standard_normal((N, 3)) * (0.05 * (1.0 + restart * 0.5))
            pts = _normalize(points + noise)
        
        lr = lr_start
        for phase in range(3):
            n_phase = n_iters // 3
            lr_phase = lr_start * (lr_end / lr_start) ** (phase / 3.0)
            pts, energy = _gradient_descent(pts, lr_phase, n_phase, momentum=0.9)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        
        pts, energy = _gradient_descent(pts, lr_end, n_iters // 5, momentum=0.99)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    return best_points, best_energy


def _spectral_optimize():
    """Optimize using spherical harmonic coefficients as a lower-dimensional basis."""
    l_max = 12
    # Number of coefficients: sum_{l=0}^{l_max} (2l+1) = (l_max+1)^2
    K = (l_max + 1) ** 2
    
    # Generate initial points using Fibonacci spiral
    points_init = _fibonacci_spiral(N)
    points_init = _normalize(points_init)
    
    # Precompute spherical harmonic gradients at each point
    # Y_l^m(theta, phi) for l=0..l_max, m=-l..l
    # We need the gradient of each Y_l^m w.r.t. Cartesian coordinates
    # For a unit sphere point (x,y,z), theta = arccos(z), phi = arctan2(y,x)
    
    # Compute theta and phi for each point
    z = points_init[:, 2]
    x = points_init[:, 0]
    y = points_init[:, 1]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    # Build list of (l, m) pairs
    lm_pairs = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))
    
    # Compute gradients of spherical harmonics w.r.t. Cartesian coordinates
    # Using the relation: grad(Y_l^m) = sum of partial derivatives
    # For efficiency, we compute Y_l^m and its gradient numerically or analytically
    # Here we use a finite difference approach for simplicity
    
    eps = 1e-5
    Y_grads = np.zeros((N, len(lm_pairs), 3), dtype=np.complex128)
    
    for idx, (l, m) in enumerate(lm_pairs):
        # Compute Y_l^m at each point using scipy.special
        # scipy.special.sph_harm(m, l, phi, theta)
        Y_vals = scipy.special.sph_harm(m, l, phi, theta)
        
        # Gradient via finite differences in Cartesian space
        # We need dY/dx, dY/dy, dY/dz
        # Use chain rule: dY/dx = dY/dtheta * dtheta/dx + dY/dphi * dphi/dx
        # But it's easier to just perturb in Cartesian and renormalize
        
        for dim in range(3):
            points_pert = points_init.copy()
            points_pert[:, dim] += eps
            points_pert = _normalize(points_pert)
            
            z_p = points_pert[:, 2]
            x_p = points_pert[:, 0]
            y_p = points_pert[:, 1]
            theta_p = np.arccos(np.clip(z_p, -1.0, 1.0))
            phi_p = np.arctan2(y_p, x_p)
            
            Y_pert = scipy.special.sph_harm(m, l, phi_p, theta_p)
            Y_grads[:, idx, dim] = (Y_pert - Y_vals) / eps
    
    # Define the energy function in terms of coefficients
    def energy_and_grad(c_flat):
        c_real = c_flat[:K]
        c_imag = c_flat[K:]
        c = c_real + 1j * c_imag
        
        # Displacement D_i = sum c_k * gradY_k
        D = np.einsum('ijk,k->ij', Y_grads, c)
        D_real = D.real
        
        x_raw = points_init + 0.1 * D_real
        norms = np.linalg.norm(x_raw, axis=1, keepdims=True)
        x = x_raw / norms
        
        # Compute Energy and Grad w.r.t x
        E = _log_energy(x)
        Gx = _log_energy_grad(x)
        
        # Project to tangent space
        proj = Gx - (Gx * x).sum(axis=1, keepdims=True) * x
        
        # Backprop to x_raw
        Gx_raw = proj / norms
        
        # Backprop to c
        Gc_real = np.einsum('ijk,ij->k', Y_grads.real, Gx_raw) * 0.1
        Gc_imag = -np.einsum('ijk,ij->k', Y_grads.imag, Gx_raw) * 0.1
        
        grad_flat = np.concatenate([Gc_real, Gc_imag])
        return E, grad_flat
    
    rng = np.random.default_rng(0)
    c0 = rng.standard_normal(2 * K) * 0.05
    
    res = scipy.optimize.minimize(
        energy_and_grad, c0, method='L-BFGS-B', jac=True,
        options={'maxiter': 300, 'ftol': 1e-12}
    )
    
    c_opt = res.x
    c = c_opt[:K] + 1j * c_opt[K:]
    D = np.einsum('ijk,k->ij', Y_grads, c)
    D_real = D.real
    x_raw = points_init + 0.1 * D_real
    x = x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)
    
    return x


def construct():
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    points = _normalize(points)
    
    # Multi-scale optimization: coarse to fine
    points, e1 = _gradient_descent(points, lr=0.005, n_iters=500, momentum=0.9)
    points, e2 = _gradient_descent(points, lr=0.001, n_iters=1000, momentum=0.9)
    points, e3 = _gradient_descent(points, lr=0.0001, n_iters=2000, momentum=0.95)
    
    # Multi-restart to escape local minima
    points, e4 = _anneal(points, n_restarts=3, lr_start=0.002, lr_end=0.00005, n_iters=1500, seed=42)
    
    # Final polish
    points, e5 = _gradient_descent(points, lr=0.00005, n_iters=1000, momentum=0.99)
    
    # Try spectral optimization from current best
    points_spec = _spectral_optimize()
    e_spec = _log_energy(points_spec)
    e_cur = _log_energy(points)
    
    if e_spec < e_cur:
        points = points_spec
        # Polish the spectral result
        points, e6 = _gradient_descent(points, lr=0.0001, n_iters=2000, momentum=0.95)
        points, e7 = _gradient_descent(points, lr=0.00005, n_iters=1000, momentum=0.99)
    
    # Ensure valid output
    points = _normalize(points)
    
    # Verify no points are too close
    d = _pairwise_distances(points)
    if d.min() < 1e-9:
        points = _fibonacci_spiral(N)
        points = _normalize(points)
    
    return points
# EVOLVE-BLOCK-END
