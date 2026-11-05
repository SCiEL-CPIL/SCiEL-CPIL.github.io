"""
Dzugutov cluster global minima finder using continuous annealing of softness parameters.

Approach:
- The Dzugutov potential has exponential softness parameters c and d that control the
  steepness of the repulsive/soft-core barriers. By starting with very large c and d
  (making the potential very soft/smooth) and gradually annealing them down to their
  target values (c=1.10, d=0.27), we create a smooth energy landscape where a single
  gradient-based trajectory can flow into the global polytetrahedral basin.
- We use scipy's L-BFGS-B optimizer with the exact Dzugutov energy and its gradient
  (via chem_clusters.dzugutov_energy_gradient for the final potential, and a custom
  gradient for the annealed potentials).
- The annealing schedule is continuous: at each step we reduce c and d towards target,
  run a few gradient steps, and repeat. This avoids discrete restarts.
"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_energy_custom(positions, A=5.82, B=1.28, a=1.87, b=1.94,
                            c=1.10, d=0.27, m=16.0):
    """Compute Dzugutov energy with custom c, d parameters."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E


def _dzugutov_gradient_custom(positions, A=5.82, B=1.28, a=1.87, b=1.94,
                              c=1.10, d=0.27, m=16.0):
    """Compute Dzugutov energy gradient with custom c, d parameters."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                # V(r) = A * (r^-m - B) * exp(c/(r-a))
                # dV/dr = A * (-m*r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2))
                rnegm = r ** (-m)
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1) * exp_term
                             + (rnegm - B) * exp_term * (-c / (r - a) ** 2))
            elif r < b:
                # V(r) = B * exp(d/(r-b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                exp_term = np.exp(d / (r - b))
                dV_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                continue

            # Force on i is -dV/dr * (rij/r), on j is +dV/dr * (rij/r)
            # grad_i += dV/dr * (positions[j]-positions[i])/r  = -dV_dr * rij / r
            # Actually: dV/d(rij) = dV_dr * (rij/r), and grad_i = -dV/d(positions_i) = -dV_dr*(positions[j]-positions[i])/r
            # Let's be careful: V depends on r = |ri - rj|
            # dV/dri = dV/dr * (ri - rj)/r
            # grad_i = sum_j dV/dri = dV_dr * (ri - rj)/r for each j
            # grad_j = -dV_dr * (ri - rj)/r = dV_dr * (rj - ri)/r
            unit = rij / r
            grad[i] += dV_dr * unit
            grad[j] -= dV_dr * unit
    return grad


def _make_initial_positions(n, rng):
    """Create an initial compact cluster with reasonable spacing."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    # If we couldn't place all atoms, add them with small random offsets
    while len(points) < n:
        base = points[len(points) % max(len(points), 1)] if points else np.zeros(3)
        p = base + rng.uniform(-0.3, 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _anneal_optimize(positions, n, rng, max_time=45.0):
    """
    Run continuous annealing on c and d parameters with gradient-based optimization.
    Start with large c, d (soft landscape) and anneal to target values.
    """
    import time
    t0 = time.time()
    
    # Target parameters
    c_target = 1.10
    d_target = 0.27
    A = 5.82
    B = 1.28
    a = 1.87
    b = 1.94
    m = 16.0
    
    # Annealing schedule: start with large c, d and reduce to target
    # Use a smooth exponential decay schedule
    c_start = 8.0
    d_start = 2.0
    
    # Number of annealing steps
    n_steps = 12
    # For each step, run a few L-BFGS iterations
    
    pos = positions.copy()
    
    for step in range(n_steps):
        # Interpolate c and d from start to target
        frac = (step + 1) / n_steps
        # Use a smooth interpolation (ease-in-out style)
        c_val = c_target + (c_start - c_target) * (1 - frac) ** 2
        d_val = d_target + (d_start - d_target) * (1 - frac) ** 2
        
        # Define objective and gradient for current c, d
        def obj(x):
            p = x.reshape(-1, 3)
            return _dzugutov_energy_custom(p, A, B, a, b, c_val, d_val, m)
        
        def grad(x):
            p = x.reshape(-1, 3)
            g = _dzugutov_gradient_custom(p, A, B, a, b, c_val, d_val, m)
            return g.ravel()
        
        # Run L-BFGS-B for a limited number of iterations
        max_iter = max(5, 30 - step * 2)  # Fewer iterations as we approach target
        result = minimize(obj, pos.ravel(), jac=grad, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'maxfun': max_iter * 2,
                                   'gtol': 1e-3, 'maxls': 20})
        pos = result.x.reshape(-1, 3)
        
        # Check time
        if time.time() - t0 > max_time - 5:
            break
    
    # Final optimization with the exact target parameters using the provided gradient
    def final_obj(x):
        p = x.reshape(-1, 3)
        # Use the provided energy function via relax? No, let's compute directly
        return _dzugutov_energy_custom(p, A, B, a, b, c_target, d_target, m)
    
    def final_grad(x):
        p = x.reshape(-1, 3)
        # Use the provided gradient
        g = dzugutov_energy_gradient(p)
        return g.ravel()
    
    # Final polish with more iterations
    max_iter_final = 50
    result = minimize(final_obj, pos.ravel(), jac=final_grad, method='L-BFGS-B',
                      options={'maxiter': max_iter_final, 'maxfun': max_iter_final * 3,
                               'gtol': 1e-4, 'maxls': 30})
    pos = result.x.reshape(-1, 3)
    
    return pos


def construct(instance):
    """
    Find low-energy Dzugutov clusters using continuous annealing of softness parameters.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Try a few different initial configurations with the annealing approach
    n_trials = 3
    time_per_trial = 40.0  # ~120s total for 3 trials, leaving buffer
    
    for trial in range(n_trials):
        # Create initial positions with some variation
        if trial == 0:
            # Compact random
            positions = _make_initial_positions(size, rng)
        elif trial == 1:
            # Slightly more spread out
            spacing = 1.1
            radius = spacing * size ** (1.0 / 3.0) * 0.9
            points = []
            attempts = 0
            while len(points) < size and attempts < 10000:
                attempts += 1
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                        points.append(p)
            while len(points) < size:
                base = points[len(points) % max(len(points), 1)] if points else np.zeros(3)
                p = base + rng.uniform(-0.4, 0.4, 3)
                points.append(p)
            positions = np.array(points[:size])
        else:
            # Icosahedral-like shell structure
            positions = _make_icosahedral_shell(size, rng)
        
        # Run annealing optimization
        optimized = _anneal_optimize(positions, size, rng, max_time=time_per_trial)
        
        # Compute final energy with the exact potential
        final_energy = _dzugutov_energy_custom(optimized, 5.82, 1.28, 1.87, 1.94,
                                                1.10, 0.27, 16.0)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = optimized
    
    # Ensure minimum separation constraint
    if best_positions is not None:
        # Check and fix any violations of 0.5 minimum distance
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                rij = best_positions[i] - best_positions[j]
                r = np.linalg.norm(rij)
                if r < 0.5 and r > 1e-10:
                    # Push apart
                    direction = rij / r
                    shift = (0.5 - r) / 2.0 + 0.01
                    best_positions[i] += direction * shift
                    best_positions[j] -= direction * shift
    
    return best_positions


def _make_icosahedral_shell(n, rng):
    """Create an icosahedral-like shell structure as initial guess."""
    # icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # normalize to unit sphere
    
    points = []
    # Center atom
    if n > 0:
        points.append(np.zeros(3))
    
    # First shell: icosahedron vertices at radius ~1.1
    r1 = 1.1
    for v in ico_verts:
        if len(points) >= n:
            break
        points.append(v * r1)
    
    # Second shell: more atoms at larger radius
    r2 = 1.9
    # Generate points on sphere for second shell
    while len(points) < n:
        # Random point on sphere
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r2
        # Check distance to existing points
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
        else:
            # Try a different random point
            continue
    
    return np.array(points[:n])
# EVOLVE-BLOCK-END
