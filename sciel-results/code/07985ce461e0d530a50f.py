import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient of E = sum log(1/d) = - sum log(d)
    # d/dx_i log(1/|x_i - x_j|) = - 1/|x_i - x_j|^2 * (x_i - x_j)
    # Sum over j
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space: g - (g . p) p
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _sphere_to_flat(pts):
    """Convert points on sphere to a flat parameterization using local tangent spaces."""
    n = len(pts)
    # Use the first two coordinates as parameters, solve for z
    # x^2 + y^2 + z^2 = 1 => z = sqrt(1 - x^2 - y^2) (take positive branch, handle sign)
    # Better: use stereographic projection from south pole
    # For point (x, y, z) on unit sphere, stereographic projection from (0,0,-1):
    # u = x / (1 - z), v = y / (1 - z)
    # Inverse: x = 2u / (1 + u^2 + v^2), y = 2v / (1 + u^2 + v^2), z = (1 - u^2 - v^2) / (1 + u^2 + v^2)
    # This covers the whole sphere except the south pole
    z = pts[:, 2]
    # Avoid division by zero when z = 1 (north pole)
    denom = 1.0 - z
    denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    u = pts[:, 0] / denom
    v = pts[:, 1] / denom
    return np.column_stack([u, v]).flatten()

def _flat_to_sphere(flat):
    """Convert flat parameters back to sphere using inverse stereographic projection."""
    n = len(flat) // 2
    flat = flat.reshape(n, 2)
    u = flat[:, 0]
    v = flat[:, 1]
    r2 = u * u + v * v
    denom = 1.0 + r2
    x = 2.0 * u / denom
    y = 2.0 * v / denom
    z = (1.0 - r2) / denom
    pts = np.column_stack([x, y, z])
    return pts

def _energy_flat(flat):
    """Energy in flat coordinates."""
    pts = _flat_to_sphere(flat)
    e, _ = _energy_and_grad(pts)
    return e

def _grad_flat(flat):
    """Gradient in flat coordinates (Jacobian transpose of sphere gradient)."""
    pts = _flat_to_sphere(flat)
    n = len(pts)
    e, grad_sphere = _energy_and_grad(pts)
    
    # Compute Jacobian of stereographic projection inverse
    # x = 2u / (1+u^2+v^2), y = 2v / (1+u^2+v^2), z = (1-u^2-v^2) / (1+u^2+v^2)
    # dx/du = 2(1+u^2+v^2) - 2u*2u all over (1+u^2+v^2)^2 = 2(1+v^2-u^2) / (1+u^2+v^2)^2
    # dx/dv = -2u*2v / (1+u^2+v^2)^2 = -4uv / (1+u^2+v^2)^2
    # dy/du = -4uv / (1+u^2+v^2)^2
    # dy/dv = 2(1+u^2-v^2) / (1+u^2+v^2)^2
    # dz/du = -2u(1+u^2+v^2) - (1-u^2-v^2)*2u all over (1+u^2+v^2)^2 = -4u / (1+u^2+v^2)^2
    # dz/dv = -4v / (1+u^2+v^2)^2
    
    u = flat[0::2]
    v = flat[1::2]
    r2 = u * u + v * v
    denom2 = (1.0 + r2) ** 2
    
    # Jacobian rows for each point: [dx/du, dx/dv, dy/du, dy/dv, dz/du, dz/dv]
    # grad_flat[i*2]   = sum over k of grad_sphere[i,k] * d(point_i,k)/du
    # grad_flat[i*2+1] = sum over k of grad_sphere[i,k] * d(point_i,k)/dv
    
    dx_du = 2.0 * (1.0 + v * v - u * u) / denom2
    dx_dv = -4.0 * u * v / denom2
    dy_du = -4.0 * u * v / denom2
    dy_dv = 2.0 * (1.0 + u * u - v * v) / denom2
    dz_du = -4.0 * u / denom2
    dz_dv = -4.0 * v / denom2
    
    grad_flat = np.zeros_like(flat)
    grad_flat[0::2] = grad_sphere[:, 0] * dx_du + grad_sphere[:, 1] * dy_du + grad_sphere[:, 2] * dz_du
    grad_flat[1::2] = grad_sphere[:, 0] * dx_dv + grad_sphere[:, 1] * dy_dv + grad_sphere[:, 2] * dz_dv
    
    return grad_flat

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200):
    """L-BFGS-B optimization on the sphere using stereographic projection."""
    start = time.time()
    flat0 = _sphere_to_flat(pts)
    
    def fun_and_grad(flat):
        if time.time() - start > time_limit - 0.1:
            return float('inf'), np.zeros_like(flat)
        e = _energy_flat(flat)
        g = _grad_flat(flat)
        return e, g
    
    try:
        result = minimize(
            fun_and_grad,
            flat0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        pts_opt = _flat_to_sphere(result.x)
        pts_opt = _normalize(pts_opt)
        e_opt = _log_energy_fast(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy_fast(pts)

def _riemannian_cg(pts, time_limit=10.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    prev_grad = grad.copy()
    
    # Initial search direction
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Wolfe line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        # Strong Wolfe conditions approximation
        c1 = 1e-4
        c2 = 0.9
        
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Sufficient decrease (Armijo)
            if new_energy > energy + c1 * alpha * np.sum(grad * d):
                break
            
            # Curvature condition
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy > energy:
                break
                
            alpha *= 0.7
            
            if alpha < 1e-12:
                break
                
        if best_alpha == 0.0:
            break
            
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta
        d_new = -grad
        beta = 0.0
        if it > 0:
            num = np.sum(grad * (grad - prev_grad))
            den = np.sum(prev_grad * prev_grad)
            if den > 1e-30:
                beta = max(0.0, num / den)
        else:
            beta = 0.0
            
        d = d_new + beta * d
        prev_grad = grad.copy()
        
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left_init = time_limit - (time.time() - start) - 3
    if time_left_init > 3:
         current, energy = _lbfgs_sphere(current, time_limit=min(12, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.5
    decay = 0.92
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2:
            break
            
        # Perturb with varying scale based on temperature
        rng = np.random.default_rng()
        perturbations = rng.standard_normal((N, 3))
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        # Scale by temperature
        perturb_scale = 0.15 * math.sqrt(temp)
        new_pts = _exponential_map(current, perturbations * perturb_scale)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        time_left = time_limit - (time.time() - start) - 1
        if time_left < 4:
             break
             
        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=min(8, time_left))
        new_energy = _log_energy_fast(new_opt_pts)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or (temp > 0.01 and np.random.rand() < math.exp(-delta_e / temp)):
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
            
        temp *= decay
        
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_init(n):
    """Generate points based on icosahedron symmetry."""
    # Icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    vertices = _normalize(vertices)
    
    # Create a grid on the sphere using spherical coordinates
    # Divide sphere into patches and place points
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    # Use golden angle for azimuthal distribution
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden_angle * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    pts = _normalize(pts)
    
    return pts

def _evolve_population(pop, time_limit, relax_time=2.0):
    """
    Evolutionary Strategy for global optimization.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    pop_size = len(pop)
    
    # Evaluate initial population
    energies = []
    for i in range(pop_size):
        pts, e = _lbfgs_sphere(pop[i], time_limit=relax_time)
        e = _log_energy_fast(pts)
        pop[i] = pts
        energies.append(e)
    
    best_idx = np.argmin(energies)
    best_pts = pop[best_idx].copy()
    best_energy = energies[best_idx]
    
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        # Selection: Keep top 20% (elitism)
        k = max(2, pop_size // 5)
        sorted_indices = np.argsort(energies)
        elites = [pop[i].copy() for i in sorted_indices[:k]]
        
        # Create new population
        new_pop = []
        new_energies = []
        
        # Add elites
        for i in range(k):
            new_pop.append(elites[i])
            new_energies.append(energies[sorted_indices[i]])
        
        # Create offspring
        while len(new_pop) < pop_size:
            if time.time() > deadline - 15:
                break
                
            # Crossover: Uniform crossover from two random parents
            i1 = np.random.randint(0, pop_size)
            i2 = np.random.randint(0, pop_size)
            
            # Pick points from either parent
            mask = np.random.rand(N) > 0.5
            child = np.where(mask[:, np.newaxis], pop[i1], pop[i2])
            
            # Mutation: Add random noise
            noise = np.random.standard_normal((N, 3))
            dot_prods = np.sum(noise * child, axis=1, keepdims=True)
            noise -= dot_prods * child
            
            # Scale mutation
            scale = 0.1
            child = _exponential_map(child, noise * scale)
            child = _normalize(child)
            
            # Relax the child
            remaining = deadline - time.time() - 15
            if remaining < 1.0:
                break
                
            child_relaxed, e_child = _lbfgs_sphere(child, time_limit=min(1.5, remaining / 10))
            e_child = _log_energy_fast(child_relaxed)
            
            new_pop.append(child_relaxed)
            new_energies.append(e_child)
            
            # Update global best
            if e_child < best_energy:
                best_energy = e_child
                best_pts = child_relaxed.copy()
        
        # Replace population
        while len(new_pop) < pop_size:
            e_idx = np.random.randint(0, len(elites))
            child = elites[e_idx].copy()
            noise = np.random.standard_normal((N, 3))
            dot_prods = np.sum(noise * child, axis=1, keepdims=True)
            noise -= dot_prods * child
            child = _exponential_map(child, noise * 0.05)
            child = _normalize(child)
            child_relaxed, e_child = _lbfgs_sphere(child, time_limit=1.0)
            e_child = _log_energy_fast(child_relaxed)
            new_pop.append(child_relaxed)
            new_energies.append(e_child)
            
            if e_child < best_energy:
                best_energy = e_child
                best_pts = child_relaxed.copy()
                
        pop = new_pop
        energies = new_energies
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2s safety margin
    
    # Initializations - diverse
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N, base_start=2))
    inits.append(_halton_sphere(N, base_start=3))
    inits.append(_icosahedron_init(N))
    
    # Random inits
    for s in range(5):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci
    for s, scale in [(100, 0.1), (101, 0.2)]:
        rng = np.random.default_rng(s)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
        
    # Use Evolutionary Strategy
    pop_size = min(10, len(inits))
    initial_pop = [inits[i].copy() for i in range(pop_size)]
    
    # Run EA
    ea_time = deadline - time.time() - 20
    best_pts, best_energy = _evolve_population(initial_pop, ea_time, relax_time=2.0)
    
    # Final Polish with L-BFGS
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 5
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=500)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
