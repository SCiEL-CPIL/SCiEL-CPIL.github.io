"""Optimize logarithmic energy of 150 points on the unit sphere using Riemannian optimization."""
import numpy as np
import math
import time

N = 150
DEADLINE = 143.0  # safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    
    grad = weighted_sum - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (faster, no gradient)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


def _exp_map(pts, tangent_vecs, step_size):
    """Riemannian exponential map on the unit sphere.
    
    For a point x on S^2 and tangent vector v at x,
    exp_x(v) = x*cos(||v||) + v/sin(||v||) * sin(||v||) * ... 
    Actually: exp_x(v) = x*cos(||v||) + (v/||v||)*sin(||v||)
    """
    v_norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(v_norms > 1e-15, v_norms, 1.0)
    cos_factor = np.cos(v_norms * step_size)
    sin_factor = np.sin(v_norms * step_size) / safe_norms
    # Where norm is ~0, cos(0)=1, sin(0)/1 = 0, so result = x
    result = pts * cos_factor + tangent_vecs * sin_factor
    # For points where v_norm was ~0, result should just be pts
    mask = v_norms.squeeze() < 1e-15
    if np.any(mask):
        result[mask] = pts[mask]
    return result


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n):
    """Generate points using icosahedron subdivision for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    if n == 12:
        return ico
    
    fib = _fibonacci_sphere(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _so3_rotate(pts, rng):
    """Apply random SO(3) rotation to points."""
    # Use QR decomposition for proper random rotation
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    return pts @ Q.T


def _riemannian_optimize(pts0, max_time, lr_init=0.05, tol=1e-12):
    """Riemannian gradient descent with exponential map, momentum, and adaptive step.
    
    This respects the sphere geometry and converges faster than Euclidean projection.
    """
    start_time = time.time()
    
    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    # Momentum in tangent space
    velocity = np.zeros_like(pts)
    momentum = 0.85
    
    # Adaptive step size
    lr = lr_init
    lr_max = 0.5
    lr_min = 1e-6
    
    prev_e = e
    stall_count = 0
    
    for iteration in range(200000):
        if time.time() - start_time >= max_time:
            break
        
        # Compute tangent gradient
        g_tan = _tangent_project(pts, g)
        g_norm = np.linalg.norm(g_tan)
        
        if g_norm < tol:
            break
        
        # Momentum update in tangent space
        velocity = momentum * velocity - lr * g_tan
        
        # Project velocity to tangent space (should already be, but ensure)
        v_dot_n = np.sum(velocity * pts, axis=1, keepdims=True)
        velocity = velocity - v_dot_n * pts
        
        # Exponential map update
        new_pts = _exp_map(pts, velocity, 1.0)
        
        # Compute new energy
        new_e, new_g = _energy_and_grad(new_pts)
        
        # Backtracking line search
        if new_e > e + 1e-10:
            # Reduce learning rate and retry
            for _ in range(8):
                lr *= 0.5
                velocity = momentum * velocity - lr * g_tan
                v_dot_n = np.sum(velocity * pts, axis=1, keepdims=True)
                velocity = velocity - v_dot_n * pts
                new_pts = _exp_map(pts, velocity, 1.0)
                new_e, new_g = _energy_and_grad(new_pts)
                if new_e <= e + 1e-10:
                    break
            else:
                # Failed to find good step, reduce more
                lr *= 0.1
                if lr < lr_min:
                    # Try smaller random perturbation
                    break
                continue
        
        # Accept step
        pts = new_pts
        e = new_e
        g = new_g
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Adaptive learning rate
        improvement = prev_e - e
        if improvement > 1e-8:
            stall_count = 0
            # Increase lr if making good progress
            if improvement > 1e-6 and lr < lr_max:
                lr = min(lr * 1.1, lr_max)
        else:
            stall_count += 1
            if stall_count > 20:
                lr *= 0.7
                stall_count = 0
        
        prev_e = e
    
    return best_pts, best_e


def _simulated_annealing(pts, max_time, rng):
    """Simulated annealing with batch moves to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = 0.03
    T_min = 1e-10
    cooling = 0.997
    
    while time.time() - start_time < max_time and T > T_min:
        # Move a small batch of points
        batch_size = rng.integers(1, 8)
        indices = rng.choice(N, size=batch_size, replace=False)
        
        new_pts = current.copy()
        for idx in indices:
            # Random direction in tangent space
            rand_dir = rng.standard_normal(3)
            rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-15:
                rand_dir /= norm
            
            # Move along random direction
            step_size = T * 0.5
            new_pt = current[idx].copy()
            new_pt += step_size * rand_dir
            new_pt = new_pt / np.linalg.norm(new_pt)
            new_pts[idx] = new_pt
        
        new_e, _ = _energy_and_grad(new_pts)
        delta_e = new_e - current_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-12)):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
    
    return best, best_e


def _polish(pts, max_time):
    """High-precision polish with very tight tolerance."""
    return _riemannian_optimize(pts, max_time, lr_init=0.01, tol=1e-14)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))
    
    # 3. Perturbed Fibonacci
    for _ in range(3):
        fib = _fibonacci_sphere(N)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 4. Random points
    for _ in range(4):
        starts.append(_random_sphere(N, rng))
    
    # 5. Rotated Fibonacci
    for _ in range(2):
        fib = _fibonacci_sphere(N)
        starts.append(_so3_rotate(fib, rng))
    
    # 6. Clustered starts
    for _ in range(2):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    n_starts = len(starts)
    
    # Time allocation: 60% for initial optimizations, 20% for SA, 20% for polish
    elapsed = time.time() - start_time
    remaining_total = DEADLINE - elapsed
    
    opt_budget = remaining_total * 0.60
    sa_budget = remaining_total * 0.20
    polish_budget = remaining_total * 0.20
    
    time_per_opt = opt_budget / n_starts
    time_per_opt = max(2.0, min(12.0, time_per_opt))
    
    for i, s in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
        
        opt_time = min(time_per_opt, remaining - 10)
        if opt_time < 2:
            break
        
        try:
            pts_opt, e_opt = _riemannian_optimize(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Simulated annealing on best result
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 15
        if remaining > 5:
            try:
                pts_sa, e_sa = _simulated_annealing(best_pts, remaining, rng)
                if e_sa < best_energy:
                    best_energy = e_sa
                    best_pts = pts_sa.copy()
            except Exception:
                pass
    
    # Final high-precision polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5
        if remaining > 3:
            try:
                pts_final, e_final = _polish(best_pts, remaining)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    # One more polish if time allows
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 3
        if remaining > 2:
            try:
                pts_final2, e_final2 = _polish(best_pts, remaining)
                if e_final2 < best_energy:
                    best_energy = e_final2
                    best_pts = pts_final2.copy()
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
