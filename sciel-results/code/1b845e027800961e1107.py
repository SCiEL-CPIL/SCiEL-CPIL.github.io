"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Compute gradient: dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Fast energy for optimizer
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))
    
    return energy, grads_tangent.flatten()


def _geodesic_dome_init(n):
    """Generate initial points using a geodesic dome subdivision."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    def subdivide_face(v0, v1, v2, subdiv=2):
        """Subdivide a triangle into subdiv^2 smaller triangles."""
        if subdiv == 1:
            return [(v0, v1, v2)]
        
        m01 = (v0 + v1) / 2.0
        m12 = (v1 + v2) / 2.0
        m20 = (v2 + v0) / 2.0
        m01 = m01 / np.linalg.norm(m01)
        m12 = m12 / np.linalg.norm(m12)
        m20 = m20 / np.linalg.norm(m20)
        
        return (subdivide_face(v0, m01, m20, subdiv-1) +
                subdivide_face(m01, v1, m12, subdiv-1) +
                subdivide_face(m20, m12, v2, subdiv-1) +
                subdivide_face(m01, m12, m20, subdiv-1))
    
    all_verts = set()
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        sub_faces = subdivide_face(v0, v1, v2, subdiv=2)
        for sf in sub_faces:
            for v in sf:
                all_verts.add(tuple(np.round(v, 12)))
    
    result = np.array(list(all_verts))
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    
    if len(result) > n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(result), size=n, replace=False)
        result = result[idx]
    elif len(result) < n:
        extra = _random_uniform_sphere(n - len(result), seed=99)
        result = np.vstack([result, extra])
    
    return result


def _local_minimize(points, deadline, maxiter=3000):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _basin_hopping_single(init_points, deadline, rng, max_iter=40):
    """
    Basin hopping with Metropolis acceptance:
    1. Local minimize
    2. Perturb random subset
    3. Local minimize
    4. Accept if better (or Metropolis)
    """
    current_points, current_energy = _local_minimize(init_points, deadline)
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.03
    T_min = 1e-10
    cooling = 0.97

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        n_perturb = rng.randint(1, 5)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * 0.5
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=2000)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def _parallel_tempering_step(replicas, energies, deadline, rng, maxiter=2000):
    """
    Parallel tempering step:
    - Each replica has points and energy
    - Try to swap adjacent replicas based on Metropolis criterion
    - Returns updated replicas and energies
    """
    n_rep = len(replicas)
    temps = [0.01 * (2.0 ** i) for i in range(n_rep)]
    
    for i in range(n_rep - 1):
        if time.time() > deadline - 5.0:
            break
        
        # Try to swap replicas i and i+1
        T1, T2 = temps[i], temps[i+1]
        
        # Compute delta for swap
        # Swap is accepted with probability min(1, exp(-delta/T))
        # where delta = (E2 - E1) * (1/T2 - 1/T1) ... actually for PT:
        # P(accept) = min(1, exp(-(E1/T2 + E2/T1 - E1/T1 - E2/T2)))
        delta_beta = (energies[i] / T2 + energies[i+1] / T1 - energies[i] / T1 - energies[i+1] / T2)
        
        if delta_beta <= 0 or rng.rand() < math.exp(-delta_beta):
            # Swap
            replicas[i], replicas[i+1] = replicas[i+1].copy(), replicas[i].copy()
            energies[i], energies[i+1] = energies[i+1], energies[i]
    
    return replicas, energies


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_energy = float('inf')
    
    # Generate diverse initializations
    inits = [
        _geodesic_dome_init(N),
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
    ]
    
    # First pass: local minimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        p_res, e_res = _local_minimize(init, deadline, maxiter=3000)
        
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()
    
    # Second pass: basin hopping from the best few initializations
    # Re-run local minimization to get updated energies for top candidates
    candidates = []
    for idx, init in enumerate(inits[:4]):  # Top 4 initializations
        if time.time() > deadline - 20.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=2000)
        candidates.append((p_res, e_res))
    
    # Sort by energy
    candidates.sort(key=lambda x: x[1])
    
    # Run basin hopping on the best candidate
    if candidates:
        best_init = candidates[0][0]
        if time.time() < deadline - 10.0:
            bh_points, bh_energy = _basin_hopping_single(best_init, deadline, rng, max_iter=30)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_points = bh_points.copy()
    
    # Third pass: parallel tempering with multiple replicas
    if time.time() < deadline - 15.0 and best_points is not None:
        n_rep = 4
        replicas = []
        energies = []
        
        # Initialize replicas from best point with small perturbations
        for r in range(n_rep):
            if r == 0:
                rep = best_points.copy()
            else:
                rep = best_points.copy()
                n_pert = rng.randint(3, 8)
                idxs = rng.choice(N, n_pert, replace=False)
                for idx in idxs:
                    tangent = _tangent_vector(rep[idx], rng)
                    rep[idx] = rep[idx] + 0.02 * tangent
                    rep[idx] = rep[idx] / np.linalg.norm(rep[idx])
                rep = _project_to_sphere(rep)
            
            e = _log_energy(rep)
            replicas.append(rep)
            energies.append(e)
        
        # Run parallel tempering iterations
        n_pt_iters = 20
        for it in range(n_pt_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Local minimize each replica
            for r in range(n_rep):
                if time.time() > deadline - 5.0:
                    break
                replicas[r], energies[r] = _local_minimize(replicas[r], deadline, maxiter=1000)
            
            # Try swaps
            replicas, energies = _parallel_tempering_step(replicas, energies, deadline, rng)
            
            # Update best
            for r in range(n_rep):
                if energies[r] < best_energy:
                    best_energy = energies[r]
                    best_points = replicas[r].copy()
    
    # Final polish with very tight tolerances
    if best_points is not None and time.time() < deadline - 5.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Additional polish pass if time allows
    if best_points is not None and time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < (1e-9) ** 2:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
