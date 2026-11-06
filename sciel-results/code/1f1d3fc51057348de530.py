"""Morse cluster global minima via lattice initialization and multi-restart relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _morse_energy(positions, rho):
    """Compute Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _enforce_min_dist(P, min_d=0.5):
    """Ensure all pair distances >= min_d by small local pushes."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(30):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                d = P[i] - P[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-12:
                        d = np.array([1.0, 0, 0])
                    else:
                        d = d / dist
                    push = (min_d - dist) / 2.0 + 0.005
                    P[i] += push * d
                    P[j] -= push * d
                    ok = False
        if ok:
            break
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from FCC lattice centered at origin."""
    points = []
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * spacing
    
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
    grid_size = int(r_max / spacing) + 3
    for ix in range(-grid_size, grid_size + 1):
        for iy in range(-grid_size, grid_size + 1):
            for iz in range(-grid_size, grid_size + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * spacing + b
                    if np.linalg.norm(p) <= r_max:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    if len(points) >= n:
        points = points[:n]
    else:
        while len(points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            points = np.vstack([points, p])
    
    points -= np.mean(points, axis=0)
    return points


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * (k + (1.0/3.0) * ((j % 2) + 1))
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from BCC lattice centered at origin."""
    r_max = spacing * (n ** (1.0 / 3.0) + 2)
    
    points = []
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ]) * spacing
    
    grid_size = int(r_max / spacing) + 3
    for ix in range(-grid_size, grid_size + 1):
        for iy in range(-grid_size, grid_size + 1):
            for iz in range(-grid_size, grid_size + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) * spacing + b
                    if np.linalg.norm(p) <= r_max:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    
    if len(points) >= n:
        points = points[:n]
    else:
        while len(points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            points = np.vstack([points, p])
    
    points -= np.mean(points, axis=0)
    return points


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                pts.append(np.array([i, j, k]) * a)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _icosahedral_shells(n):
    """Generate points on nested icosahedral shells (good for small clusters)."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    shells = [1.0, 1.45, 1.85, 2.2, 2.5, 2.8, 3.1]
    for s in shells:
        if len(pts) >= n:
            break
        shell_pts = ico * s
        rotations = [np.eye(3),
                     np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                     np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])]
        for rot in rotations:
            rotated = shell_pts @ rot.T
            for p in rotated:
                if len(pts) >= n:
                    break
                pts.append(p)
    pts = np.array(pts[:n])
    return pts


def _random_compact(n, spacing=1.0, rng=None):
    """Generate random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _perturb(positions, rng, scale=0.3):
    """Add random perturbation to positions."""
    noise = rng.normal(0, scale, size=positions.shape)
    return positions + noise


def _langevin_refine(positions, potential, rho, rng, time_budget=15.0):
    """
    Physics-inspired simulated annealing via Langevin dynamics.
    Uses morse_energy_gradient to drive motion with adaptive step sizes
    and a linear cooling schedule.
    """
    n = len(positions)
    P = np.array(positions, dtype=float)
    
    # Initial energy
    try:
        E0 = _morse_energy(P, rho)
    except Exception:
        return positions, float('inf')
    
    # Langevin parameters
    m = 1.0  # mass
    T_start = 5.0 * rho  # high temperature
    T_end = 0.01 * rho   # low temperature
    dt = 0.01            # time step
    
    # Adaptive friction coefficient
    gamma = 0.5
    
    best_P = P.copy()
    best_E = E0
    
    start_time = time.time()
    n_steps = 0
    max_steps = int(time_budget / dt)  # rough estimate
    
    # Linear cooling schedule
    for step in range(max_steps):
        elapsed = time.time() - start_time
        if elapsed > time_budget or _time_left() < 5.0:
            break
        
        # Current temperature (linear cooling)
        frac = step / max(1, max_steps)
        T = T_start * (1.0 - frac) + T_end * frac
        
        # Compute gradient
        try:
            E, grad = morse_energy_gradient(P, rho)
        except Exception:
            break
        
        if not np.isfinite(E):
            break
        
        if E < best_E:
            best_E = E
            best_P = P.copy()
        
        # Langevin update: m*dv/dt = -grad - gamma*v + sqrt(2*m*T*gamma)*noise
        # Simplified: P_new = P - dt * grad/m + sqrt(2*dt*T/m)*noise
        noise = rng.normal(0, np.sqrt(2.0 * dt * T / m), P.shape)
        P = P - dt * grad / m + noise
        
        # Enforce minimum distance constraint softly
        P = _enforce_min_dist(P, 0.5)
        
        # Keep cluster centered
        P -= P.mean(axis=0)
        
        n_steps += 1
    
    return best_P, best_E


def construct(instance):
    """Find low-energy Morse cluster configurations using lattice init and multi-restart relaxation."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size * 1000 + int(rho))
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # Lattice initializations
    try:
        inits.append(_fcc_points(size, spacing=1.0))
    except Exception:
        pass
    
    try:
        inits.append(_hcp_points(size, spacing=1.0))
    except Exception:
        pass
    
    try:
        inits.append(_bcc_points(size, spacing=1.0))
    except Exception:
        pass
    
    try:
        inits.append(_sc_points(size, spacing=1.0))
    except Exception:
        pass
    
    # Icosahedral shells for smaller clusters
    if size <= 60:
        try:
            inits.append(_icosahedral_shells(size))
        except Exception:
            pass
    
    # Random compact clusters
    for seed_offset in [0, 7, 13, 23, 42]:
        try:
            rng_sub = np.random.default_rng(size * 1000 + int(rho) + seed_offset)
            inits.append(_random_compact(size, spacing=1.0, rng=rng_sub))
        except Exception:
            pass
    
    # Perturbed FCC variants
    if len(inits) > 0:
        for scale in [0.08, 0.15, 0.2]:
            try:
                perturbed = _perturb(inits[0].copy(), rng, scale=scale)
                inits.append(perturbed)
            except Exception:
                pass
    
    # Try each initialization
    for init_idx, init_pos in enumerate(inits):
        if _time_left() < 5.0:
            break
        
        try:
            init_pos = _enforce_min_dist(init_pos, 0.5)
            energy, positions = relax(init_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Phase 2: Langevin refinement of the best configuration
    if best_positions is not None and _time_left() > 20.0:
        try:
            refined_pos, refined_E = _langevin_refine(
                best_positions, potential, rho, rng, time_budget=15.0
            )
            if refined_E < best_energy:
                best_energy = refined_E
                best_positions = refined_pos.copy()
        except Exception:
            pass
    
    # Phase 3: Perturb-best-and-relax cycles with decreasing scales
    if best_positions is not None and _time_left() > 10.0:
        scales = [0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
        for scale in scales:
            if _time_left() < 4.0:
                break
            n_tries = 3 if scale > 0.1 else 4
            for _ in range(n_tries):
                if _time_left() < 4.0:
                    break
                try:
                    perturbed = _perturb(best_positions.copy(), rng, scale=scale)
                    perturbed = _enforce_min_dist(perturbed, 0.5)
                    energy, positions = relax(perturbed, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except Exception:
                    continue
    
    # Phase 4: Larger perturbations if time remains
    if best_positions is not None and _time_left() > 8.0:
        for scale in [0.4, 0.5]:
            if _time_left() < 4.0:
                break
            for _ in range(2):
                if _time_left() < 4.0:
                    break
                try:
                    perturbed = _perturb(best_positions.copy(), rng, scale=scale)
                    perturbed = _enforce_min_dist(perturbed, 0.5)
                    energy, positions = relax(perturbed, potential, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
                except Exception:
                    continue
    
    # Phase 5: Final Langevin refinement if time remains
    if best_positions is not None and _time_left() > 10.0:
        try:
            refined_pos, refined_E = _langevin_refine(
                best_positions, potential, rho, rng, time_budget=min(10.0, _time_left() - 5.0)
            )
            if refined_E < best_energy:
                best_energy = refined_E
                best_positions = refined_pos.copy()
        except Exception:
            pass
    
    # Fallback: if no valid result, return a simple compact cluster
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, rng=rng)
        best_positions = _enforce_min_dist(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
