"""Spectral packing: low-dimensional Fourier coefficient search for 300 circles in unit disk."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Feasible common radius: min of boundary clearance and half min pairwise distance."""
    if points is None or len(points) < 2:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dist_boundary))
    d = pdist(points)
    min_pair = float(np.min(d)) / 2.0
    return min(min_boundary, min_pair)


def _spectral_positions(coeffs, n_modes=8):
    """
    Generate N circle centers from a low-dimensional coefficient vector.
    
    coeffs: array of shape (2 * (n_modes+1)^2,) representing complex Fourier coefficients
            on a 2D grid of modes up to n_modes.
    
    The position of point k is determined by the real part of a band-limited function
    evaluated at N points on a regular grid, then mapped to the disk.
    
    We use a simpler approach: parameterize positions as a sum of radial basis functions
    or as a low-rank deformation of a hexagonal lattice.
    
    Actually, let's use a cleaner spectral approach:
    - Start with a hexagonal lattice of ~N points
    - Apply a smooth, band-limited displacement field parameterized by coeffs
    - The displacement field is a sum of a few low-frequency Fourier modes
    """
    # Base: hexagonal lattice
    spacing = 0.055
    k = int(1.0 / spacing) + 3
    base_pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                base_pts.append((x, y))
    
    base_pts = np.array(base_pts)
    dists = np.linalg.norm(base_pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    base_pts = base_pts[idx]
    
    if len(base_pts) < N:
        # Fallback: concentric rings
        angles = 2 * np.pi * np.arange(N) / N
        base_pts = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    # Now apply a band-limited displacement field
    # coeffs encodes a small number of 2D Fourier modes
    # displacement(x, y) = sum_{m,n in modes} a_{m,n} * exp(i*(m*x + n*y))
    # We use real-valued displacement: Re and Im parts
    
    n_modes = 4  # modes -1..1 in each direction => 9 modes, 18 coeffs
    # coeffs shape: (2, 3, 3) -> flattened to (18,)
    # modes: m in {-1, 0, 1}, n in {-1, 0, 1}
    
    coeffs = np.array(coeffs).reshape(-1)
    
    # Parse coefficients: for each (m, n) mode, we have (real_amp, imag_amp)
    # Total modes: 3x3 = 9, so 18 coefficients
    if len(coeffs) < 18:
        coeffs = np.zeros(18)
    coeffs = coeffs[:18]
    
    modes = []
    for m in [-1, 0, 1]:
        for n in [-1, 0, 1]:
            modes.append((m, n))
    
    disp_x = np.zeros(len(base_pts))
    disp_y = np.zeros(len(base_pts))
    
    for idx_mode, (m, n) in enumerate(modes):
        a_re = coeffs[2 * idx_mode]
        a_im = coeffs[2 * idx_mode + 1]
        # displacement contribution: (a_re * cos(theta) - a_im * sin(theta), a_re * sin(theta) + a_im * cos(theta))
        theta = m * base_pts[:, 0] + n * base_pts[:, 1]
        disp_x += a_re * np.cos(theta) - a_im * np.sin(theta)
        disp_y += a_re * np.sin(theta) + a_im * np.cos(theta)
    
    points = base_pts.copy()
    points[:, 0] += disp_x
    points[:, 1] += disp_y
    
    # Hard radial cutoff: project any point outside the disk back inside
    norms = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        if norms[i] > 0.99:
            points[i] *= 0.99 / norms[i]
    
    return points


def _spectral_objective(coeffs):
    """Negative of the packing radius (to be minimized)."""
    points = _spectral_positions(coeffs)
    r = _compute_radius(points)
    return -r


def _hexagonal_lattice(spacing):
    """Hexagonal lattice points inside the unit disk (with margin), closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _concentric_rings(n_rings):
    """Concentric rings packing; returns points or None."""
    if n_rings <= 0:
        return None
    if n_rings == 1:
        ring_counts = [N]
    else:
        remaining = N - 1
        base = remaining // (n_rings - 1)
        extra = remaining % (n_rings - 1)
        ring_counts = [1] + [base + (1 if i < extra else 0) for i in range(n_rings - 1)]
    pts = []
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = ring_idx / n_rings * 0.95
        for i in range(count):
            angle = 2 * np.pi * i / count
            pts.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
    if len(pts) != N:
        return None
    return np.array(pts)


def _refine_local(points, deadline, max_iters=3000, step_size=0.015):
    """Hill-climbing local refinement; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    rng = np.random.default_rng(12345)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        best_local = original
        best_local_r = best_r
        for _ in range(8):
            if time.time() > deadline:
                break
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            points[idx] = original + np.array([dx, dy])
            r = _compute_radius(points)
            if r > best_local_r:
                best_local_r = r
                best_local = points[idx].copy()
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s wall-clock limit

    candidates = []

    # 1. Generate initial candidates from various lattice/ring configurations
    for trial in range(30):
        spacing = 0.05 + trial * 0.004
        pts = _hexagonal_lattice(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    for n_rings in range(4, 15):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # 2. Spectral optimization: search in low-dimensional coefficient space
    #    Use Nelder-Mead on the 18-dimensional coefficient vector
    rng = np.random.default_rng(42)
    
    # Try multiple random initializations in coefficient space
    n_spectral_tries = 5
    for s_try in range(n_spectral_tries):
        if time.time() > deadline - 60.0:
            break
        
        # Initialize coefficients with small random values
        init_coeffs = rng.normal(0, 0.005, size=18)
        
        try:
            result = minimize(
                _spectral_objective,
                init_coeffs,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < _spectral_objective(init_coeffs):
                spectral_points = _spectral_positions(result.x)
                spectral_r = _compute_radius(spectral_points)
                if spectral_r > best_r:
                    best_points = spectral_points
                    best_r = spectral_r
        except Exception:
            pass
    
    # 3. Also try spectral optimization starting from near-zero coefficients
    #    (i.e., starting from the base hexagonal lattice)
    if time.time() < deadline - 40.0:
        zero_coeffs = np.zeros(18)
        try:
            result = minimize(
                _spectral_objective,
                zero_coeffs,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            spectral_points = _spectral_positions(result.x)
            spectral_r = _compute_radius(spectral_points)
            if spectral_r > best_r:
                best_points = spectral_points
                best_r = spectral_r
        except Exception:
            pass

    # 4. Local refinement on the best result so far
    if time.time() < deadline - 15.0:
        refined_pts, refined_r = _refine_local(best_points, deadline - 5.0,
                                               max_iters=5000, step_size=0.01)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # 5. Final tight polish
    if time.time() < deadline - 2.0:
        refined_pts, refined_r = _refine_local(best_points, deadline,
                                               max_iters=2000, step_size=0.003)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    return best_points

# EVOLVE-BLOCK-END
