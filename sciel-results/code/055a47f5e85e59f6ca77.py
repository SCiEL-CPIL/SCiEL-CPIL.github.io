"""Parametric hexagonal lattice packing for 50 circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)

def _build_lattice(params):
    """Build circle centers from parametric hexagonal lattice.
    
    params: [dx, dy, x_off, y_off, row_phase]
    Rows have counts [7,6,7,6,7,6,7,4] = 50 total.
    Each row is a horizontal line of equally-spaced circles.
    Odd rows (0-indexed) are offset by dx/2 for hexagonal packing.
    row_phase allows per-row x-shift for the last (short) row.
    """
    dx, dy, x_off, y_off, row_phase = params
    row_counts = [7, 6, 7, 6, 7, 6, 7, 4]
    n_rows = len(row_counts)
    
    centers = []
    for row_idx, count in enumerate(row_counts):
        # Vertical position
        y = y_off + row_idx * dy
        
        # Horizontal: center the row
        if count > 1:
            x_start = x_off - (count - 1) * dx / 2.0
        else:
            x_start = x_off
        
        # Hexagonal offset for odd rows
        if row_idx % 2 == 1:
            x_start += dx / 2.0
        
        # Extra phase for the last short row (row 7)
        if row_idx == 7:
            x_start += row_phase * dx
        
        for col_idx in range(count):
            x = x_start + col_idx * dx
            centers.append([x, y])
    
    return np.array(centers, dtype=float)

def _lattice_objective(params):
    """Objective: minimize negative radius (i.e., maximize radius)."""
    centers = _build_lattice(params)
    
    # Check boundaries
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    if min_boundary < -1e-6:
        return 10.0  # penalty for going outside
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    radius = min(min_boundary, min_pair)
    return -radius

def _lattice_objective_grad(params):
    """Gradient of the lattice objective (numerical, for L-BFGS-B)."""
    # Use finite differences via scipy's default
    return None  # Let scipy handle it

def _optimize_lattice(initial_params, deadline):
    """Optimize the parametric lattice using L-BFGS-B."""
    # Bounds: dx in (0.05, 0.3), dy in (0.05, 0.3), x_off in (-0.1, 0.1),
    #         y_off in (-0.1, 0.1), row_phase in (-0.5, 0.5)
    bounds = [
        (0.05, 0.30),   # dx
        (0.05, 0.30),   # dy
        (-0.10, 0.10),  # x_off
        (-0.10, 0.10),  # y_off
        (-0.50, 0.50),  # row_phase
    ]
    
    result = minimize(
        _lattice_objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000},
    )
    
    if time.time() > deadline:
        return initial_params  # Return initial if timeout
    
    return result.x

def _local_refine(centers, n_iters=3000, deadline=None):
    """Fast local refinement: move one point at a time to improve radius."""
    best_centers = centers.copy().astype(float)
    best_radius = _fast_radius(best_centers)
    rng = np.random.RandomState(123)
    
    current_step = 0.015
    no_improve = 0
    max_no_improve = 80
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = it % N  # Cycle through points
        orig = best_centers[idx].copy()
        
        improved = False
        for attempt in range(4):
            s = current_step * (0.5 ** attempt)
            if s < 1e-5:
                break
            perturbation = rng.normal(0, s, size=2)
            candidate = orig + perturbation
            
            # Keep inside square
            np.clip(candidate, -0.5 + 1e-9, 0.5 - 1e-9, out=candidate)
            best_centers[idx] = candidate
            
            new_radius = _fast_radius(best_centers)
            
            if new_radius > best_radius + 1e-14:
                best_radius = new_radius
                improved = True
                break
            else:
                best_centers[idx] = orig
        
        if improved:
            no_improve = 0
            current_step = min(0.03, current_step * 1.01)
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                current_step = max(0.002, current_step * 0.7)
                no_improve = 0
    
    return best_centers, best_radius

def _multi_start_lattice(deadline):
    """Try multiple initial parameter sets for the lattice."""
    # Various initial guesses
    initial_params_list = [
        [0.14, 0.14, 0.0, 0.0, 0.0],      # basic hex
        [0.135, 0.145, 0.0, 0.0, 0.0],    # slightly compressed x
        [0.14, 0.135, 0.0, 0.0, 0.0],     # slightly compressed y
        [0.142, 0.138, 0.01, -0.01, 0.0], # shifted
        [0.138, 0.142, -0.01, 0.01, 0.1], # shifted other way
        [0.14, 0.14, 0.0, 0.0, 0.15],     # phase on short row
        [0.14, 0.14, 0.0, 0.0, -0.15],    # negative phase
        [0.136, 0.144, 0.02, 0.0, 0.05],  # asymmetric
        [0.144, 0.136, -0.02, 0.0, -0.05],# asymmetric other
        [0.14, 0.14, 0.0, 0.0, 0.3],      # large phase
    ]
    
    best_params = initial_params_list[0]
    best_val = _lattice_objective(best_params)
    
    for params in initial_params_list:
        if time.time() > deadline:
            break
        val = _lattice_objective(params)
        if val < best_val:
            best_val = val
            best_params = params
    
    return best_params

def construct():
    """Main function: parametric lattice optimization + local refinement."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    # Phase 1: Find best initial lattice parameters
    best_params = _multi_start_lattice(deadline)
    
    # Phase 2: Optimize lattice parameters with L-BFGS-B
    if time.time() < deadline - 30.0:
        optimized_params = _optimize_lattice(best_params, deadline - 30.0)
        best_params = optimized_params
    
    # Build centers from optimized lattice
    centers = _build_lattice(best_params)
    
    # Ensure valid (clip to square)
    np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9, out=centers)
    
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Phase 3: Local refinement from lattice solution
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        n_iters = max(500, int(min(5000, 5000 * remaining / 30.0)))
        refined, r = _local_refine(best_centers, n_iters=n_iters, deadline=deadline)
        if r > best_radius:
            best_radius = r
            best_centers = refined
    
    # Phase 4: Second round of local refinement with smaller steps
    if time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        n_iters = max(200, int(min(3000, 3000 * remaining / 20.0)))
        refined, r = _local_refine(best_centers, n_iters=n_iters, deadline=deadline)
        if r > best_radius:
            best_radius = r
            best_centers = refined
    
    # Phase 5: Try a second lattice with different row arrangement
    # Row counts [6,7,6,7,6,7,6,5] = 50
    if time.time() < deadline - 15.0:
        row_counts_alt = [6, 7, 6, 7, 6, 7, 6, 5]
        # Quick optimization with this arrangement
        def _build_lattice_alt(params):
            dx, dy, x_off, y_off, row_phase = params
            centers = []
            for row_idx, count in enumerate(row_counts_alt):
                y = y_off + row_idx * dy
                if count > 1:
                    x_start = x_off - (count - 1) * dx / 2.0
                else:
                    x_start = x_off
                if row_idx % 2 == 1:
                    x_start += dx / 2.0
                if row_idx == 7:
                    x_start += row_phase * dx
                for col_idx in range(count):
                    x = x_start + col_idx * dx
                    centers.append([x, y])
            return np.array(centers, dtype=float)
        
        def _lattice_objective_alt(params):
            c = _build_lattice_alt(params)
            bd = 0.5 - np.abs(c)
            mb = np.min(bd)
            if mb < -1e-6:
                return 10.0
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            mp = np.min(dists) / 2.0
            return -min(mb, mp)
        
        bounds = [
            (0.05, 0.30), (0.05, 0.30),
            (-0.10, 0.10), (-0.10, 0.10),
            (-0.50, 0.50),
        ]
        
        init_alt = [0.14, 0.14, 0.0, 0.0, 0.0]
        result_alt = minimize(
            _lattice_objective_alt, init_alt,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'maxfun': 5000},
        )
        
        if time.time() < deadline - 10.0:
            centers_alt = _build_lattice_alt(result_alt.x)
            np.clip(centers_alt, -0.5 + 1e-9, 0.5 - 1e-9, out=centers_alt)
            r_alt = _compute_radius(centers_alt)
            
            if r_alt > best_radius:
                best_radius = r_alt
                best_centers = centers_alt.copy()
            
            # Local refine the alt lattice too
            if time.time() < deadline - 5.0:
                remaining = deadline - time.time() - 5.0
                n_iters = max(200, int(min(3000, 3000 * remaining / 20.0)))
                refined_alt, r = _local_refine(best_centers, n_iters=n_iters, deadline=deadline)
                if r > best_radius:
                    best_radius = r
                    best_centers = refined
    
    # Final validation
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback: simple grid
        m = 8
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].astype(float)
    
    return best_centers

# EVOLVE-BLOCK-END
