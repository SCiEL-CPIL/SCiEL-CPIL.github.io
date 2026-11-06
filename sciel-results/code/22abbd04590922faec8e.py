import numpy as np
import time
import sys

def _get_d4_orbits(n=32):
    """Get all D4 orbits of pixels in an n x n grid."""
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            # Identity
            orbit.add((i, j))
            # 90 deg rotation
            orbit.add((j, n-1-i))
            # 180 deg rotation
            orbit.add((n-1-i, n-1-j))
            # 270 deg rotation
            orbit.add((n-1-j, i))
            # Mirror x (horizontal flip)
            orbit.add((i, n-1-j))
            # Mirror y (vertical flip)
            orbit.add((n-1-i, j))
            # Diagonal mirror
            orbit.add((j, i))
            # Anti-diagonal mirror
            orbit.add((n-1-j, n-1-i))
            orbit = list(orbit)
            for p in orbit:
                seen.add(p)
            orbits.append(orbit)
    return orbits

def _apply_orbits(base, orbit_values, orbits):
    """Apply orbit values to a base grid. orbit_values is array of 0/1."""
    result = base.copy()
    for orbit, val in zip(orbits, orbit_values):
        for (i, j) in orbit:
            result[i, j] = val
    return result

def _compute_eps_fourier(cell_grid, size=32, eps_low=1.0, eps_high=12.0):
    """Compute Fourier coefficients of epsilon from the pixel grid."""
    eps = eps_low + (eps_high - eps_low) * cell_grid.astype(float)
    F = np.fft.fft2(eps)
    F /= (size * size)
    return F

def _solve_tm_bands(kx, ky, eps_F, G_vecs, size=32):
    """Solve TM eigenvalue problem at given k-point. Returns sorted eigenvalues (omega^2/c^2)."""
    nG = len(G_vecs)
    
    # k + G vectors
    kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
    kG2 = kG[:, 0]**2 + kG[:, 1]**2
    
    # Build the matrix A where A[n,m] = eps_F[G_n - G_m]
    # eps_F is a 2D FFT, so we need to index it properly
    # G_vecs are in units of 2*pi, so G_n - G_m = (2*pi*(dm, dn))
    # The FFT index for (dm, dn) is (dm % size, dn % size)
    
    A = np.zeros((nG, nG), dtype=complex)
    for n in range(nG):
        for m in range(nG):
            dm = round(G_vecs[n, 0] / (2*np.pi)) - round(G_vecs[m, 0] / (2*np.pi))
            dn = round(G_vecs[n, 1] / (2*np.pi)) - round(G_vecs[m, 1] / (2*np.pi))
            idx_m = dm % size
            idx_n = dn % size
            A[n, m] = eps_F[idx_n, idx_m]
    
    # D = diag(kG2)
    # Solve: A v = (1/omega^2) D v
    # Or: D^{-1} A v = (1/omega^2) v
    # Let mu = 1/omega^2, then D^{-1} A v = mu v
    # So omega^2 = 1/mu
    
    # Handle case where some kG2 are zero (at Gamma point)
    # Add small regularization
    kG2_reg = kG2.copy()
    kG2_reg[kG2_reg < 1e-12] = 1e-12
    
    D_inv = np.diag(1.0 / kG2_reg)
    B = D_inv @ A
    
    # B should be Hermitian (since eps is real, A is Hermitian, D is real diagonal)
    # Ensure Hermiticity
    B = (B + B.conj().T) / 2
    
    try:
        eigvals = np.linalg.eigvalsh(B)
        eigvals = eigvals.real
        eigvals = np.sort(eigvals)
        # Convert to omega^2 = 1/mu
        # But wait, we need to be careful. Let me re-derive.
        # The eigenvalue problem is: A E = mu D E where mu = 1/(omega/c)^2
        # So (omega/c)^2 = 1/mu
        # But mu = 1/omega^2, so this is consistent.
        # However, eigvalsh gives mu values, so omega^2 = 1/mu
        # But some mu might be zero or negative due to numerical issues
        omega2 = np.zeros_like(eigvals)
        valid = np.abs(eigvals) > 1e-12
        omega2[valid] = 1.0 / eigvals[valid]
        # For invalid ones, set to a large value (high band)
        omega2[~valid] = np.inf
        return np.sort(omega2)
    except:
        return None

def _get_k_points(n_per_seg=10):
    """Get k-points along the irreducible BZ boundary for square lattice."""
    k_points = []
    # Segment 1: Gamma(0,0) to X(pi, 0)
    for i in range(n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((t * np.pi, 0.0))
    # Segment 2: X(pi, 0) to M(pi, pi)
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((np.pi, t * np.pi))
    # Segment 3: M(pi, pi) to Gamma(0, 0)
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    return k_points

def _compute_gap_score(cell_grid, k_points, G_vecs, size=32, eps_low=1.0, eps_high=12.0, n_bands=6):
    """Compute the 4-5 TM band gap score."""
    eps_F = _compute_eps_fourier(cell_grid, size, eps_low, eps_high)
    
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = _solve_tm_bands(kx, ky, eps_F, G_vecs, size)
        if eigvals is None or len(eigvals) < 5:
            return -1.0
        # eigvals are omega^2, sorted ascending
        # Band 4 is index 3, Band 5 is index 4
        omega4 = np.sqrt(max(0, eigvals[3]))
        omega5 = np.sqrt(max(0, eigvals[4]))
        omega4_vals.append(omega4)
        omega5_vals.append(omega5)
    
    if not omega4_vals:
        return -1.0
    
    max_omega4 = max(omega4_vals)
    min_omega5 = min(omega5_vals)
    
    if min_omega5 <= max_omega4:
        if (min_omega5 + max_omega4) > 1e-12:
            return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return -1.0
    else:
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

def _is_symmetric(cell):
    """Check if cell has D4 symmetry."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
    return True

def _fast_objective(cell_grid, size=32):
    """
    Fast approximate objective function.
    For a large band gap, we want strong scattering at specific G vectors.
    The gap size is roughly proportional to the Fourier component magnitude |eps_G|
    at the G vector that couples the bands. For the 4-5 gap in a square lattice,
    the relevant G vectors are often around (1,1) or (2,0) in reciprocal units.
    We maximize the sum of squares of the first few non-zero Fourier coefficients
    weighted by their expected coupling strength.
    """
    eps = 1.0 + 11.0 * cell_grid.astype(float)
    F = np.fft.fft2(eps)
    F /= (size * size)
    
    # We want to maximize the "scattering power"
    # Typically, the gap is opened by the G=(1,1) or G=(2,0) components.
    # Let's look at the magnitude of the first ring of G vectors.
    # G vectors are indexed by (m, n) in the FFT.
    # (0,0) is DC.
    # (1,0), (0,1), (1,1), (1,-1) etc.
    
    # Extract specific coefficients
    # Note: FFT output is ordered such that index [i,j] corresponds to frequency (i, j)
    # with wrapping.
    # F[1, 0] -> G=(1,0)
    # F[0, 1] -> G=(0,1)
    # F[1, 1] -> G=(1,1)
    # F[31, 1] -> G=(-1,1)
    
    # We sum the magnitudes squared of the first few shells
    # Shell 1: (1,0), (0,1), (-1,0), (0,-1)
    # Shell 2: (1,1), (1,-1), (-1,1), (-1,-1)
    # Shell 3: (2,0), (0,2), etc.
    
    # For a 32x32 grid, the DC term is huge. We exclude it.
    # We want to maximize the contrast in the higher orders.
    
    # Let's just take the sum of |F|^2 for all non-DC terms, 
    # but weighted to favor the specific orders that open the gap.
    # A common heuristic for photonic crystals is that a larger gap corresponds
    # to a more "square-like" or "cross-like" dielectric distribution,
    # which has strong (1,1) and (2,0) components.
    
    # Let's compute the power in the first 3 shells.
    power = 0.0
    # Shell 1
    power += np.abs(F[1, 0])**2 + np.abs(F[0, 1])**2 + np.abs(F[31, 0])**2 + np.abs(F[0, 31])**2
    # Shell 2 (diagonals)
    power += np.abs(F[1, 1])**2 + np.abs(F[1, 31])**2 + np.abs(F[31, 1])**2 + np.abs(F[31, 31])**2
    # Shell 3 (2,0 etc)
    power += np.abs(F[2, 0])**2 + np.abs(F[0, 2])**2 + np.abs(F[30, 0])**2 + np.abs(F[0, 30])**2
    
    # Also, we want a reasonable fill fraction. 
    # Too much dielectric closes the gap or shifts it.
    # Ideal fill fraction is around 0.3-0.5.
    fill = np.sum(cell_grid) / (size * size)
    
    # Penalize if fill is too high or too low
    fill_penalty = 0.0
    if fill < 0.2:
        fill_penalty = (0.2 - fill) * 10.0
    elif fill > 0.6:
        fill_penalty = (fill - 0.6) * 10.0
        
    return power - fill_penalty

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin for 230s limit
    
    # Get D4 orbits
    orbits = _get_d4_orbits(size)
    n_orbits = len(orbits)
    
    # Set up G vectors for plane-wave solver (for final validation)
    # Use a diamond of half-width 6 for 32x32 grid
    half = 6
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:
                G_vecs.append((2*np.pi*m, 2*np.pi*n))
    G_vecs = np.array(G_vecs, dtype=float)
    
    # Get k-points for validation
    k_points = _get_k_points(n_per_seg=8)
    
    # Initialize with a four-rod design
    # Rods at (8,8), (23,8), (8,23), (23,23) with radius ~4
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    radius_sq = 16.0  # radius = 4
    
    for i in range(size):
        for j in range(size):
            for (cx, cy) in centers:
                dx = i - cx
                dy = j - cy
                if dx*dx + dy*dy < radius_sq:
                    cell[i, j] = 1
                    break
    
    # Convert to orbit-based representation
    orbit_values = np.zeros(n_orbits, dtype=int)
    for oi, orbit in enumerate(orbits):
        i, j = orbit[0]
        orbit_values[oi] = cell[i, j]
    
    current_cell = _apply_orbits(np.zeros((size, size), dtype=int), orbit_values, orbits)
    
    # Compute initial fast score and exact score
    current_fast_score = _fast_objective(current_cell, size)
    current_exact_score = _compute_gap_score(current_cell, k_points, G_vecs, size, eps_low, eps_high)
    
    best_cell = current_cell.copy()
    best_exact_score = current_exact_score
    best_orbit_values = orbit_values.copy()
    
    # Simulated annealing / Hill Climbing using FAST objective
    # We use the fast objective for the vast majority of steps.
    # We only compute the exact score for the best candidates found.
    
    T_start = 1.0
    T_end = 0.01
    n_iter_max = 50000 # Can afford more iterations with fast objective
    
    rng = np.random.RandomState(42)
    
    # Track the best fast score seen so far
    best_fast_score = current_fast_score
    
    for it in range(n_iter_max):
        if time.time() > deadline - 30:
            break
        
        # Temperature schedule
        t_frac = it / max(1, n_iter_max - 1)
        T = T_start + (T_end - T_start) * t_frac
        
        # Pick a random orbit to flip
        oi = rng.randint(n_orbits)
        
        # Flip the orbit
        old_val = orbit_values[oi]
        orbit_values[oi] = 1 - old_val
        new_cell = _apply_orbits(np.zeros((size, size), dtype=int), orbit_values, orbits)
        
        # Compute new FAST score
        new_fast_score = _fast_objective(new_cell, size)
        
        delta_fast = new_fast_score - current_fast_score
        
        if new_fast_score > current_fast_score:
            # Accept
            current_fast_score = new_fast_score
            current_cell = new_cell
            
            if new_fast_score > best_fast_score:
                best_fast_score = new_fast_score
                # Check if this is a good candidate for exact evaluation
                # We evaluate exact score every time we find a new best fast score,
                # but we can throttle this if it's too slow.
                # For now, let's evaluate exact score for the new best fast score.
                exact_score = _compute_gap_score(new_cell, k_points, G_vecs, size, eps_low, eps_high)
                if exact_score > best_exact_score:
                    best_exact_score = exact_score
                    best_cell = new_cell.copy()
                    best_orbit_values = orbit_values.copy()
        else:
            # Metropolis criterion
            if T > 1e-12 and rng.rand() < np.exp(delta_fast / T):
                current_fast_score = new_fast_score
                current_cell = new_cell
            else:
                # Revert
                orbit_values[oi] = old_val
    
    # Final check: ensure best_cell is symmetric and valid
    best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    if not _is_symmetric(best_list):
        # Fallback: use the orbit-based best which is guaranteed symmetric
        best_cell = _apply_orbits(np.zeros((size, size), dtype=int), best_orbit_values, orbits)
        best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
        
    return best_list
