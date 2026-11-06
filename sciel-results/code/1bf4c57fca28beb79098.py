"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _fourier_coefficients(cell, N=289):
    """Compute exact Fourier coefficients of the pixel map for plane-wave method.
    cell: 32x32 nested list of 0/1
    Returns: dict mapping (mx, my) to coefficient value (complex)
    We only need up to |mx|,|my| <= 12 (since N=289 = 17^2, so max index is 16, but we cap at 12 for speed)
    Actually N=289 means we use a 17x17 grid of plane waves, so indices -16 to 16.
    """
    size = len(cell)
    # Convert to numpy array
    eps = np.array(cell, dtype=float)
    # epsilon = 1 + 11 * cell (air=1, dielectric=12)
    eps = 1.0 + 11.0 * eps
    
    # For TM polarization, we need 1/epsilon in the eigenvalue problem
    # The plane-wave method for TM uses:
    # sum_{G'} (k-G)·(k-G') * (1/epsilon)(G-G') * c_{G'} = (omega^2/c^2) * c_k
    # So we need Fourier coefficients of 1/epsilon
    
    inv_eps = 1.0 / eps
    
    # Compute Fourier coefficients via direct summation
    # (mx, my) range: we need |mx|, |my| up to 16 for 17x17 grid
    max_m = 16
    coeffs = {}
    for mx in range(-max_m, max_m + 1):
        for my in range(-max_m, max_m + 1):
            # Fourier coefficient: (1/N^2) sum_{i,j} f(i,j) exp(-2πi (mx*i + my*j)/N)
            # But our grid is 32x32 pixels representing one unit cell
            # The pixel centers are at (i+0.5)/32, (j+0.5)/32
            # Fourier coeff = integral over cell of f(x,y) exp(-i 2π (mx x + my y)) dxdy
            # Approximate by pixel sum: sum_{i,j} f(i,j) * exp(-i 2π (mx*(i+0.5)/32 + my*(j+0.5)/32)) * (1/32)^2
            # Actually for exact coefficients, we should integrate each pixel as a rectangle
            # But the problem says "exact Fourier coefficients of the pixel map"
            # For a pixel map, the standard approach is to treat each pixel as a point or as a small rectangle
            # Let's use the point-sample approximation which is standard for this type of problem
            # Actually, let's use the rectangle integration for better accuracy
            
            # Simple approach: treat each pixel value as constant over its area
            # Fourier coeff = (1/32^2) * sum_{i,j} inv_eps[i,j] * exp(-i 2π (mx*(i+0.5)/32 + my*(j+0.5)/32))
            # This is the midpoint rule approximation
            
            phase_x = np.exp(-2j * np.pi * mx * (np.arange(size) + 0.5) / size)
            phase_y = np.exp(-2j * np.pi * my * (np.arange(size) + 0.5) / size)
            
            # 2D Fourier coefficient
            coeff = np.sum(inv_eps * np.outer(phase_x, phase_y)) / (size * size)
            coeffs[(mx, my)] = coeff
    return coeffs


def _solve_tm_bands(coeffs, kx, ky, num_bands=5, N=289):
    """Solve for the first num_bands TM bands at wavevector (kx, ky).
    coeffs: dict of Fourier coefficients of 1/epsilon
    Returns: list of omega values (in units of a/λ)
    """
    # Plane wave indices: -16 to 16 (17 values each direction)
    max_g = 16
    g_indices = list(range(-max_g, max_g + 1))
    
    # Build the matrix
    n = len(g_indices)
    mat = np.zeros((n, n), dtype=complex)
    
    for i, gi in enumerate(g_indices):
        for j, gj in enumerate(g_indices):
            # (k - G_i) · (k - G_j)
            dx = kx - gi
            dy = ky - gj
            dot = dx * (kx - g_indices[i]) + dy * (ky - g_indices[i])
            # Wait, need to be more careful
            # G_i = (gi, gj), G_j = (gj, gk) -- no, let me restructure
            
    # Let me redo this properly
    # Plane waves are labeled by (mx, my) where mx, my in -16..16
    # For each pair (mx1,my1) and (mx2,my2):
    # Matrix element = (k - G1) · (k - G2) * (1/eps)(G1 - G2)
    # where G1 = (mx1, my1), G2 = (mx2, my2)
    
    mat = np.zeros((n, n), dtype=complex)
    for i, (mx1, my1) in enumerate([(g, h) for g in g_indices for h in g_indices]):
        pass
    
    # Actually let me flatten properly
    plane_waves = [(mx, my) for mx in g_indices for my in g_indices]
    n_pw = len(plane_waves)
    mat = np.zeros((n_pw, n_pw), dtype=complex)
    
    for i, (mx1, my1) in enumerate(plane_waves):
        for j, (mx2, my2) in enumerate(plane_waves):
            # (k - G1) · (k - G2)
            dx = kx - mx1
            dy = ky - my1
            dot = dx * (kx - mx2) + dy * (ky - my2)
            
            # Fourier coefficient of 1/eps at (G1 - G2)
            dm = mx1 - mx2
            dn = my1 - my2
            if (dm, dn) in coeffs:
                coeff = coeffs[(dm, dn)]
            else:
                coeff = 0.0
            
            mat[i, j] = dot * coeff
    
    # Solve eigenvalue problem
    # The eigenvalues are omega^2 * a^2 / (2π)^2 or similar
    # For TM: det[(k-G)·(k-G') / eps(G-G') - ω^2/c^2 δ] = 0
    # So eigenvalues of mat give ω^2 a^2 / c^2 (in some normalization)
    
    # Use scipy for eigenvalue solve
    from scipy.linalg import eigh
    
    # The matrix should be Hermitian for real coefficients
    # But with complex coefficients it may not be. Let's check.
    # For a real dielectric function, 1/eps is real, so the matrix is real and symmetric
    # if we use real k. Let's just use eigh on the real part if imaginary is small
    
    # Actually, the matrix elements are: (k-G_i)·(k-G_j) * (1/eps)(G_i-G_j)
    # If 1/eps is real, then mat[i,j] = mat[j,i]* (conjugate if complex)
    # Since (1/eps)(-G) = conj((1/eps)(G)) for real eps, and (k-G_i)·(k-G_j) is real
    # So mat[i,j] = mat[j,i] * conj factor... it should be Hermitian
    
    # For safety, use eigsh or eigh on the Hermitian part
    mat_herm = (mat + mat.conj().T) / 2
    eigvals = eigh(mat_herm, subset_by_index=[0, num_bands - 1])
    
    # Eigenvalues are in units of (2π/a)^2 * omega^2/c^2
    # So omega = sqrt(eigval) * c / (2π/a) ... let's just return sqrt(eigvals)
    # The actual normalization doesn't matter for the ratio
    omega_vals = np.sqrt(np.maximum(eigvals, 0))
    return omega_vals.tolist()


def _compute_gap(cell, coeffs, k_points):
    """Compute the gap-midgap ratio between bands 4 and 5."""
    max_omega4 = 0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        omegas = _solve_tm_bands(coeffs, kx, ky, num_bands=5)
        if len(omegas) >= 5:
            max_omega4 = max(max_omega4, omegas[3])
            min_omega5 = min(min_omega5, omegas[4])
    
    if min_omega5 == float('inf'):
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap == 0:
        return -1.0
    
    return 2 * gap / midgap


def _generate_irreducible_bz_points(n=30):
    """Generate 30 wave vectors covering the irreducible Brillouin zone.
    The irreducible BZ for a square lattice is the triangle from (0,0) to (π/a,0) to (π/a,π/a).
    In units of 2π/a, it's from (0,0) to (0.5, 0) to (0.5, 0.5).
    """
    points = []
    # Sample along the boundary and interior
    # Edge 1: (0,0) to (0.5, 0)
    # Edge 2: (0.5, 0) to (0.5, 0.5)
    # Edge 3: (0.5, 0.5) to (0, 0)
    # And interior points
    
    # Let's use a grid approach within the triangle
    # Triangle vertices: (0,0), (0.5,0), (0.5,0.5)
    
    # Generate points on a fine grid and keep those in the triangle
    grid_size = 12
    for i in range(grid_size):
        for j in range(grid_size):
            x = i / (grid_size - 1) * 0.5
            y = j / (grid_size - 1) * 0.5
            # Check if in triangle: x >= 0, y >= 0, y <= x (for the triangle from (0,0)-(0.5,0)-(0.5,0.5))
            # Actually the triangle is: x from 0 to 0.5, y from 0 to x
            if x >= 0 and y >= 0 and y <= x + 1e-10:
                points.append((x, y))
    
    # If we have more than 30, subsample
    if len(points) > 30:
        # Take evenly spaced points
        indices = np.linspace(0, len(points) - 1, 30, dtype=int)
        points = [points[i] for i in indices]
    elif len(points) < 30:
        # Add more points
        extra = 30 - len(points)
        for i in range(extra):
            t = i / extra
            points.append((0.25 * (1 - t), 0.25 * t))  # line from (0.25, 0) to (0.25, 0.25)
    
    return points[:30]


def _make_symmetric(cell):
    """Ensure the cell is symmetric under the square symmetry group."""
    size = len(cell)
    # Symmetrize: average over all 8 symmetries, then threshold
    # For a binary cell, we can just enforce:
    # 1. cell == cell[::-1] row-wise (vertical flip)
    # 2. cell == cell[::-1] column-wise (horizontal flip)  
    # 3. cell == cell.T (transpose)
    
    # Start with the original
    result = [row[:] for row in cell]
    
    # Enforce horizontal symmetry (left-right)
    for i in range(size):
        for j in range(size // 2):
            avg = (result[i][j] + result[i][size - 1 - j]) / 2
            val = 1 if avg >= 0.5 else 0
            result[i][j] = val
            result[i][size - 1 - j] = val
    
    # Enforce vertical symmetry (top-bottom)
    for i in range(size // 2):
        for j in range(size):
            avg = (result[i][j] + result[size - 1 - i][j]) / 2
            val = 1 if avg >= 0.5 else 0
            result[i][j] = val
            result[size - 1 - i][j] = val
    
    # Enforce transpose symmetry
    for i in range(size):
        for j in range(i + 1, size):
            avg = (result[i][j] + result[j][i]) / 2
            val = 1 if avg >= 0.5 else 0
            result[i][j] = val
            result[j][i] = val
    
    return result


def _fourier_harmonic_design(a20, a02, a22, phase20, phase02, phase22, size=32):
    """Create a symmetric cell from Fourier harmonics.
    a20, a02, a22: amplitudes of (2,0), (0,2), (2,2) modes
    phase20, phase02, phase22: phases
    """
    cell = []
    for i in range(size):
        row = []
        y = (i + 0.5) / size - 0.5
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            # Dielectric fraction from Fourier series
            # f(x,y) = base + a20*cos(2π*2*x + phase20) + a02*cos(2π*2*y + phase02) + a22*cos(2π*(2*x+2*y) + phase22)
            # For symmetry, we need specific phase choices
            # cos(4πx) is symmetric about x=0 and x=±0.5
            # Let's use: f = 0.5 + a20*cos(4πx) + a02*cos(4πy) + a22*cos(4π(x+y))
            # This is automatically symmetric under x->-x, y->-y, x<->y
            
            val = 0.5 + a20 * np.cos(4 * np.pi * x + phase20) \
                        + a02 * np.cos(4 * np.pi * y + phase02) \
                        + a22 * np.cos(4 * np.pi * (x + y) + phase22)
            # Clamp and threshold
            val = max(0, min(1, val))
            row.append(1 if val >= 0.5 else 0)
        cell.append(row)
    
    # Enforce full symmetry
    cell = _make_symmetric(cell)
    return cell


def design(problem):
    """Design a symmetric 32x32-pixel unit cell for maximum 4-5 TM band gap."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s CPU budget, use 110s to be safe
    
    size = problem["pixels"]
    k_points = _generate_irreducible_bz_points(30)
    
    # Initial design: four-rod pattern at (±a/4, ±a/4) which showed score 5.265
    # This corresponds to strong (2,0) and (0,2) Fourier components
    
    best_cell = None
    best_score = -1.0
    
    # Start with the four-rod pattern
    cell = []
    for i in range(size):
        y = (i + 0.5) / size - 0.5
        row = []
        for j in range(size):
            x = (j + 0.5) / size - 0.5
            # Four rods at (±0.25, ±0.25) with radius ~0.15
            d1 = (x - 0.25)**2 + (y - 0.25)**2
            d2 = (x + 0.25)**2 + (y - 0.25)**2
            d3 = (x - 0.25)**2 + (y + 0.25)**2
            d4 = (x + 0.25)**2 + (y + 0.25)**2
            val = 1 if min(d1, d2, d3, d4) < 0.15**2 else 0
            row.append(val)
        cell.append(row)
    cell = _make_symmetric(cell)
    
    # Compute initial score
    coeffs = _fourier_coefficients(cell)
    score = _compute_gap(cell, coeffs, k_points)
    best_cell = [row[:] for row in cell]
    best_score = score
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_cell
    
    # Now optimize using Fourier harmonic parameterization
    # Parameters: a20, a02, a22, phase20, phase02, phase22
    # Initial guess from four-rod pattern: strong (2,0) and (0,2), moderate (2,2)
    
    # Use a simple grid search / hill climbing
    params = [0.3, 0.3, 0.2, 0.0, 0.0, 0.0]  # a20, a02, a22, ph20, ph02, ph22
    
    step_sizes = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
    
    max_iter = 200
    for iteration in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        improved = False
        for p_idx in range(len(params)):
            for direction in [-1, 1]:
                new_params = params[:]
                new_params[p_idx] += direction * step_sizes[p_idx]
                
                # Keep amplitudes in reasonable range
                if p_idx < 3:
                    new_params[p_idx] = max(-0.5, min(0.5, new_params[p_idx]))
                else:
                    # Phases: keep in [0, 2π)
                    new_params[p_idx] = new_params[p_idx] % (2 * np.pi)
                
                test_cell = _fourier_harmonic_design(
                    new_params[0], new_params[1], new_params[2],
                    new_params[3], new_params[4], new_params[5], size
                )
                
                test_coeffs = _fourier_coefficients(test_cell)
                test_score = _compute_gap(test_cell, test_coeffs, k_points)
                
                if test_score > best_score:
                    best_score = test_score
                    best_cell = [row[:] for row in test_cell]
                    params = new_params[:]
                    improved = True
                    break
        
        # Reduce step sizes if no improvement
        if not improved:
            step_sizes = [s * 0.7 for s in step_sizes]
            if max(step_sizes) < 0.001:
                break
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        return best_cell
    
    # Return the best cell found
    return best_cell
# EVOLVE-BLOCK-END
