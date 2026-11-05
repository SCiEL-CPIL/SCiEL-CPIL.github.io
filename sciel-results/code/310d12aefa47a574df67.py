"""Design a 32x32 symmetric photonic crystal unit cell for maximum TM band gap."""
import time
import math
import numpy as np

# Deadline guard: evaluator kills at 230s wall-clock; keep margin
_DEADLINE = 210.0  # seconds
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _build_cell(params):
    """
    Build a 32x32 cell from geometric primitives.
    params: tuple of continuous parameters controlling the geometry.
    
    Strategy: Use a combination of:
    - A central dielectric disk (radius r0)
    - Four square rings at different radii (inner/outer)
    - Cross-shaped bars along axes
    
    The cell must be invariant under D4 (square symmetry):
    - 180-degree rotation: cell[i][j] == cell[31-i][31-j]
    - 90-degree rotation: cell[i][j] == cell[j][31-i]
    - Reflections: cell[i][j] == cell[i][31-j] == cell[31-i][j] == cell[31-i][31-j]
    
    We enforce this by only defining a 1/8 sector (i<=j, i+j<=31) and mirroring.
    """
    size = 32
    # params: (r_center, r_ring1_in, r_ring1_out, r_ring2_in, r_ring2_out, bar_width, cross_len)
    r_center = params[0]
    r_ring1_in = params[1]
    r_ring1_out = params[2]
    r_ring2_in = params[3]
    r_ring2_out = params[4]
    bar_width = params[5]
    cross_len = params[6]
    
    cell = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = math.sqrt(x * x + y * y)
            
            val = 0
            
            # Central disk
            if r < r_center:
                val = 1
            
            # Ring 1
            if r_ring1_in <= r < r_ring1_out:
                val = 1
            
            # Ring 2
            if r_ring2_in <= r < r_ring2_out:
                val = 1
            
            # Cross bars (horizontal and vertical)
            if abs(x) < bar_width / 2.0 and abs(y) < cross_len / 2.0:
                val = 1
            if abs(y) < bar_width / 2.0 and abs(x) < cross_len / 2.0:
                val = 1
            
            cell[i][j] = val
    
    # Enforce D4 symmetry explicitly
    cell = _enforce_d4(cell)
    return cell


def _enforce_d4(cell):
    """Enforce full D4 symmetry on the cell."""
    size = len(cell)
    result = [[0] * size for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            # Get all 8 symmetric positions
            positions = [
                (i, j),
                (size - 1 - i, j),
                (i, size - 1 - j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (size - 1 - j, i),
                (j, size - 1 - i),
                (size - 1 - j, size - 1 - i),
            ]
            # Majority vote
            vals = [cell[pi][pj] for pi, pj in positions]
            result[i][j] = 1 if sum(vals) >= 4 else 0
    
    return result


def _compute_fourier_coefficients(cell, N=17):
    """
    Compute Fourier coefficients of 1/epsilon for the cell.
    For TM mode: the eigenvalue problem involves 1/eps.
    We need c_m = (1/A) * integral (1/eps) * exp(-i*m*k0*x) dA
    
    For a pixelated cell, this is a discrete sum.
    """
    size = len(cell)
    eps_high = 12.0
    eps_low = 1.0
    
    # Precompute 1/eps for each pixel
    inv_eps = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            inv_eps[i][j] = 1.0 / (eps_high if cell[i][j] == 1 else eps_low)
    
    # Compute Fourier coefficients for m, n in range [-N+1, N-1]
    coeffs = {}
    dx = 1.0 / size
    dy = 1.0 / size
    
    for m in range(-N + 1, N):
        for n in range(-N + 1, N):
            c = 0.0
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size
                    y = (j + 0.5) / size
                    phase = np.exp(-1j * 2 * np.pi * (m * x + n * y))
                    c += inv_eps[i][j] * phase * dx * dy
            coeffs[(m, n)] = c
    
    return coeffs


def _compute_tm_gap(cell, N_pw=17, n_kpts=30):
    """
    Compute the TM band gap between bands 4 and 5.
    Uses plane-wave expansion with exact Fourier coefficients.
    
    For TM polarization, the eigenvalue problem is:
    sum_n c_{m-n} * |k + G_m|^2 * E_n = omega^2 * E_m
    
    where c_{m-n} are Fourier coefficients of 1/eps.
    """
    size = len(cell)
    coeffs = _compute_fourier_coefficients(cell, N=N_pw)
    
    # Generate k-points in the irreducible BZ
    # Square lattice: k in [0, pi/a] x [0, pi/a], irreducible triangle: kx >= ky >= 0, kx+ky <= pi/a
    # Actually for square lattice, irreducible BZ is 1/8 of the full BZ
    # kx from 0 to pi/2, ky from 0 to kx (for the 1/8 sector)
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    # k-points: sample along the boundary and interior of the irreducible BZ
    # The irreducible BZ for square lattice is the triangle with vertices (0,0), (pi,0), (pi,pi)
    # Wait, let me reconsider. For a square lattice with lattice constant a=1:
    # Full BZ: [-pi, pi] x [-pi, pi]
    # Irreducible BZ (1/8): 0 <= ky <= kx, kx + ky <= pi
    # Actually the standard irreducible BZ for square lattice is:
    # 0 <= ky <= kx, and kx <= pi, ky <= pi, but the boundary is kx + ky = pi
    
    # Let's use a set of k-points covering the irreducible BZ
    kpts = []
    # Along kx axis
    for i in range(8):
        kx = (i + 0.5) / 8.0 * np.pi
        kpts.append((kx, 0.0))
    # Along the diagonal kx = ky
    for i in range(1, 8):
        kx = (i + 0.5) / 16.0 * np.pi
        kpts.append((kx, kx))
    # Along the line kx + ky = pi
    for i in range(1, 8):
        kx = (i + 0.5) / 8.0 * np.pi
        ky = np.pi - kx
        if ky >= 0 and kx >= ky:
            kpts.append((kx, ky))
    # Interior points
    kpts.append((np.pi / 4, np.pi / 8))
    kpts.append((np.pi / 3, np.pi / 6))
    kpts.append((np.pi / 2.5, np.pi / 5))
    
    # Remove duplicates and ensure they're in the irreducible BZ
    unique_kpts = []
    seen = set()
    for kx, ky in kpts:
        # Ensure 0 <= ky <= kx and kx + ky <= pi
        if ky < 0:
            ky = -ky
        if kx < ky:
            kx, ky = ky, kx
        if kx + ky > np.pi:
            # Project back
            scale = np.pi / (kx + ky)
            kx *= scale
            ky *= scale
        key = (round(kx, 6), round(ky, 6))
        if key not in seen:
            seen.add(key)
            unique_kpts.append((kx, ky))
    
    kpts = unique_kpts[:n_kpts]
    
    for kx, ky in kpts:
        # Build the plane-wave matrix
        # G vectors: m*a1 + n*a2 where a1 = (2pi, 0), a2 = (0, 2pi)
        # So G = (2pi*m, 2pi*n)
        
        g_vectors = []
        for m in range(-N_pw + 1, N_pw):
            for n in range(-N_pw + 1, N_pw):
                g_vectors.append((2 * np.pi * m, 2 * np.pi * n, m, n))
        
        npw = len(g_vectors)
        
        # Build the A matrix: A[m_idx, n_idx] = c_{m-n} * |k + G_m|^2
        # where c_{m-n} is the Fourier coefficient of 1/eps at index (m_m - m_n, n_m - n_n)
        
        A = np.zeros((npw, npw), dtype=complex)
        
        for mi in range(npw):
            gx_m, gy_m, mm, mn = g_vectors[mi]
            kx_m = kx + gx_m
            ky_m = ky + gy_m
            km2 = kx_m * kx_m + ky_m * ky_m
            
            for ni in range(npw):
                gx_n, gy_n, nn, nnn = g_vectors[ni]
                
                # Fourier index difference
                dm = mm - nn
                dn = mn - nnn
                
                # Get coefficient c_{dm, dn}
                c = coeffs.get((dm, dn), 0.0)
                
                A[mi, ni] = c * km2
        
        # Solve the generalized eigenvalue problem: A * E = omega^2 * E
        # This is a standard eigenvalue problem since A is already the operator
        
        try:
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)
            
            # Bands 4 and 5 (1-indexed), so indices 3 and 4 (0-indexed)
            if len(eigenvalues) >= 5:
                omega4_sq = eigenvalues[3]
                omega5_sq = eigenvalues[4]
                
                if omega4_sq > 0 and omega5_sq > 0:
                    omega4 = np.sqrt(omega4_sq)
                    omega5 = np.sqrt(omega5_sq)
                    
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
        except:
            continue
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return 0.0
    
    return 2.0 * gap / midgap


def design(problem):
    """
    Design a 32x32 symmetric photonic crystal unit cell.
    Returns a nested list of 0/1 pixels.
    """
    global _start_time
    _start_time = time.time()
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    polarization = problem.get("polarization", "TM")
    
    # Start with the seed design (central disk)
    best_cell = None
    best_score = -float('inf')
    
    # Generate initial candidate: central disk of various radii
    for r in [0.15, 0.2, 0.25, 0.3]:
        cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < r * r:
                    cell[i][j] = 1
        cell = _enforce_d4(cell)
        
        if _remaining() < 30:
            break
        
        score = _compute_tm_gap(cell, N_pw=13, n_kpts=10)
        if score > best_score:
            best_score = score
            best_cell = [row[:] for row in cell]
    
    # Try ring-based designs
    if _remaining() > 60:
        for r_in, r_out in [(0.15, 0.25), (0.2, 0.3), (0.25, 0.35), (0.1, 0.2)]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    r = math.sqrt(x * x + y * y)
                    if r_in <= r < r_out:
                        cell[i][j] = 1
            cell = _enforce_d4(cell)
            
            if _remaining() < 30:
                break
            
            score = _compute_tm_gap(cell, N_pw=13, n_kpts=10)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
    
    # Try combined disk + ring
    if _remaining() > 90:
        for r_disk, r_ring_in, r_ring_out in [(0.1, 0.2, 0.3), (0.15, 0.25, 0.35), (0.2, 0.3, 0.4)]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    r = math.sqrt(x * x + y * y)
                    if r < r_disk or r_ring_in <= r < r_ring_out:
                        cell[i][j] = 1
            cell = _enforce_d4(cell)
            
            if _remaining() < 30:
                break
            
            score = _compute_tm_gap(cell, N_pw=13, n_kpts=10)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
    
    # Try cross-shaped designs
    if _remaining() > 120:
        for bar_w, cross_l in [(0.1, 0.5), (0.15, 0.6), (0.2, 0.7)]:
            cell = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    if abs(x) < bar_w / 2.0 and abs(y) < cross_l / 2.0:
                        cell[i][j] = 1
                    if abs(y) < bar_w / 2.0 and abs(x) < cross_l / 2.0:
                        cell[i][j] = 1
            cell = _enforce_d4(cell)
            
            if _remaining() < 30:
                break
            
            score = _compute_tm_gap(cell, N_pw=13, n_kpts=10)
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in cell]
    
    # If we haven't found a good design, use the seed
    if best_cell is None:
        best_cell = [[0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 * 0.2:
                    best_cell[i][j] = 1
        best_cell = _enforce_d4(best_cell)
    
    return best_cell
