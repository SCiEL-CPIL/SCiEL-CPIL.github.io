"""Design a symmetric 32x32 photonic-crystal unit cell to maximize the TM band-4/5 gap."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = None  # set on first call

def _enforce_symmetry(cell):
    """Make the 32x32 cell fully symmetric: 180-deg rotation + transpose."""
    n = len(cell)
    a = np.array(cell, dtype=float)
    # 180-degree rotation symmetry
    a = (a + a[::-1, ::-1]) / 2.0
    # transpose symmetry
    a = (a + a.T) / 2.0
    # round to nearest integer (0 or 1)
    a = (a >= 0.5).astype(int)
    return a

def _make_cell(shape_params, size=32):
    """Create a superellipse cell. shape_params = (a, b, p) where the shape is
    |x/a|^p + |y/b|^p <= 1, centered in the cell. a and b are half-widths in pixels,
    p controls the exponent (p=2 circle, p=4 square-ish, p->inf square)."""
    a, b, p = shape_params
    if p <= 0:
        p = 0.1
    # Use a finite large exponent for very large p
    p_eff = p if p < 100 else 100.0
    cell = np.zeros((size, size), dtype=int)
    cx = (size - 1) / 2.0
    cy = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            dx = (i - cx) / a if a > 0 else 10.0
            dy = (j - cy) / b if b > 0 else 10.0
            val = abs(dx) ** p_eff + abs(dy) ** p_eff
            if val <= 1.0:
                cell[i, j] = 1
    return cell

def _fourier_coefficients(cell, eps_low, eps_high, size=32):
    """Compute 1/epsilon Fourier coefficients for the plane-wave method."""
    eps = np.where(cell > 0, eps_high, eps_low)
    inv_eps = 1.0 / eps
    # Fourier coefficients: G = (gx, gy) in units of 2*pi/a, a=1 lattice constant
    # c_G = (1/N^2) sum_{i,j} inv_eps(i,j) * exp(-i G . r_ij)
    # For a 32x32 grid, we compute coefficients for G vectors with |G| <= Gmax
    N = size
    c = {}
    # We need coefficients for all G in the plane-wave basis
    # For a 289-plane-wave basis, Gmax such that (2*Gmax+1)^2 <= 289 -> Gmax=8
    Gmax = 8
    for gx in range(-Gmax, Gmax + 1):
        for gy in range(-Gmax, Gmax + 1):
            # G vector in units of 2*pi (lattice constant = 1)
            # r_ij = (i/N, j/N) for pixel centers
            # c_G = (1/N^2) sum_ij inv_eps(i,j) exp(-i 2*pi (gx*i/N + gy*j/N))
            # This is just the 2D DFT of inv_eps scaled
            # Use numpy FFT for efficiency
            pass
    # Compute all coefficients at once using FFT
    # inv_eps is N x N, we want c_{gx,gy} = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-i 2*pi (gx i/N + gy j/N))
    # This is exactly the 2D DFT divided by N^2
    fft_vals = np.fft.fft2(inv_eps)
    c = {}
    for gx in range(-Gmax, Gmax + 1):
        for gy in range(-Gmax, Gmax + 1):
            # Map negative indices
            igx = gx % N
            igy = gy % N
            c[(gx, gy)] = fft_vals[igy, igx] / (N * N)
    return c

def _solve_tm_bands(cell, kx, ky, eps_low, eps_high, size=32, Gmax=8, n_bands=6):
    """Solve for TM bands at wave vector (kx, ky) using plane-wave expansion.
    Returns the first n_bands eigenvalues (omega^2)."""
    c = _fourier_coefficients(cell, eps_low, eps_high, size)
    # Build the matrix M where M_{G,G'} = |k+G| * c_{G-G'} * |k+G'|
    # For TM: (k+G) x (k+G') / eps_{G-G'} where 1/eps is Fourier decomposed
    # Actually for TM: the eigenvalue equation is
    # sum_{G'} |k+G'| * (1/eps)_{G-G'} * |k+G'| * E_{G'} = omega^2 E_G
    # where (1/eps)_{G} are Fourier coefficients of 1/epsilon
    
    Gvecs = []
    for gx in range(-Gmax, Gmax + 1):
        for gy in range(-Gmax, Gmax + 1):
            Gvecs.append((gx, gy))
    Npw = len(Gvecs)
    
    # Build matrix
    M = np.zeros((Npw, Npw), dtype=complex)
    for i, Gi in enumerate(Gvecs):
        for j, Gj in enumerate(Gvecs):
            # k + Gi
            kxGi = kx + Gi[0]
            kyGi = ky + Gi[1]
            kxGj = kx + Gj[0]
            kyGj = ky + Gj[1]
            # |k+Gi| * |k+Gj|
            mag_i = math.sqrt(kxGi**2 + kyGi**2)
            mag_j = math.sqrt(kxGj**2 + kyGj**2)
            # G - G'
            dGx = Gi[0] - Gj[0]
            dGy = Gi[1] - Gj[1]
            cval = c.get((dGx, dGy), 0.0)
            M[i, j] = mag_i * cval * mag_j
    
    # Symmetrize (should already be Hermitian if c_{-G} = conj(c_G))
    M = (M + M.conj().T) / 2.0
    
    # Solve eigenvalue problem
    try:
        eigvals, eigvecs = np.linalg.eigh(M)
    except:
        return [0.0] * n_bands
    
    # Return the n_bands smallest positive eigenvalues (omega^2)
    # Sort and take positive ones
    eigvals = np.sort(np.real(eigvals))
    # Filter to positive
    positive = eigvals[eigvals > 0]
    if len(positive) < n_bands:
        return list(positive) + [0.0] * (n_bands - len(positive))
    return list(positive[:n_bands])

def _evaluate_gap(cell, eps_low, eps_high, size=32):
    """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
    # Irreducible Brillouin zone for square lattice
    # Key points: Gamma (0,0), X (pi,0), M (pi,pi), and paths between them
    # 30 wave vectors covering boundary and interior
    kpoints = []
    # Gamma to X: k = (t*pi, 0), t in [0,1]
    for t in np.linspace(0, 1, 10):
        kpoints.append((t * math.pi, 0.0))
    # X to M: k = (pi, t*pi), t in [0,1]
    for t in np.linspace(0, 1, 10):
        kpoints.append((math.pi, t * math.pi))
    # M to Gamma: k = (t*pi, t*pi), t in [1,0] -> (pi - t*pi, pi - t*pi)
    for t in np.linspace(0, 1, 10):
        kpoints.append(((1 - t) * math.pi, (1 - t) * math.pi))
    
    # Remove duplicates approximately
    unique_k = []
    for k in kpoints:
        if not any(abs(k[0]-uk[0]) < 1e-6 and abs(k[1]-uk[1]) < 1e-6 for uk in unique_k):
            unique_k.append(k)
    
    min_omega5_sq = float('inf')
    max_omega4_sq = 0.0
    
    for kx, ky in unique_k:
        eigvals = _solve_tm_bands(cell, kx, ky, eps_low, eps_high, size, Gmax=8, n_bands=6)
        if len(eigvals) >= 5:
            omega4_sq = eigvals[3]  # 4th band (0-indexed: 3)
            omega5_sq = eigvals[4]  # 5th band (0-indexed: 4)
            if omega5_sq > 0:
                min_omega5_sq = min(min_omega5_sq, omega5_sq)
                max_omega4_sq = max(max_omega4_sq, omega4_sq)
    
    if min_omega5_sq == float('inf') or max_omega4_sq == 0:
        return 0.0
    
    min_omega5 = math.sqrt(min_omega5_sq)
    max_omega4 = math.sqrt(max_omega4_sq)
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap == 0:
        return 0.0
    
    return 2.0 * gap / midgap

def design(problem):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 180  # 20s safety margin
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    best_cell = None
    best_score = -float('inf')
    
    # Start with the seed: circle of radius 0.2
    # In our parameterization, a circle with radius r pixels: a=b=r, p=2
    # Radius 0.2 lattice constants = 0.2 * 32 = 6.4 pixels
    initial_params = (6.4, 6.4, 2.0)
    cell = _make_cell(initial_params, size)
    cell = _enforce_symmetry(cell)
    score = _evaluate_gap(cell, eps_low, eps_high, size)
    best_cell = cell
    best_score = score
    
    # Local search over superellipse parameters
    # Parameters: (a, b, p) with a=b for symmetry (since we need transpose symmetry)
    # Actually a and b must be equal for transpose symmetry
    # So we search over (a, p) where a is the half-width and p is the exponent
    
    # Grid search over a in [4, 12] and p in [1, 8]
    a_values = np.linspace(4, 12, 20)
    p_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    
    for a in a_values:
        if time.time() > _DEADLINE - 30:
            break
        for p in p_values:
            if time.time() > _DEADLINE - 30:
                break
            params = (a, a, p)
            cell = _make_cell(params, size)
            cell = _enforce_symmetry(cell)
            score = _evaluate_gap(cell, eps_low, eps_high, size)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Refine around the best found
    if best_score > -float('inf'):
        # Find the best parameters from the grid
        best_a = None
        best_p = None
        for a in a_values:
            for p in p_values:
                if time.time() > _DEADLINE - 20:
                    break
                params = (a, a, p)
                cell = _make_cell(params, size)
                cell = _enforce_symmetry(cell)
                score = _evaluate_gap(cell, eps_low, eps_high, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
                    best_a = a
                    best_p = p
        else:
            if time.time() > _DEADLINE - 20:
                break
    
    # If we found a good candidate, do a finer local search
    if best_a is not None and time.time() < _DEADLINE - 15:
        for da in [-1, -0.5, 0.5, 1]:
            for dp in [-0.5, 0.5]:
                if time.time() > _DEADLINE - 10:
                    break
                a_new = best_a + da
                p_new = max(0.5, best_p + dp)
                if a_new < 2 or a_new > 14:
                    continue
                params = (a_new, a_new, p_new)
                cell = _make_cell(params, size)
                cell = _enforce_symmetry(cell)
                score = _evaluate_gap(cell, eps_low, eps_high, size)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Return as nested list
    return best_cell.tolist()
# EVOLVE-BLOCK-END
