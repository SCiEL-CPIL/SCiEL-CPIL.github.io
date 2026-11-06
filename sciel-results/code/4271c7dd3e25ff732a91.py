"""Photonic crystal design: evolutionary search with C4v symmetry and plane-wave evaluation."""
import time
import math
import random
import numpy as np

def _apply_c4v(cell):
    """Enforce C4v symmetry (transpose, row-reverse, col-reverse) on a 32x32 binary cell."""
    n = len(cell)
    # Work with a flat numpy array for speed
    arr = np.array(cell, dtype=np.int8).copy()
    # C4v: the cell must be symmetric under:
    # 1. Transpose (reflection across main diagonal)
    # 2. Row reversal (reflection across vertical axis through center)
    # 3. Column reversal (reflection across horizontal axis through center)
    # These generate the full C4v group.
    # For a 32x32 grid, the fundamental domain is the top-left 16x16 quadrant
    # (rows 0-15, cols 0-15), but we need to be careful about the center.
    # Actually for even size 32, the symmetry axes pass between pixels.
    # Pixel (i,j) maps to:
    #   (j, i)         - transpose
    #   (i, n-1-j)     - col reverse
    #   (n-1-i, j)     - row reverse
    #   (n-1-i, n-1-j) - 180 rotation
    #   (n-1-j, i)     - transpose + col reverse
    #   (n-1-j, n-1-i) - 180 + transpose
    #   (j, n-1-i)     - transpose + row reverse
    #
    # The fundamental domain is the 16x16 top-left block (i in 0..15, j in 0..15).
    # Each pixel in the fundamental domain determines 8 symmetric partners (or 4 if on axis).
    # For n=32 (even), no pixel lies exactly on the symmetry axes (which are between pixels),
    # so each fundamental domain pixel has exactly 8 partners.
    
    n = 32
    half = n // 2  # 16
    
    # Extract fundamental domain
    fd = arr[:half, :half].copy()
    
    # Reconstruct full cell from fundamental domain
    result = np.zeros((n, n), dtype=np.int8)
    for i in range(half):
        for j in range(half):
            v = fd[i, j]
            result[i, j] = v
            result[j, i] = v          # transpose
            result[i, n-1-j] = v      # col reverse
            result[n-1-j, i] = v      # transpose + col reverse
            result[n-1-i, j] = v      # row reverse
            result[n-1-i, n-1-j] = v  # 180 rotation
            result[j, n-1-i] = v      # transpose + row reverse
            result[n-1-j, n-1-i] = v  # 180 + transpose
    
    return result

def _c4v_from_fd(fd):
    """Build full 32x32 cell from 16x16 fundamental domain."""
    n = 32
    half = 16
    result = np.zeros((n, n), dtype=np.int8)
    for i in range(half):
        for j in range(half):
            v = fd[i, j]
            result[i, j] = v
            result[j, i] = v
            result[i, n-1-j] = v
            result[n-1-j, i] = v
            result[n-1-i, j] = v
            result[n-1-i, n-1-j] = v
            result[j, n-1-i] = v
            result[n-1-j, n-1-i] = v
    return result

def _fd_from_cell(cell):
    """Extract 16x16 fundamental domain from 32x32 cell."""
    arr = np.array(cell, dtype=np.int8)
    return arr[:16, :16].copy()

def _fourier_coeffs(cell, n_pw):
    """
    Compute Fourier coefficients of 1/epsilon for a 32x32 pixel cell.
    Returns dict mapping (mx, my) to coefficient for |mx|,|my| <= n_pw.
    """
    n = 32
    eps_high = 12.0
    eps_low = 1.0
    arr = np.array(cell, dtype=np.float64)
    
    # 1/epsilon map
    inv_eps = np.where(arr == 1, 1.0/eps_high, 1.0/eps_low)
    
    # Compute DFT
    # The unit cell is 1x1, pixels are at (i+0.5)/n, (j+0.5)/n
    # epsilon(x,y) = sum of pixel values * indicator functions
    # Fourier coeff: f_{mn} = integral epsilon(x,y) exp(-i 2pi (mx x + my y)) dx dy
    # For pixel grid: f_{mn} = (1/n^2) sum_{i,j} eps_{ij} exp(-i 2pi (mx(i+0.5)/n + my(j+0.5)/n))
    
    coeffs = {}
    indices = list(range(-n_pw, n_pw+1))
    for mx in indices:
        for my in indices:
            if mx == 0 and my == 0:
                coeffs[(mx, my)] = np.mean(inv_eps)
                continue
            # Use numpy for speed
            phase_x = np.exp(-2j * np.pi * mx * (np.arange(n) + 0.5) / n)
            phase_y = np.exp(-2j * np.pi * my * (np.arange(n) + 0.5) / n)
            # 2D DFT
            coeff = np.sum(inv_eps * np.outer(phase_x, phase_y).T) / (n * n)
            coeffs[(mx, my)] = coeff
    return coeffs

def _compute_bands_tm(coeffs, kx, ky, n_pw, n_bands=6):
    """
    Compute TM bands using plane-wave expansion.
    For TM polarization: (1/epsilon) * k^2 = omega^2
    The matrix equation: sum_m' (k - G_m) . (k - G_m') / epsilon_{m-m'} = omega^2 * delta_mm'
    where G_m = 2pi * (mx, my)
    """
    n_pw = 289  # evaluator uses this, but we use smaller for speed
    # Actually let's use the n_pw parameter
    indices = list(range(-n_pw, n_pw+1))
    pw_list = []
    for mx in indices:
        for my in indices:
            if mx*mx + my*my <= n_pw*n_pw:
                pw_list.append((mx, my))
    
    # Actually the evaluator uses 289 plane waves. Let's just use a fixed set.
    # For speed in our search, use a smaller basis.
    # Let's use all (mx,my) with mx^2+my^2 <= R^2 where R gives ~289 points
    # 289 = 17^2, so use a square of half-width 8: (2*8+1)^2 = 289
    # That's mx,my in [-8,8]
    
    pw_list = [(mx, my) for mx in range(-8, 9) for my in range(-8, 9)]
    N = len(pw_list)
    
    # Build matrix
    A = np.zeros((N, N), dtype=np.complex128)
    for i, (mx, my) in enumerate(pw_list):
        kx_i = kx + 2*np.pi*mx
        ky_i = ky + 2*np.pi*my
        for j, (mx_j, my_j) in enumerate(pw_list):
            dm = mx - mx_j
            dn = my - my_j
            if (dm, dn) not in coeffs:
                continue
            coeff = coeffs[(dm, dn)]
            kx_j = kx + 2*np.pi*mx_j
            ky_j = ky + 2*np.pi*my_j
            A[i, j] += coeff * (kx_i * kx_j + ky_i * ky_j)
    
    # Solve eigenvalue problem
    # A is Hermitian (should be, up to numerical error)
    A = (A + A.conj().T) / 2
    eigvals = np.linalg.eigvalsh(A)
    eigvals = np.sort(np.real(eigvals))
    
    # omega^2 = eigval, so omega = sqrt(eigval)
    # We need bands 4 and 5 (0-indexed: 3 and 4)
    if len(eigvals) < 6:
        return None
    omega = np.sqrt(np.maximum(eigvals, 0))
    return omega[:n_bands]

def _evaluate_gap(cell, n_pw_search=5):
    """
    Evaluate the gap-midgap ratio between bands 4 and 5.
    Uses a reduced plane-wave basis for speed.
    """
    coeffs = _fourier_coeffs(cell, n_pw_search)
    
    # Sample several k-points in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= kx <= ky, kx >= 0, ky >= 0, kx+ky <= pi
    # Actually for square lattice with lattice constant 1, BZ is [-pi,pi]^2
    # Irreducible wedge: 0 <= kx <= ky, 0 <= ky <= pi
    # Key points: Gamma(0,0), X(pi/2, 0), M(pi/2, pi/2), and line between them
    
    k_points = []
    # Gamma
    k_points.append((0, 0))
    # X
    k_points.append((np.pi, 0))
    # M
    k_points.append((np.pi, np.pi))
    # Points along Gamma-X
    for t in [0.25, 0.5, 0.75]:
        k_points.append((t*np.pi, 0))
    # Points along X-M
    for t in [0.25, 0.5, 0.75]:
        k_points.append((np.pi, t*np.pi))
    # Points along Gamma-M (diagonal)
    for t in [0.25, 0.5, 0.75]:
        k_points.append((t*np.pi, t*np.pi))
    
    max_omega4 = 0
    min_omega5 = float('inf')
    
    for kx, ky in k_points:
        bands = _compute_bands_tm(coeffs, kx, ky, 8, n_bands=6)
        if bands is None:
            continue
        # bands[3] is omega_4, bands[4] is omega_5 (1-indexed: 4th and 5th)
        if len(bands) >= 5:
            max_omega4 = max(max_omega4, bands[3])
            min_omega5 = min(min_omega5, bands[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    if midgap <= 0:
        return -1.0
    return 2 * gap / midgap

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell with C4v symmetry.
    Uses evolutionary search with plane-wave evaluation.
    """
    start_time = time.time()
    deadline = start_time + 200  # safety margin before 230s kill
    
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Initialize with the seed design (centered rod)
    best_cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < 0.2**2 else 0)
        best_cell.append(row)
    
    best_score = _evaluate_gap(best_cell, n_pw_search=5)
    
    # Evolutionary search on fundamental domain
    fd_best = _fd_from_cell(best_cell)
    
    # Population-based search
    pop_size = 20
    mutation_rate = 0.1
    
    # Create initial population
    population = []
    for _ in range(pop_size):
        fd = fd_best.copy()
        # Random mutations
        for _ in range(random.randint(2, 8)):
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            fd[i, j] = 1 - fd[i, j]
        population.append(fd)
    
    generation = 0
    max_generations = 50
    
    while time.time() < deadline - 30 and generation < max_generations:
        gen_start = time.time()
        
        # Evaluate population
        scored = []
        for fd in population:
            cell = _c4v_from_fd(fd).tolist()
            score = _evaluate_gap(cell, n_pw_search=5)
            scored.append((score, fd))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            fd_best = scored[0][1].copy()
        
        # Create next generation
        new_pop = []
        
        # Elitism: keep top 2
        for _, fd in scored[:2]:
            new_pop.append(fd.copy())
        
        # Fill rest with mutations of top candidates
        while len(new_pop) < pop_size:
            if time.time() > deadline - 30:
                break
            # Select parent (tournament)
            candidates = scored[:max(3, len(scored)//2)]
            parent_fd = random.choice(candidates)[1]
            
            # Mutate
            child_fd = parent_fd.copy()
            n_mut = random.randint(1, 4)
            for _ in range(n_mut):
                i = random.randint(0, 15)
                j = random.randint(0, 15)
                child_fd[i, j] = 1 - child_fd[i, j]
            
            new_pop.append(child_fd)
        
        population = new_pop
        generation += 1
    
    # Final refinement: try local search around best
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            fd_trial = fd_best.copy()
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            fd_trial[i, j] = 1 - fd_trial[i, j]
            cell_trial = _c4v_from_fd(fd_trial).tolist()
            score_trial = _evaluate_gap(cell_trial, n_pw_search=5)
            if score_trial > best_score:
                best_score = score_trial
                fd_best = fd_trial
    
    # Return best cell as nested list
    best_arr = _c4v_from_fd(fd_best)
    return best_arr.tolist()
