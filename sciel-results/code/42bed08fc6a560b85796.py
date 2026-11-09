"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def _symmetrize(cell):
    """Enforce square symmetry: 180-degree rotation and transpose invariance."""
    n = len(cell)
    cell = np.array(cell, dtype=np.float64)
    # Average over the four symmetry operations: identity, 180-rot, transpose, both
    rot180 = cell[::-1, ::-1]
    trans = cell.T
    rot180_trans = rot180.T
    cell = (cell + rot180 + trans + rot180_trans) / 4.0
    return (cell > 0.5).astype(int)

def _fourier_coefficients(cell, n_pw=17):
    """Compute Fourier coefficients of the inverse permittivity for TM mode."""
    n = len(cell)
    eps = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    # Compute 2D FFT of inv_eps, normalized
    # The plane wave expansion: 1/eps(G) = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-i G . r)
    # We need coefficients for G in the plane wave set
    coeffs = np.zeros((n_pw, n_pw), dtype=np.complex128)
    
    # Plane wave indices: from -(n_pw-1)//2 to (n_pw-1)//2
    half = (n_pw - 1) // 2
    G_indices = np.arange(-half, half + 1)
    
    for gi in range(n_pw):
        for gj in range(n_pw):
            Gx = G_indices[gi]
            Gy = G_indices[gj]
            # Compute (1/N^2) sum_{i,j} inv_eps[i,j] exp(-2*pi*i*(Gx*(i+0.5)/N + Gy*(j+0.5)/N))
            # This is essentially the DFT at frequency (Gx, Gy)
            # Using numpy FFT
            phase_x = np.exp(-2j * np.pi * Gx * np.arange(n) / n)
            phase_y = np.exp(-2j * np.pi * Gy * np.arange(n) / n)
            # Multiply by 0.5 pixel shift factor
            shift_x = np.exp(-1j * np.pi * Gx / n)
            shift_y = np.exp(-1j * np.pi * Gy / n)
            val = np.sum(inv_eps * np.outer(phase_x, phase_y)) / (n * n)
            val *= shift_x * shift_y
            coeffs[gi, gj] = val
    
    return coeffs

def _pw_solver_fast(cell, n_pw=13, k_points=None):
    """Fast approximate plane-wave solver for TM bands. Returns list of band energies at each k-point."""
    n = len(cell)
    half = (n_pw - 1) // 2
    G_indices = np.arange(-half, half + 1)
    
    # Precompute Fourier coefficients
    coeffs = _fourier_coefficients(cell, n_pw)
    
    # Build the matrix for a given k-point
    # For TM: the eigenvalue problem is sum_G' 1/eps(G-G') * |k+G|^2 * c(G') = omega^2 * c(G)
    
    results = []
    
    if k_points is None:
        # Use a few key k-points in the irreducible BZ
        k_points = [(0, 0), (0.5, 0), (0.5, 0.5), (0.25, 0.25), (0.3, 0.1), (0.4, 0.2)]
    
    for kx, ky in k_points:
        # Build matrix A where A[G, G'] = 1/eps(G-G') * |k+G'|^2
        mat = np.zeros((n_pw * n_pw, n_pw * n_pw), dtype=np.complex128)
        
        for i in range(n_pw):
            for j in range(n_pw):
                idx = i * n_pw + j
                Gx = G_indices[i]
                Gy = G_indices[j]
                
                for ii in range(n_pw):
                    for jj in range(n_pw):
                        idx2 = ii * n_pw + jj
                        Gx2 = G_indices[ii]
                        Gy2 = G_indices[jj]
                        
                        # G - G'
                        dGx = Gx - Gx2
                        dGy = Gy - Gy2
                        
                        # Check if dG is in our coefficient set
                        if -half <= dGx <= half and -half <= dGy <= half:
                            coeff = coeffs[dGx + half, dGy + half]
                            kx_eff = kx + Gx2
                            ky_eff = ky + Gy2
                            k_sq = kx_eff * kx_eff + ky_eff * ky_eff
                            mat[idx, idx2] = coeff * k_sq
        
        # Solve eigenvalue problem
        # The matrix should be Hermitian for real coefficients
        mat_real = mat.real
        # Symmetrize
        mat_real = (mat_real + mat_real.T) / 2
        
        try:
            eigenvalues = np.linalg.eigvalsh(mat_real)
            # Take the smallest n_pw eigenvalues
            results.append(eigenvalues[:n_pw])
        except:
            results.append(np.zeros(n_pw))
    
    return results

def _score_cell(cell, n_pw=13):
    """Score a cell by the gap between bands 4 and 5 (TM)."""
    bands = _pw_solver_fast(cell, n_pw)
    if not bands:
        return -1.0
    
    # Find max of band 4 (index 3) and min of band 5 (index 4)
    band4_max = max(b[3] for b in bands if len(b) > 3)
    band5_min = min(b[4] for b in bands if len(b) > 4)
    
    if band5_min <= band4_max:
        return -1.0
    
    gap = band5_min - band4_max
    midgap = (band5_min + band4_max) / 2.0
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _init_population(pop_size, n=32):
    """Initialize population with random symmetric cells."""
    pop = []
    # The symmetry-constrained degrees of freedom: for a 32x32 cell with 180-rot and transpose symmetry,
    # we need cell[i][j] == cell[31-i][31-j] == cell[j][i] == cell[31-j][31-i]
    # So we only need to specify the upper-left 16x16 quadrant (indices 0-15 for both i and j)
    # Actually, let's think more carefully:
    # Symmetry 1: cell[i][j] = cell[31-i][31-j] (180 deg rotation)
    # Symmetry 2: cell[i][j] = cell[j][i] (transpose)
    # Combined: cell[i][j] = cell[31-i][31-j] = cell[j][i] = cell[31-j][31-i]
    # The fundamental domain is the set of (i,j) where i <= 15 and j <= 15 and i <= j
    # No wait, that's not quite right. Let me just generate random and symmetrize.
    
    for _ in range(pop_size):
        # Random cell
        cell = np.random.randint(0, 2, (n, n))
        # Symmetrize
        cell = _symmetrize(cell)
        pop.append(cell)
    
    return pop

def _crossover(parent1, parent2):
    """Uniform crossover on the symmetry-constrained representation."""
    n = len(parent1)
    child = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if np.random.rand() < 0.5:
                child[i, j] = parent1[i, j]
            else:
                child[i, j] = parent2[i, j]
    return _symmetrize(child)

def _mutate(cell, rate=0.05):
    """Bit-flip mutation."""
    n = len(cell)
    cell = cell.copy()
    # Only mutate the fundamental domain to maintain symmetry
    for i in range(n // 2):
        for j in range(n // 2):
            if np.random.rand() < rate:
                cell[i, j] = 1 - cell[i, j]
    return _symmetrize(cell)

def _ga_search(deadline, pop_size=40, n_pw=11):
    """Genetic algorithm search."""
    start_time = time.time()
    pop = _init_population(pop_size)
    
    # Evaluate initial population
    scores = []
    for cell in pop:
        if time.time() > deadline - 30:
            break
        score = _score_cell(cell, n_pw)
        scores.append(score)
    
    best_cell = pop[0]
    best_score = max(scores) if scores else -1.0
    if scores:
        best_idx = scores.index(max(scores))
        best_cell = pop[best_idx]
    
    # GA loop
    gen = 0
    while time.time() < deadline - 30:
        gen += 1
        # Evaluate all
        scores = []
        for cell in pop:
            score = _score_cell(cell, n_pw)
            scores.append(score)
        
        # Update best
        if scores:
            max_score = max(scores)
            if max_score > best_score:
                best_score = max_score
                best_idx = scores.index(max_score)
                best_cell = pop[best_idx]
        
        # Selection: tournament
        new_pop = []
        # Keep best individual
        best_idx = scores.index(max(scores)) if scores else 0
        new_pop.append(pop[best_idx].copy())
        
        while len(new_pop) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] >= scores[idx2]:
                p1 = pop[idx1]
            else:
                p1 = pop[idx2]
            
            idx1 = np.random.randint(0, pop_size)
            idx2 = np.random.randint(0, pop_size)
            if scores[idx1] >= scores[idx2]:
                p2 = pop[idx1]
            else:
                p2 = pop[idx2]
            
            child = _crossover(p1, p2)
            child = _mutate(child, rate=0.03)
            new_pop.append(child)
        
        pop = new_pop
    
    return best_cell, best_score

def _refine_cell(cell, deadline, n_pw=13):
    """Local refinement: try flipping individual pixels in the fundamental domain."""
    n = len(cell)
    current = cell.copy()
    current_score = _score_cell(current, n_pw)
    
    # Try flipping each pixel in the upper-left quadrant
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(n // 2):
            if time.time() > deadline - 20:
                break
            for j in range(n // 2):
                if time.time() > deadline - 20:
                    break
                # Try flipping
                test = current.copy()
                test[i, j] = 1 - test[i, j]
                test = _symmetrize(test)
                score = _score_cell(test, n_pw)
                if score > current_score + 1e-6:
                    current = test
                    current_score = score
                    improved = True
    
    return current, current_score

def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band 4-5 gap."""
    start_time = time.time()
    deadline = start_time + 100.0  # Leave margin before the 120s limit
    
    size = problem["pixels"]
    
    # Start with a good initial guess: dielectric background with air holes
    # For TM gaps, a common good structure is ~15% air fill fraction
    # Create a pattern with air holes at specific positions
    cell = np.ones((size, size), dtype=int)  # Start with all dielectric
    
    # Add air holes in a pattern that respects symmetry
    # Place holes at positions that break the degeneracy
    # For a 32x32 cell, center is at (15.5, 15.5)
    # Symmetry: cell[i][j] = cell[31-i][31-j] = cell[j][i] = cell[31-j][31-i]
    
    # Create a pattern with air holes
    # The fundamental domain is i in [0,15], j in [0,15], i <= j (due to transpose symmetry)
    # Plus the diagonal i=j and the anti-diagonal considerations
    
    # Let's use a known good starting pattern: periodic air holes
    # Place air holes at (4,4), (8,8), (12,12), etc. in the fundamental domain
    hole_positions = [
        (4, 4), (4, 8), (4, 12),
        (8, 4), (8, 8), (8, 12),
        (12, 4), (12, 8), (12, 12),
    ]
    for (i, j) in hole_positions:
        # Set the symmetric equivalents
        cell[i, j] = 0
        cell[31-i, 31-j] = 0
        cell[j, i] = 0
        cell[31-j, 31-i] = 0
    
    # Make holes slightly larger (2x2 or 3x3)
    for (i, j) in hole_positions:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii, jj = i + di, j + dj
                if 0 <= ii < size and 0 <= jj < size:
                    cell[ii, jj] = 0
                    cell[31-ii, 31-jj] = 0
                    cell[jj, ii] = 0
                    cell[31-jj, 31-ii] = 0
    
    cell = _symmetrize(cell)
    
    # Evaluate initial
    initial_score = _score_cell(cell, n_pw=11)
    best_cell = cell
    best_score = initial_score
    
    # Run GA search
    if time.time() < deadline - 40:
        ga_cell, ga_score = _ga_search(deadline, pop_size=30, n_pw=11)
        if ga_score > best_score:
            best_cell = ga_cell
            best_score = ga_score
    
    # Local refinement
    if time.time() < deadline - 15:
        refined_cell, refined_score = _refine_cell(best_cell, deadline, n_pw=11)
        if refined_score > best_score:
            best_cell = refined_cell
            best_score = refined_score
    
    # Final evaluation with higher precision
    final_score = _score_cell(best_cell, n_pw=13)
    
    # Convert to nested list
    result = best_cell.tolist()
    
    return result
# EVOLVE-BLOCK-END
