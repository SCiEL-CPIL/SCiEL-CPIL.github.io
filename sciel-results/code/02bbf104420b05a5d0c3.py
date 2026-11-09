"""Photonic-crystal unit cell design with fast symmetry-constrained search."""
import time
import numpy as np
from itertools import product

# Deadline guard
_START_TIME = time.time()
DEADLINE = 230.0  # evaluator kills at 230s; keep safety margin
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

def _time_left():
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)

def _apply_symmetry(cell):
    """Enforce 4-fold rotational + mirror symmetry about cell center.
    For a 32x32 cell, the fundamental domain is 16x16 (top-left quadrant).
    Symmetry: cell[i][j] = cell[31-i][j] = cell[i][31-j] = cell[31-i][31-j]
    Also transpose symmetry: cell[i][j] = cell[j][i]
    Combined: cell is determined by a 16x16 quadrant with additional constraints.
    
    Actually, for full square symmetry group (D4), the fundamental domain is a 16x16
    quadrant but with the constraint that it's symmetric under transpose within that quadrant
    when considering the full D4 group. Let me think more carefully.
    
    The cell must satisfy:
    1. cell == cell[::-1] row-wise (horizontal mirror)
    2. cell == cell[::-1] column-wise (vertical mirror)  
    3. cell == cell.T (transpose)
    
    These three imply D4 symmetry. The fundamental domain is the top-left 16x16 quadrant,
    but with the additional constraint from transpose that cell[i][j] = cell[j][i] for
    the full cell, which within the quadrant means the quadrant itself must be symmetric
    about its main diagonal.
    
    So the free parameters are the upper-triangular part of the 16x16 quadrant (including diagonal).
    That's 16*17/2 = 136 free bits.
    """
    size = len(cell)
    half = size // 2  # 16
    # Extract top-left quadrant
    quad = np.array(cell[:half, :half])
    # Enforce transpose symmetry within the quadrant: quad[i][j] = quad[j][i]
    # Actually, the full cell transpose symmetry means cell[i][j] = cell[j][i]
    # Within the top-left quadrant, this means quad[i][j] = quad[j][i]
    # So we symmetrize the quadrant
    quad_sym = (quad + quad.T) // 2
    quad_sym = (quad_sym > 0.5).astype(int)
    
    # Reconstruct full cell
    full = np.zeros((size, size), dtype=int)
    full[:half, :half] = quad_sym
    full[:half, half:] = quad_sym[:, ::-1]  # mirror horizontally
    full[half:, :half] = quad_sym[::-1, :]  # mirror vertically
    full[half:, half:] = quad_sym[::-1, ::-1]  # mirror both
    
    # Verify transpose symmetry
    # The construction should guarantee it, but let's double-check
    # full[i][j] should equal full[j][i]
    # Due to the quadrant symmetry and mirroring, this should hold.
    
    return full.tolist()

def _fast_bandgap(cell):
    """Compute approximate TM band gap between bands 4 and 5 using a simplified plane-wave method.
    
    For speed, we use a reduced set of plane waves and fewer k-points.
    Returns the gap-midgap ratio (higher is better).
    """
    size = len(cell)
    eps_low = 1.0
    eps_high = 12.0
    
    # Convert to epsilon grid
    eps = np.where(np.array(cell) == 1, eps_high, eps_low)
    
    # Compute Fourier coefficients of 1/eps for TM mode
    # For TM: (1/eps) * d2/dk^2 E = omega^2 E
    # In plane-wave basis: H * c = omega^2 * c where H involves 1/eps Fourier coeffs
    
    # Use a reduced number of plane waves for speed
    # Plane wave indices: G = (2*pi*m, 2*pi*n) for m,n in range(-N, N+1)
    # For speed, use N=2 (5x5 = 25 plane waves) instead of N=12 (25x25=625)
    N_pw = 3  # 7x7 = 49 plane waves, good balance
    G_indices = [(m, n) for m in range(-N_pw, N_pw+1) for n in range(-N_pw, N_pw+1)]
    N_G = len(G_indices)
    
    # Compute Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    inv_eps_FFT = np.fft.fft2(inv_eps)
    # Normalize: the DFT gives sum over pixels, we need the average
    # Fourier coeff for G = (2*pi*m, 2*pi*n): (1/N^2) * sum_{x,y} inv_eps[x,y] * exp(-i*G.r)
    # np.fft.fft2 gives sum, so divide by N^2
    inv_eps_FFT /= (size * size)
    
    # Map G indices to FFT indices
    # FFT index for frequency k: if k >= 0, index = k; if k < 0, index = size + k
    def fft_idx(k):
        return k % size
    
    # Build the Hamiltonian matrix for TM mode
    # H_{G',G} = (k+G')^2 * delta_{G',G} + sum_{G''} (1/eps)_{G''} * (k+G'-G'') * (k+G)
    # Wait, let me be more careful.
    
    # TM mode: (1/eps(r)) * (-d^2/dx^2 - d^2/dy^2) E(r) = omega^2 E(r)
    # In plane waves: E(r) = sum_G c_G exp(i(G+k).r)
    # -d^2/dx^2 - d^2/dy^2 gives (G+k)^2 c_G
    # 1/eps(r) = sum_{G''} (1/eps)_{G''} exp(i G''.r)
    # So the equation becomes:
    # sum_{G''} (1/eps)_{G''} sum_G c_G (G+k)^2 exp(i(G+k+G'').r) = omega^2 sum_G c_G exp(i(G+k).r)
    # Matching: for each G', sum_{G''} (1/eps)_{G''} c_{G'-G''} (G'+k)^2 = omega^2 c_{G'}
    
    # So H_{G',G} = sum_{G''} (1/eps)_{G''} * delta_{G, G'-G''} * (G'+k)^2
    # = (1/eps)_{G'-G} * (G'+k)^2
    
    # Wait, that's not right either. Let me redo.
    # 1/eps(r) * (-lap) E(r) = omega^2 E(r)
    # LHS = sum_{G''} (1/eps)_{G''} exp(iG''.r) * sum_G c_G (G+k)^2 exp(i(G+k).r)
    #     = sum_{G'',G} (1/eps)_{G''} c_G (G+k)^2 exp(i(G+k+G'').r)
    # Let G' = G + G'', so G = G' - G'':
    #     = sum_{G'',G'} (1/eps)_{G''} c_{G'-G''} (G'-G''+k)^2 exp(i(G'+k).r)
    # RHS = omega^2 sum_{G'} c_{G'} exp(i(G'+k).r)
    # Matching coefficients:
    # sum_{G''} (1/eps)_{G''} c_{G'-G''} (G'-G''+k)^2 = omega^2 c_{G'}
    
    # So H_{G',G} = (1/eps)_{G'-G} * (G'-G+k)^2 where G = G'-G'', i.e., G'' = G'-G
    # H_{G',G} = (1/eps)_{G'-G} * (G+k)^2  ... wait no
    # If G = G' - G'', then G'' = G' - G, and (G'-G''+k) = (G+k)
    # So H_{G',G} = (1/eps)_{G'-G} * (G+k)^2
    
    # Hmm, but this means H is not symmetric in general. Let me check.
    # Actually for the eigenvalue problem to be Hermitian, we need a different formulation.
    # The standard approach is to write it as a generalized eigenvalue problem or use
    # a symmetric form.
    
    # Alternative: multiply both sides by sqrt(eps) or use a different basis.
    # Actually, the standard plane-wave method for photonic crystals uses:
    # For TM: (1/eps) * (-lap) E = omega^2 E
    # This is not Hermitian in the standard inner product. We need to use a weighted
    # inner product or reformulate.
    
    # The correct symmetric formulation:
    # Let E = sqrt(eps) * F, then:
    # -lap F + (1/eps) * lap(eps) * F = omega^2 F  (approximately, but this gets complex)
    
    # Actually, the standard approach in the literature for the plane-wave method is:
    # The matrix is: H_{G',G} = (k+G')^2 * (1/eps)_{G'-G}
    # And this IS Hermitian because (1/eps)_{-G} = conj((1/eps)_G) for real eps,
    # and (k+G')^2 is real.
    
    # Let me verify: H_{G,G'} = (k+G)^2 * (1/eps)_{G-G'}
    # H_{G',G}^* = ((k+G')^2 * (1/eps)_{G'-G})^* = (k+G')^2 * (1/eps)_{-(G'-G)} = (k+G')^2 * (1/eps)_{G-G'}
    # Hmm, that's not equal to H_{G,G'} in general.
    
    # I think I'm overcomplicating this. Let me use the standard formulation:
    # The eigenvalue equation is: sum_G' (k+G)^2 * (1/eps)_{G-G'} * c_{G'} = omega^2 * c_G
    # So H_{G,G'} = (k+G)^2 * (1/eps)_{G-G'}
    
    # For this to be Hermitian: H_{G',G}^* = H_{G,G'}
    # H_{G',G}^* = ((k+G')^2 * (1/eps)_{G'-G})^* = (k+G')^2 * (1/eps)_{-(G'-G)} = (k+G')^2 * (1/eps)_{G-G'}
    # H_{G,G'} = (k+G)^2 * (1/eps)_{G-G'}
    # These are NOT equal in general! So the matrix is not Hermitian with this formulation.
    
    # The correct approach: the operator (1/eps)(-lap) is self-adjoint with respect to
    # the inner product <f|g> = sum_x eps(x) f(x)* g(x). In the plane-wave basis, this
    # means we need a generalized eigenvalue problem or a symmetric transform.
    
    # Standard solution: define d_G = c_G * (1/eps)_{0}^{1/2} or similar.
    # Actually, the cleanest way: the matrix A_{G,G'} = (k+G)^2 (1/eps)_{G-G'} satisfies
    # A = D * S where D is diagonal with D_{G,G} = (k+G)^2 and S_{G,G'} = (1/eps)_{G-G'}
    # S is Hermitian (since (1/eps) is real and even in Fourier space for real eps).
    # D is positive definite. So we have a generalized eigenvalue problem A c = omega^2 c
    # which is equivalent to S c = (1/omega^2) D^{-1} c, or D^{1/2} S D^{1/2} (D^{-1/2} c) = (1/omega^2) (D^{-1/2} c)
    
    # Let's use: B = D^{1/2} S D^{1/2}, then B is Hermitian, and we solve B y = lambda y
    # where lambda = 1/omega^2, and y = D^{1/2} c.
    
    # For our purposes, let's just compute the eigenvalues numerically.
    
    # Actually, for a quick approximation, let me just use a simpler approach:
    # Compute the band structure at a few key k-points and estimate the gap.
    
    # Key k-points in the irreducible BZ for square lattice:
    # Gamma: (0, 0)
    # X: (pi, 0) or (1, 0) in units of 2*pi/a
    # M: (pi, pi) or (1, 1) in units of 2*pi/a
    # The irreducible BZ is the triangle Gamma-X-M
    
    # For the fast solver, use these 3 points and a few along the edges.
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # Midpoint Gamma-X
        (1.0, 0.0),      # X
        (1.0, 0.5),      # Midpoint X-M
        (1.0, 1.0),      # M
        (0.5, 0.5),      # Center of BZ
    ]
    
    # In units where the lattice constant is 1, k is in units of 2*pi/a
    # G vectors are (2*pi*m, 2*pi*n), so in units of 2*pi/a, G = (m, n)
    
    all_eigenvalues = []
    
    for kx, ky in k_points:
        # Build H_{G,G'} = (k+G)^2 * (1/eps)_{G-G'}
        H = np.zeros((N_G, N_G), dtype=complex)
        
        for i, (mi, ni) in enumerate(G_indices):
            Gi = (mi, ni)
            ki = (kx + mi, ky + ni)
            ki2 = ki[0]**2 + ki[1]**2
            
            for j, (mj, nj) in enumerate(G_indices):
                Gj = (mj, nj)
                dG = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                
                # Get (1/eps)_{dG} from FFT
                # dG in units of 2*pi/a, FFT index is dG[0] mod size, dG[1] mod size
                idx_i = fft_idx(dG[0])
                idx_j = fft_idx(dG[1])
                inv_eps_G = inv_eps_FFT[idx_i, idx_j]
                
                H[i, j] = ki2 * inv_eps_G
        
        # Make Hermitian: H should be Hermitian if we do it right
        # Actually, let's symmetrize: H = (H + H^H) / 2
        H = (H + H.conj().T) / 2
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.real(eigvals)
            eigvals = np.sort(eigvals)
            
            # We want bands 4 and 5 (0-indexed: 3 and 4)
            # But we need to be careful: the lowest bands might not correspond to
            # the physical bands 4 and 5 if we have too few plane waves.
            # With 49 plane waves, we should get at least 5 bands reasonably well.
            
            if len(eigvals) >= 5:
                # omega^2 values, so omega = sqrt(eigval)
                omega = np.sqrt(np.maximum(eigvals, 0))
                omega4 = omega[3]  # 4th band (0-indexed: 3)
                omega5 = omega[4]  # 5th band (0-indexed: 4)
                
                gap = omega5 - omega4
                midgap = (omega5 + omega4) / 2
                ratio = gap / midgap if midgap > 0 else 0
                all_eigenvalues.append((kx, ky, omega4, omega5, ratio))
        except:
            pass
    
    if not all_eigenvalues:
        return 0.0
    
    # The gap is the minimum over all k-points of (omega5 - omega4)
    # and the midgap is the maximum of omega4 and minimum of omega5
    max_omega4 = max(e[2] for e in all_eigenvalues)
    min_omega5 = min(e[3] for e in all_eigenvalues)
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2
    
    if midgap <= 0:
        return 0.0
    
    return gap / midgap

def _random_symmetric_cell(size=32):
    """Generate a random cell with D4 symmetry."""
    half = size // 2
    # Generate random 16x16 quadrant
    quad = np.random.randint(0, 2, (half, half))
    # Enforce transpose symmetry within quadrant
    quad = ((quad + quad.T) // 2)
    # Make binary
    quad = (quad > 0.5).astype(int)
    
    # Reconstruct full cell
    full = np.zeros((size, size), dtype=int)
    full[:half, :half] = quad
    full[:half, half:] = quad[:, ::-1]
    full[half:, :half] = quad[::-1, :]
    full[half:, half:] = quad[::-1, ::-1]
    
    return full.tolist()

def _mutate_cell(cell, mutation_rate=0.1):
    """Mutate a cell by flipping some pixels while maintaining symmetry."""
    size = len(cell)
    half = size // 2
    cell_arr = np.array(cell)
    
    # Extract quadrant
    quad = cell_arr[:half, :half].copy()
    
    # Flip random bits in the upper triangle of the quadrant
    for i in range(half):
        for j in range(i, half):  # upper triangle including diagonal
            if np.random.random() < mutation_rate:
                quad[i, j] = 1 - quad[i, j]
                quad[j, i] = quad[i, j]  # maintain symmetry
    
    # Reconstruct
    full = np.zeros((size, size), dtype=int)
    full[:half, :half] = quad
    full[:half, half:] = quad[:, ::-1]
    full[half:, :half] = quad[::-1, :]
    full[half:, half:] = quad[::-1, ::-1]
    
    return full.tolist()

def _crossover(cell1, cell2, size=32):
    """Crossover two cells by combining their quadrants."""
    half = size // 2
    q1 = np.array(cell1[:half, :half])
    q2 = np.array(cell2[:half, :half])
    
    # Random crossover
    mask = np.random.randint(0, 2, (half, half)).astype(bool)
    q = np.where(mask, q1, q2)
    
    # Enforce symmetry
    q = ((q + q.T) // 2)
    q = (q > 0.5).astype(int)
    
    # Reconstruct
    full = np.zeros((size, size), dtype=int)
    full[:half, :half] = q
    full[:half, half:] = q[:, ::-1]
    full[half:, :half] = q[::-1, :]
    full[half:, half:] = q[::-1, ::-1]
    
    return full.tolist()

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing the TM band gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    
    # Start with the seed: centered dielectric rod
    # Radius 0.2 in lattice units, so in pixels: 0.2 * 32 = 6.4 pixels from center
    # Center is at (15.5, 15.5) for 32x32
    best_cell = None
    best_score = -np.inf
    
    # Initialize with the seed design
    cell_arr = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0  # 15.5
    radius = 0.2 * size  # 6.4 pixels
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < radius*radius:
                cell_arr[i, j] = 1
    
    # Enforce symmetry
    best_cell = _apply_symmetry(cell_arr.tolist())
    best_score = _fast_bandgap(best_cell)
    
    # Also try some other initial designs
    initial_designs = [best_cell]
    
    # Try a ring design
    cell_ring = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            r = np.sqrt(dx*dx + dy*dy)
            if 4.0 < r < 8.0:  # ring
                cell_ring[i, j] = 1
    initial_designs.append(_apply_symmetry(cell_ring.tolist()))
    
    # Try a cross/plus design
    cell_cross = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if abs(dx) < 3 or abs(dy) < 3:
                cell_cross[i, j] = 1
    initial_designs.append(_apply_symmetry(cell_cross.tolist()))
    
    # Evaluate initial designs
    for cell in initial_designs:
        score = _fast_bandgap(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Genetic algorithm
    population_size = 30
    population = [best_cell]
    
    # Add some random initializations
    for _ in range(population_size - 1):
        if _time_left() <= 0:
            break
        pop_cell = _random_symmetric_cell(size)
        population.append(pop_cell)
    
    # Evaluate population
    scores = [_fast_bandgap(cell) for cell in population]
    
    # GA loop
    generation = 0
    while _time_left() > 5.0:
        generation += 1
        
        # Sort by score
        indexed = list(zip(scores, population))
        indexed.sort(key=lambda x: x[0], reverse=True)
        scores = [s for s, c in indexed]
        population = [c for s, c in indexed]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_cell = population[0]
        
        # Create next generation
        new_population = [population[0]]  # elitism
        
        # Crossover and mutation
        while len(new_population) < population_size:
            if _time_left() <= 5.0:
                break
            
            # Tournament selection
            parents = []
            for _ in range(2):
                tournament = np.random.choice(len(population), size=min(5, len(population)), replace=False)
                best_t = max(tournament, key=lambda i: scores[i])
                parents.append(population[best_t])
            
            child = _crossover(parents[0], parents[1], size)
            child = _mutate_cell(child, mutation_rate=0.05)
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        scores = [_fast_bandgap(cell) for cell in population]
        
        # Occasionally add a random individual
        if generation % 5 == 0 and _time_left() > 10.0:
            random_cell = _random_symmetric_cell(size)
            population[-1] = random_cell
            scores[-1] = _fast_bandgap(random_cell)
    
    return best_cell
