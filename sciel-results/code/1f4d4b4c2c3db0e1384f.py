"""Photonic crystal design: maximize TM band-4/5 gap via direct pixel optimization."""
import numpy as np
import time

def design(problem):
    size = problem["pixels"]  # 32
    eps_high = problem["eps_high"]  # 12
    eps_low = problem["eps_low"]  # 1
    deadline = time.time() + 210  # safety margin before 230s kill

    # Precompute symmetry orbits for 32x32 cell (invariant under D4: rotations + reflections)
    # A pixel (i,j) maps to:
    # (i,j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j)  [reflections]
    # (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i) [transpose + reflections]
    # Full D4 orbit of (i,j):
    # (i,j), (i,N-1-j), (N-1-i,j), (N-1-i,N-1-j),
    # (j,i), (j,N-1-i), (N-1-j,i), (N-1-j,N-1-i)
    N = size

    # Build unique orbits
    orbits = []
    visited = set()
    for i in range(N):
        for j in range(N):
            if (i, j) in visited:
                continue
            orbit = set()
            coords = [(i,j), (i,N-1-j), (N-1-i,j), (N-1-i,N-1-j),
                       (j,i), (j,N-1-i), (N-1-j,i), (N-1-j,N-1-i)]
            for c in coords:
                orbit.add(c)
                visited.add(c)
            orbits.append(list(orbit))

    # Fast band-structure evaluator using plane-wave expansion
    # For TM polarization in 2D: the eigenvalue equation involves 1/eps
    # We use a small number of plane waves for speed (7x7 = 49 basis functions)
    # The key insight: we need to compute the gap between bands 4 and 5
    
    # Fourier coefficients of 1/eps(x,y) for the pixel map
    # For a pixelated structure, eps(x,y) = eps_low + (eps_high - eps_low) * indicator(x,y)
    # 1/eps(x,y) = 1/eps_low + (1/eps_high - 1/eps_low) * indicator(x,y)
    
    # Plane wave expansion: 1/eps(x,y) = sum_G eps_G * exp(iG.x)
    # For pixelated: eps_G = delta_G0 + (1/eps_high - 1/eps_low) * (1/N^2) * sum_{pixels} f(r) exp(-iG.r)
    
    # We'll use a 7x7 grid of G-vectors for speed
    G_cutoff = 3  # |Gx|, |Gy| <= 3, giving 7x7 = 49 plane waves
    
    # G vectors in units of 2*pi/a
    G_vecs = []
    for gx in range(-G_cutoff, G_cutoff+1):
        for gy in range(-G_cutoff, G_cutoff+1):
            G_vecs.append((gx, gy))
    nG = len(G_vecs)
    
    # K-points along the irreducible BZ boundary and interior
    # Irreducible BZ for square lattice: triangle with vertices (0,0), (pi,0), (pi,pi)
    # In units of 2*pi/a: (0,0), (0.5,0), (0.5,0.5)
    # We'll sample points along the path and interior
    k_points = []
    # Along X: (0,0) to (0.5, 0)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0))
    # Along M-X: (0.5, 0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((0.5, t))
    # Along Gamma-M diagonal: (0,0) to (0.5, 0.5)
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, t))
    # Some interior points
    k_points.append((0.25, 0.25))
    k_points.append((0.4, 0.1))
    k_points.append((0.1, 0.4))
    k_points.append((0.3, 0.3))
    k_points.append((0.45, 0.45))
    k_points.append((0.2, 0.1))
    
    def compute_fourier_coefficients(cell):
        """Compute eps_G for all G vectors in the cutoff."""
        # cell is N x N, 1 = dielectric, 0 = air
        # 1/eps = 1/eps_low + (1/eps_high - 1/eps_low) * f(r)
        # where f(r) = 1 if dielectric, 0 if air
        # Fourier: eps_G = delta(G,0) + (1/eps_high - 1/eps_low) * F_G
        # F_G = (1/N^2) * sum_{i,j} f(i,j) * exp(-i*(Gx*i + Gy*j)*2*pi/N)
        # In our units, G is in reciprocal lattice units, so the phase is exp(-i*pi*2*(Gx*i/N + Gy*j/N))
        # Actually with G in units of 2*pi/a, and pixel positions at (i+0.5)/N * a:
        # phase = exp(-i * 2*pi * (Gx*(i+0.5)/N + Gy*(j+0.5)/N))
        
        coeff = np.zeros(nG, dtype=complex)
        delta_val = 1.0/eps_low
        diff_val = 1.0/eps_high - 1.0/eps_low
        
        for idx, (gx, gy) in enumerate(G_vecs):
            if gx == 0 and gy == 0:
                # DC component
                frac = np.sum(cell) / (N*N)
                coeff[idx] = delta_val + diff_val * frac
            else:
                # Compute sum
                s = 0.0 + 0.0j
                for i in range(N):
                    for j in range(N):
                        if cell[i][j]:
                            phase = -2.0 * np.pi * (gx*(i+0.5)/N + gy*(j+0.5)/N)
                            s += np.exp(1j * phase)
                coeff[idx] = delta_val + diff_val * s / (N*N)
        
        return coeff
    
    def compute_bands_fast(coeff, kx, ky):
        """Compute eigenvalues (omega^2) using plane-wave method.
        For TM: the matrix element is (k+G)^2 * eps_G' where eps_G' connects G1 to G2
        Actually for TM in 2D: the equation is sum_G' eps_{G-G'} (k+G')^2 c_{G'} = omega^2 c_G
        where eps_{G-G'} is the Fourier coefficient of 1/eps.
        """
        # Build the matrix
        M = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            Gi = G_vecs[i]
            kGi = (kx + Gi[0], ky + Gi[1])
            for j in range(nG):
                Gj = G_vecs[j]
                kGj = (kx + Gj[0], ky + Gj[1])
                dG = (Gi[0] - Gj[0], Gi[1] - Gj[1])
                # Find index of dG in G_vecs
                # dG must be in our set for the matrix element to be non-zero
                try:
                    dG_idx = G_vecs.index(dG)
                except ValueError:
                    continue
                M[i, j] = coeff[dG_idx] * (kGj[0]**2 + kGj[1]**2)
        
        # Solve eigenvalue problem
        # M is Hermitian (should be), so use eigh
        eigvals = np.linalg.eigvalsh(M)
        # omega^2 = eigvals, so omega = sqrt(eigvals)
        # We want the 4th and 5th bands (1-indexed), so indices 3 and 4
        if len(eigvals) >= 5:
            omega4 = np.sqrt(max(eigvals[3], 0))
            omega5 = np.sqrt(max(eigvals[4], 0))
            return omega4, omega5
        return 0, 0
    
    def evaluate_gap(cell):
        """Compute the gap-midgap ratio for the given cell."""
        coeff = compute_fourier_coefficients(cell)
        max_omega4 = 0
        min_omega5 = float('inf')
        for kx, ky in k_points:
            o4, o5 = compute_bands_fast(coeff, kx, ky)
            if o4 > max_omega4:
                max_omega4 = o4
            if o5 < min_omega5:
                min_omega5 = o5
        if min_omega5 == float('inf') or (max_omega4 + min_omega5) == 0:
            return 0.0
        gap = min_omega5 - max_omega4
        midgap = max_omega4 + min_omega5
        return 2.0 * gap / midgap
    
    # Initial design: centered dielectric disk (the seed)
    cell = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            if x*x + y*y < 0.2**2:
                cell[i, j] = 1
    
    best_score = evaluate_gap(cell.tolist())
    best_cell = cell.copy()
    
    # Simulated annealing with symmetry constraints
    # We perturb orbits: flip all pixels in a random orbit
    rng = np.random.default_rng(42)
    
    T = 0.1  # initial temperature
    cooling = 0.995
    T_min = 0.001
    
    start_time = time.time()
    iters = 0
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Pick a random orbit to flip
        orbit_idx = rng.integers(len(orbits))
        orbit = orbits[orbit_idx]
        
        # Save old state
        old_vals = [cell[i, j] for i, j in orbit]
        
        # Flip
        for i, j in orbit:
            cell[i, j] = 1 - cell[i, j]
        
        # Evaluate
        new_score = evaluate_gap(cell.tolist())
        
        # Accept/reject
        delta = new_score - best_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            best_score = new_score
            best_cell = cell.copy()
        else:
            # Revert
            for (i, j), v in zip(orbit, old_vals):
                cell[i, j] = v
        
        T *= cooling
        iters += 1
        
        # Check time every 100 iterations
        if iters % 100 == 0 and time.time() > deadline:
            break
    
    # Return the best cell found
    return best_cell.tolist()
