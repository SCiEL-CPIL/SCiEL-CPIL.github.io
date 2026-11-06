import numpy as np
import time

def _apply_symmetry(cell):
    """Apply 4-fold symmetry: cell must equal its transpose, and be symmetric
    about both center axes (row-reverse and col-reverse)."""
    size = cell.shape[0]
    # Enforce: cell[i,j] == cell[j,i] (transpose)
    #         cell[i,j] == cell[size-1-i, size-1-j] (180 rot)
    #         cell[i,j] == cell[size-1-i, j] (horizontal flip)
    #         cell[i,j] == cell[i, size-1-j] (vertical flip)
    # Combined: 8 equivalent positions under D4
    result = np.zeros_like(cell)
    for i in range(size):
        for j in range(size):
            # Get all 8 symmetric positions
            positions = [
                (i, j),
                (j, i),                    # transpose
                (size-1-i, size-1-j),      # 180 rot
                (size-1-i, j),             # horizontal flip
                (i, size-1-j),             # vertical flip
                (size-1-j, size-1-i),      # transpose of 180
                (size-1-j, i),             # transpose of horiz flip
                (j, size-1-i),             # transpose of vert flip
            ]
            # Take majority vote (or just first)
            vals = [cell[p[0], p[1]] for p in positions]
            result[i, j] = 1 if sum(vals) >= 4 else 0
    return result

def _compute_gap(cell, n_pw=17, n_kpts=12):
    """Compute TM band gap between bands 4 and 5 using plane-wave method.
    
    Args:
        cell: 32x32 binary array (1=dielectric eps=12, 0=air eps=1)
        n_pw: number of plane waves in each direction (total = n_pw^2)
        n_kpts: number of k-points to sample
    
    Returns:
        gap_ratio: 2*(min_omega_5 - max_omega_4)/(min_omega_5 + max_omega_4)
    """
    size = cell.shape[0]
    eps_low = 1.0
    eps_high = 12.0
    
    # Compute Fourier coefficients of epsilon(r)
    # epsilon(r) = sum_m epsilon_m * exp(i*m*r)
    # For a 32x32 pixel grid, m ranges over integers
    # We need m components up to some cutoff
    
    # The plane wave method uses:
    # (k + G)^2 * eps^{-1}_{m} * E_{k+G'-m} = omega^2 * E_{k+G'}
    
    # First, compute epsilon^{-1} in Fourier space
    # For a binary dielectric, eps^{-1}(r) is also piecewise constant
    
    # Compute 1/eps in real space
    inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
    
    # Compute Fourier coefficients of 1/eps
    # We need coefficients for G vectors within a certain radius
    # For 289 plane waves (17x17), we need G components from -8 to 8
    
    max_G = n_pw // 2  # 8 for n_pw=17
    
    # Compute Fourier coefficients: f_m = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i*m.r_{i,j})
    # where r_{i,j} = ((i+0.5)/size - 0.5, (j+0.5)/size - 0.5)
    
    coords = np.arange(size)
    x_coords = (coords + 0.5) / size - 0.5
    y_coords = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Fourier coefficients
    # f[mx, my] = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(mx*X[i,j] + my*Y[i,j]))
    
    # Use FFT for efficiency
    # Pad to get the right frequency components
    # Actually, let's just compute directly for the needed range
    
    fourier_coeffs = np.zeros((2*max_G+1, 2*max_G+1), dtype=complex)
    
    for mx_idx in range(2*max_G+1):
        mx = mx_idx - max_G
        for my_idx in range(2*max_G+1):
            my = my_idx - max_G
            # Compute the coefficient
            phase = np.exp(-2j * np.pi * (mx * X + my * Y))
            fourier_coeffs[mx_idx, my_idx] = np.mean(inv_eps * phase)
    
    # Define k-points in the irreducible Brillouin zone
    # BZ: -pi <= kx,ky <= pi (for lattice constant a=1)
    # Irreducible: 0 <= ky <= kx <= pi/2 (by symmetry)
    
    # Sample points along the boundary and interior
    k_points = []
    
    # Along Gamma-X: kx from 0 to pi/2, ky=0
    for i in range(n_kpts):
        kx = (i / (n_kpts - 1)) * (np.pi / 2)
        k_points.append((kx, 0.0))
    
    # Along X-M: kx=pi/2, ky from 0 to pi/2
    for i in range(1, n_kpts):
        ky = (i / (n_kpts - 1)) * (np.pi / 2)
        k_points.append((np.pi / 2, ky))
    
    # Along Gamma-M: kx=ky, from 0 to pi/2
    for i in range(1, n_kpts - 1):
        kx = (i / (n_kpts - 1)) * (np.pi / 2)
        k_points.append((kx, kx))
    
    # Remove duplicates and sort
    k_points = list(set(k_points))
    k_points.sort(key=lambda k: (k[0], k[1]))
    
    # For each k-point, solve the eigenvalue problem
    # Build the matrix: (k+G)^2 * sum_m eps^{-1}_m * delta(G'-G-m)
    
    # G vectors: (mx, my) * 2*pi for mx,my in [-max_G, max_G]
    G_vectors = []
    for mx in range(-max_G, max_G+1):
        for my in range(-max_G, max_G+1):
            G_vectors.append((mx * 2 * np.pi, my * 2 * np.pi))
    
    n_G = len(G_vectors)
    
    # Map from (mx, my) to index
    G_to_idx = {}
    for idx, (gx, gy) in enumerate(G_vectors):
        mx = int(round(gx / (2 * np.pi)))
        my = int(round(gy / (2 * np.pi)))
        G_to_idx[(mx, my)] = idx
    
    # Precompute (k+G)^2 for each k-point
    max_omega_4 = 0.0
    min_omega_5 = np.inf
    
    for kx, ky in k_points:
        # Build the generalized eigenvalue problem
        # A * E = omega^2 * B * E
        # where A_{G,G'} = (k+G)^2 * sum_m eps^{-1}_m * delta(G'-G-m)
        # B = identity (since we're using the 1/eps formulation)
        
        # Actually, for TM polarization:
        # (k+G)^2 * eps^{-1}(r) * E(r) = omega^2 * E(r)
        # In Fourier space:
        # (k+G)^2 * sum_m eps^{-1}_m * E_{G'-m} = omega^2 * E_{G'}
        
        # So the matrix is:
        # M_{G,G'} = (k+G)^2 * eps^{-1}_{G-G'}
        
        M = np.zeros((n_G, n_G), dtype=complex)
        
        for i, (gx, gy) in enumerate(G_vectors):
            kx_total = kx + gx
            ky_total = ky + gy
            k_sq = kx_total**2 + ky_total**2
            
            for j, (gx2, gy2) in enumerate(G_vectors):
                # G - G' = (gx - gx2, gy - gy2)
                dmx = int(round((gx - gx2) / (2 * np.pi)))
                dmy = int(round((gy - gy2) / (2 * np.pi)))
                
                if abs(dmx) <= max_G and abs(dmy) <= max_G:
                    # Get eps^{-1}_{dmx, dmy}
                    eps_inv_coeff = fourier_coeffs[dmx + max_G, dmy + max_G]
                    M[i, j] = k_sq * eps_inv_coeff
        
        # Solve eigenvalue problem
        try:
            eigenvalues = np.linalg.eigvalsh(M)
            eigenvalues = np.sort(np.real(eigenvalues))
            
            # Filter out negative or zero eigenvalues (unphysical)
            positive_eigs = eigenvalues[eigenvalues > 1e-10]
            
            if len(positive_eigs) >= 5:
                omega_4 = np.sqrt(positive_eigs[3])  # 4th band (0-indexed: 3)
                omega_5 = np.sqrt(positive_eigs[4])  # 5th band (0-indexed: 4)
                
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        except:
            continue
    
    if min_omega_5 == np.inf or max_omega_4 == 0:
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2
    
    if midgap <= 0:
        return -1.0
    
    return 2 * gap / midgap

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell using local search
    with a fast plane-wave band solver.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    
    # Start with a reasonable initial design: dielectric rods in air
    # Try a few different initial patterns
    
    # Initial design: 4 dielectric rods
    cell = np.zeros((size, size), dtype=int)
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    # Place 4 rods at (±0.25, ±0.25)
    rod_radius = 0.12
    for dx in [-0.25, 0.25]:
        for dy in [-0.25, 0.25]:
            dist_sq = (X - dx)**2 + (Y - dy)**2
            cell[dist_sq <= rod_radius**2] = 1
    
    # Apply symmetry
    cell = _apply_symmetry(cell)
    
    # Compute initial gap
    current_gap = _compute_gap(cell)
    best_cell = cell.copy()
    best_gap = current_gap
    
    # Local search: flip individual pixels (respecting symmetry)
    # With 4-fold symmetry, we only need to consider a fundamental domain
    # The fundamental domain is roughly 1/8 of the cell
    
    # Get unique positions under D4 symmetry
    unique_positions = []
    visited = set()
    for i in range(size):
        for j in range(size):
            # Generate all symmetric positions
            positions = [
                (i, j),
                (j, i),
                (size-1-i, size-1-j),
                (size-1-i, j),
                (i, size-1-j),
                (size-1-j, size-1-i),
                (size-1-j, i),
                (j, size-1-i),
            ]
            positions = list(set(positions))
            key = tuple(sorted(positions))
            if key not in visited:
                visited.add(key)
                unique_positions.append(positions)
    
    # Shuffle for randomization
    np.random.seed(42)
    np.random.shuffle(unique_positions)
    
    # Local search iterations
    max_iter = 5000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Pick a random unique position
        pos_idx = iter_count % len(unique_positions)
        positions = unique_positions[pos_idx]
        
        # Try flipping
        new_cell = cell.copy()
        current_val = new_cell[positions[0][0], positions[0][1]]
        new_val = 1 - current_val
        for (pi, pj) in positions:
            new_cell[pi, pj] = new_val
        
        # Compute new gap
        new_gap = _compute_gap(new_cell)
        
        if new_gap > current_gap:
            cell = new_cell
            current_gap = new_gap
            if current_gap > best_gap:
                best_gap = current_gap
                best_cell = cell.copy()
        else:
            # Occasionally accept worse solutions (simulated annealing-like)
            if np.random.random() < 0.1:
                cell = new_cell
                current_gap = new_gap
    
    return best_cell.tolist()
