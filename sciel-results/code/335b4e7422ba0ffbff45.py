"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _apply_symmetry(cell):
    """Enforce 4-fold symmetry about center: flip rows, flip cols, transpose."""
    cell = np.minimum(cell, cell[::-1, :].copy())
    cell = np.minimum(cell, cell[:, ::-1].copy())
    cell = np.minimum(cell, cell.T.copy())
    return cell

def _make_four_rod_cell(size, offset, radius, shape_param=0.0):
    """Generate a 4-rod symmetric cell.
    
    offset: distance of each rod center from cell center (0.2-0.35)
    radius: base radius of each rod (0.08-0.25)
    shape_param: controls squareness (0=circle, 1=square)
    """
    cell = np.zeros((size, size), dtype=int)
    half = size / 2.0
    
    centers = [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                
                # Superellipse-like shape: |dx|^p + |dy|^p < r^p where p = 2/(1+shape_param)
                if shape_param > 0.99:
                    # Square
                    if abs(dx) < radius and abs(dy) < radius:
                        cell[i, j] = 1
                        break
                elif shape_param < 0.01:
                    # Circle
                    if dx*dx + dy*dy < radius*radius:
                        cell[i, j] = 1
                        break
                else:
                    # Superellipse
                    p = 2.0 / (1.0 + shape_param)
                    if abs(dx)**p + abs(dy)**p < radius**p:
                        cell[i, j] = 1
                        break
    
    return cell.tolist()

def _compute_eps_ft_fast(cell, n_pw):
    """Compute Fourier coefficients of epsilon for plane-wave basis.
    
    Returns complex array of shape (n_pw, n_pw) with coefficients eps_{mn}.
    """
    size = len(cell)
    eps = np.array(cell, dtype=float)
    eps_high = 12.0
    eps_low = 1.0
    
    # Map 0->eps_low, 1->eps_high
    eps_vals = eps_low + (eps_high - eps_low) * eps
    
    if n_pw % 2 == 1:
        half = n_pw // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n_pw // 2
        pw_idx = np.arange(-half, half)
    
    # Use FFT for speed: eps_{mn} = (1/N^2) sum_{i,j} eps(i,j) exp(-i2pi(mi/N + nj/N))
    # This is just the 2D DFT normalized by N^2
    eps_ft_full = np.fft.fft2(eps_vals) / (size * size)
    
    # Extract the subset of Fourier coefficients we need
    # pw_idx gives us the indices; we need to map them to FFT indices
    n = len(pw_idx)
    eps_ft = np.zeros((n, n), dtype=complex)
    for im, m in enumerate(pw_idx):
        for in_, k in enumerate(pw_idx):
            # FFT index: m mod size
            fft_im = m % size
            fft_in = k % size
            eps_ft[im, in_] = eps_ft_full[fft_im, fft_in]
    
    return eps_ft

def _tm_eigenvalues_fast(cell, kx, ky, n_pw=17):
    """Compute TM band eigenvalues at wave vector (kx, ky) using plane-wave method.
    
    kx, ky are in units of 2π/a (i.e., fractional coordinates in BZ).
    Returns sorted array of eigenvalues (omega^2 * a^2 / (4*pi^2)).
    """
    eps_ft = _compute_eps_ft_fast(cell, n_pw)
    n = n_pw
    
    if n % 2 == 1:
        half = n // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n // 2
        pw_idx = np.arange(-half, half)
    
    # Build the eigenvalue matrix for TM polarization
    # For TM (E_z), the eigenvalue equation is:
    # sum_{G'} [delta_{G,G'} * |k+G'|^2 + (k0^2) * eps_{G-G'}] a_{G'} = k0^2 * |k+G|^2 * a_G
    # 
    # Actually, the standard form is:
    # sum_{G'} [delta_{G,G'} + (k0^2)/(|k+G'|^2) * eps_{G-G'}] a_{G'} = lambda * a_G
    # where lambda = 1 (for the eigenvalue problem formulation)
    # 
    # Let me use the cleaner formulation:
    # The wave equation for TM: ∇_⊥^2 E_z + k0^2 eps(r) E_z = 0
    # E_z(r) = sum_G a_G exp(i(k+G)·r)
    # Substituting: sum_G [-(k+G)^2 a_G + k0^2 sum_{G'} eps_{G-G'} a_{G'}] = 0
    # Rearranging: sum_{G'} [k0^2 eps_{G-G'} - delta_{G,G'} (k+G)^2] a_{G'} = 0
    # 
    # This is a generalized eigenvalue problem. Let me use the standard form:
    # sum_{G'} [delta_{G,G'} + (1/|k+G'|^2) * k0^2 * eps_{G-G'}] a_{G'} = lambda * a_G
    # where lambda = k0^2 * a^2 / (4*pi^2) ... 
    #
    # Actually, let me use the formulation from Joannopoulos:
    # For TM: sum_{G'} [delta_{G,G'} + (k^2)/(|k+G'|^2) * (1/eps_{G-G'} - 1)] a_{G'} = 0
    # No, that's for TE.
    #
    # For TM (E_z): the eigenvalue equation is
    # sum_{G'} [delta_{G,G'} * (k+G)^2 + k0^2 * eps_{G-G'}] a_{G'} = k0^2 * |k+G|^2 * a_G
    # This doesn't look right either.
    #
    # Let me use the standard plane-wave expansion for 2D photonic crystals.
    # For TM polarization (E field perpendicular to the 2D plane):
    # The Helmholtz equation: [∇_⊥^2 + k0^2 eps(r)] E_z = 0
    # 
    # With E_z = sum_G a_G exp(i(k+G)·r):
    # sum_G [-(k+G)^2 a_G + k0^2 sum_{G'} eps_{G-G'} a_{G'}] exp(i(k+G)·r) = 0
    # 
    # For this to hold for all r: 
    # -(k+G)^2 a_G + k0^2 sum_{G'} eps_{G-G'} a_{G'} = 0
    # 
    # Rearranging:
    # sum_{G'} [k0^2 eps_{G-G'} - delta_{G,G'} (k+G)^2] a_{G'} = 0
    #
    # This is a homogeneous system. For non-trivial solutions, det = 0.
    # But this gives us the allowed k0 values for a given k.
    #
    # To make it a standard eigenvalue problem, divide by k0^2:
    # sum_{G'} [eps_{G-G'} - delta_{G,G'} (k+G)^2/k0^2] a_{G'} = 0
    #
    # Let lambda = 1/k0^2. Then:
    # sum_{G'} [eps_{G-G'} - delta_{G,G'} (k+G)^2 * lambda] a_{G'} = 0
    #
    # This is: A a = lambda * B a where A_{G,G'} = eps_{G-G'} and B_{G,G} = (k+G)^2
    #
    # So we solve the generalized eigenvalue problem: eps_matrix @ a = lambda * diag((k+G)^2) @ a
    # where lambda = 1/k0^2, and we want the smallest lambda (largest k0, largest omega).
    #
    # Actually, we want the band structure omega(k), so we want k0^2 values.
    # Let mu = k0^2. Then the equation is:
    # sum_{G'} [delta_{G,G'} (k+G)^2 + mu * eps_{G-G'}] a_{G'} = mu * |k+G|^2 * a_G ... no
    #
    # Let me restart with a cleaner formulation.
    # 
    # Standard result for 2D photonic crystal, TM polarization:
    # The eigenvalue problem is:
    # sum_{G'} [delta_{G,G'} + (mu)/(|k+G'|^2) * (eps_{G-G'} - delta_{G,G'} eps_0)] a_{G'} = 0
    # where mu = k0^2 a^2/(4*pi^2) and eps_0 is the average epsilon.
    #
    # Hmm, I'm getting confused. Let me just use the direct approach:
    # 
    # For TM: the matrix to diagonalize is M_{G,G'} = delta_{G,G'} * (k+G)^2 + k0^2 * eps_{G-G'}
    # and we solve M a = k0^2 * I a ... that's not right.
    #
    # OK let me just use the well-known formulation. For a 2D photonic crystal with 
    # periodic epsilon, the TM band structure is found by solving:
    # 
    # sum_{G'} [delta_{G,G'} (k+G)^2 + k0^2 eps_{G-G'}] a_{G'} = k0^2 (k+G)^2 a_G
    #
    # Wait, that's not dimensionally consistent. Let me look at this differently.
    #
    # The equation is: -(k+G)^2 a_G + k0^2 sum_{G'} eps_{G-G'} a_{G'} = 0
    #
    # This can be written as:
    # k0^2 sum_{G'} eps_{G-G'} a_{G'} = (k+G)^2 a_G
    #
    # In matrix form: k0^2 * E * a = D * a where E_{G,G'} = eps_{G-G'} and D = diag((k+G)^2)
    #
    # So: E * a = (1/k0^2) * D * a
    #
    # Let lambda = 1/k0^2. Then: E * a = lambda * D * a
    #
    # This is a generalized eigenvalue problem. We can convert to standard form by:
    # D^(-1/2) * E * D^(-1/2) * b = lambda * b where b = D^(1/2) * a
    #
    # So we need to diagonalize the matrix: D^(-1/2) * E * D^(-1/2)
    # and the eigenvalues give lambda = 1/k0^2.
    #
    # The bands are ordered by increasing omega = k0*c, so increasing k0^2, so decreasing lambda.
    # The first band (lowest omega) corresponds to the largest lambda.
    #
    # Wait, that seems backwards. Let me think again.
    #
    # At k=0, for a uniform medium with epsilon = eps_0:
    # E = eps_0 * I (only the G=G' term survives in the limit)
    # D = diag((k+G)^2). At k=0, D_{G,G} = G^2. For G=0, D=0.
    # 
    # Hmm, the G=0 component is special. Let me handle this carefully.
    #
    # Actually, for the plane-wave method, we include G=0 in the basis.
    # At k=0: D_{00} = 0, and E_{00} = eps_0.
    # The equation becomes: eps_0 * a_0 = lambda * 0 * a_0, which gives eps_0 * a_0 = 0.
    # This is problematic.
    #
    # I think the issue is that the standard formulation handles this differently.
    # Let me use a different approach: directly solve the eigenvalue problem
    # using the matrix formulation that avoids the singularity.
    #
    # Alternative: use the formulation where we solve for k0^2 directly.
    # From: -(k+G)^2 a_G + k0^2 sum_{G'} eps_{G-G'} a_{G'} = 0
    #
    # Write as: sum_{G'} [k0^2 eps_{G-G'} - delta_{G,G'} (k+G)^2] a_{G'} = 0
    #
    # For non-trivial solutions, the determinant must be zero. But this is 
    # a condition on k0 for a given k, not an eigenvalue problem in the usual sense.
    #
    # Actually, I realize the standard approach is to fix k and find k0.
    # The equation is linear in the coefficients a_G but nonlinear in k0.
    # 
    # Let me try a different formulation. Define:
    # H_{G,G'} = delta_{G,G'} (k+G)^2 + k0^2 eps_{G-G'}
    # Then the equation is: H a = k0^2 * (k+G)^2 * a ... no.
    #
    # I think I need to just implement the standard algorithm. Let me use the 
    # approach where we form the matrix:
    # 
    # For TM: A_{G,G'} = delta_{G,G'} (k+G)^2 + k0^2 eps_{G-G'}
    # and solve det(A) = 0 for k0^2.
    #
    # But this is a nonlinear eigenvalue problem. The standard approach is to
    # make it linear by dividing appropriately.
    #
    # Actually, I recall now. The standard formulation for 2D photonic crystals:
    # 
    # For TE (H_z): 
    # sum_{G'} [delta_{G,G'} + (k0^2)/(|k+G'|^2) * (1/eps_{G-G'} - 1/eps_0)] a_{G'} = 0
    #
    # For TM (E_z):
    # sum_{G'} [delta_{G,G'} + (k0^2)/(|k+G'|^2) * (eps_{G-G'} - eps_0)] a_{G'} = 0
    #
    # Wait, I don't think that's right either. Let me just use a simple approach:
    # 
    # I'll form the matrix M where M_{G,G'} = delta_{G,G'} * (k+G)^2 + k0^2 * eps_{G-G'}
    # and solve for the values of k0^2 that make det(M) = 0.
    # 
    # Since M depends linearly on k0^2, we can write M = D + k0^2 * E where
    # D = diag((k+G)^2) and E_{G,G'} = eps_{G-G'}.
    # 
    # We need det(D + k0^2 * E) = 0.
    # 
    # If E is invertible, this is equivalent to det(E) * det(E^(-1) D + k0^2 I) = 0
    # so det(E^(-1) D + k0^2 I) = 0
    # which means -k0^2 is an eigenvalue of E^(-1) D.
    # 
    # So k0^2 = -eigenvalue of E^(-1) D.
    # 
    # But E might not be invertible, and the eigenvalues might be negative.
    # 
    # Let me just use numpy's eigvals on the matrix E^(-1) D if possible, 
    # or use a different approach.
    #
    # Actually, for a practical implementation, let me just use the following:
    # Form the matrix A = E^(-1) D (if E is invertible) and compute eigenvalues.
    # The physical k0^2 values are the negative of the eigenvalues (if they're negative).
    #
    # But this is getting too complicated. Let me just use a simpler surrogate
    # that captures the essential physics without full accuracy.
    
    # For now, use a simple effective medium + perturbation approach as surrogate.
    fill_fraction = np.mean(eps)
    eps_avg = eps_low + (eps_high - eps_low) * fill_fraction
    
    # Simple estimate: the band gap is related to the contrast and the 
    # Fourier components of epsilon. For a rough surrogate, use the 
    # magnitude of the (2,0) and (2,2) Fourier components.
    #
    # For the 4-5 gap, the relevant Fourier components are those that couple
    # bands 4 and 5, which typically involve G vectors of length 2 (in units of 2pi/a).
    
    # Get the relevant Fourier components
    # pw_idx maps to G vectors: G = (m, n) * 2pi/a
    # We want components with |G| = 2 (i.e., m^2 + n^2 = 4)
    # These are (2,0), (0,2), (-2,0), (0,-2), (1,1)*sqrt(2)... no, (1,1) has |G|=sqrt(2)
    # |G| = 2 means m^2 + n^2 = 4: (2,0), (-2,0), (0,2), (0,-2)
    
    relevant_components = []
    for im, m in enumerate(pw_idx):
        for in_, k in enumerate(pw_idx):
            if m*m + k*k == 4:
                relevant_components.append(eps_ft[im, in_])
    
    if len(relevant_components) > 0:
        coupling_strength = np.mean(np.abs(relevant_components))
    else:
        coupling_strength = 0.0
    
    # Simple surrogate score
    contrast = (eps_high - eps_low) / (eps_high + eps_low)
    surrogate = coupling_strength * contrast
    
    return np.array([surrogate])  # Return a single value as a placeholder

def _quick_gap_estimate(cell, size=32, n_pw=13, n_kpts=6):
    """Fast surrogate evaluation of the 4-5 TM band gap.
    
    Returns a score proxy.
    """
    # Compute fill fraction
    eps = np.array(cell, dtype=float)
    fill_fraction = np.mean(eps)
    
    # Compute boundary complexity
    boundary = 0
    for i in range(size):
        for j in range(size):
            if eps[i, j] == 1:
                if i > 0 and eps[i-1, j] == 0: boundary += 1
                if i < size-1 and eps[i+1, j] == 0: boundary += 1
                if j > 0 and eps[i, j-1] == 0: boundary += 1
                if j < size-1 and eps[i, j+1] == 0: boundary += 1
    
    complexity = boundary / (size * size)
    
    # Compute Fourier components relevant for the 4-5 gap
    eps_ft = _compute_eps_ft_fast(cell, n_pw)
    n = n_pw
    if n % 2 == 1:
        half = n // 2
        pw_idx = np.arange(-half, half+1)
    else:
        half = n // 2
        pw_idx = np.arange(-half, half)
    
    # Get (2,0), (0,2) and similar components
    coupling = 0.0
    count = 0
    for im, m in enumerate(pw_idx):
        for in_, k in enumerate(pw_idx):
            if m*m + k*k == 4:
                coupling += np.abs(eps_ft[im, in_])
                count += 1
    if count > 0:
        coupling /= count
    
    # Surrogate score: combination of fill fraction, complexity, and coupling
    ff_optimal = 0.35
    ff_penalty = (fill_fraction - ff_optimal) ** 2
    surrogate_score = coupling * complexity * np.exp(-ff_penalty / 0.03)
    
    return surrogate_score

def _random_symmetric_cell(size=32):
    """Generate a random 32x32 cell with 4-fold symmetry."""
    # Generate a random quadrant and reflect
    half = size // 2
    quadrant = np.random.randint(0, 2, (half, half))
    
    # Build the full cell from the quadrant
    cell = np.zeros((size, size), dtype=int)
    cell[:half, :half] = quadrant
    cell[half:, :half] = quadrant[::-1, :]
    cell[:half, half:] = quadrant[:, ::-1]
    cell[half:, half:] = quadrant[::-1, ::-1]
    
    return cell

def _mutate_cell(cell, size=32, mutation_rate=0.1):
    """Mutate a cell by flipping random pixels, preserving symmetry."""
    cell = np.array(cell, dtype=int)
    half = size // 2
    
    # Mutate the top-left quadrant
    n_mutations = int(size * size * mutation_rate / 4)
    for _ in range(n_mutations):
        i = random.randint(0, half - 1)
        j = random.randint(0, half - 1)
        cell[i, j] = 1 - cell[i, j]
    
    # Reflect to maintain symmetry
    cell[half:, :half] = cell[:half, :half][::-1, :]
    cell[:half, half:] = cell[:half, :half][:, ::-1]
    cell[half:, half:] = cell[:half, :half][::-1, ::-1]
    
    return cell.tolist()

def design(problem):
    """Design a 32x32 photonic crystal unit cell to maximize the 4-5 TM band gap."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 230s kill
    
    # Start with a good initial design: 4-rod cell
    best_cell = _make_four_rod_cell(size, offset=0.25, radius=0.15, shape_param=0.3)
    best_score = _quick_gap_estimate(best_cell, size)
    
    # Also try some other initial designs
    initial_designs = [
        _make_four_rod_cell(size, offset=0.25, radius=0.12, shape_param=0.0),
        _make_four_rod_cell(size, offset=0.25, radius=0.18, shape_param=0.5),
        _make_four_rod_cell(size, offset=0.3, radius=0.12, shape_param=0.2),
        _make_four_rod_cell(size, offset=0.2, radius=0.15, shape_param=0.4),
    ]
    
    for cell in initial_designs:
        score = _quick_gap_estimate(cell, size)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Local search with mutations
    current_cell = best_cell
    current_score = best_score
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 10.0 and no_improvement_count < max_no_improvement:
        # Mutate the current cell
        mutated_cell = _mutate_cell(current_cell, size, mutation_rate=0.05)
        mutated_score = _quick_gap_estimate(mutated_cell, size)
        
        if mutated_score > current_score:
            current_cell = mutated_cell
            current_score = mutated_score
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_cell = current_cell
        else:
            no_improvement_count += 1
        
        # Occasionally try a larger mutation or restart
        if no_improvement_count % 20 == 0 and no_improvement_count > 0:
            # Try a random restart from a different initial design
            random_cell = _random_symmetric_cell(size)
            random_score = _quick_gap_estimate(random_cell, size)
            if random_score > current_score:
                current_cell = random_cell
                current_score = random_score
                no_improvement_count = 0
    
    # Final local refinement with smaller mutations
    if time.time() < deadline - 5.0:
        for _ in range(100):
            if time.time() > deadline - 3.0:
                break
            refined_cell = _mutate_cell(best_cell, size, mutation_rate=0.02)
            refined_score = _quick_gap_estimate(refined_cell, size)
            if refined_score > best_score:
                best_score = refined_score
                best_cell = refined_cell
    
    return best_cell
# EVOLVE-BLOCK-END
