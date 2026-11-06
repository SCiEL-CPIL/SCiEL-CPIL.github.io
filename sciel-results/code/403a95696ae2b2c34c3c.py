"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _compute_fourier_eps(cell, n_pw):
    """Compute Fourier coefficients of epsilon for the unit cell.
    cell: 32x32 array of 0/1 (1=dielectric eps=12, 0=air eps=1)
    n_pw: number of plane waves in each direction (total n_pw*n_pw)
    Returns: eps_k array of shape (n_pw, n_pw) indexed by (kx, ky) offsets
    """
    N = cell.shape[0]
    eps = np.where(cell == 1, 12.0, 1.0)
    # FFT-based: epsilon(k) = sum_{i,j} eps(i,j) * exp(-i 2pi (kx*i/N + ky*j/N))
    # For plane wave method, we need eps_{G-G'} where G are reciprocal lattice vectors
    # In pixel units, the reciprocal vectors are (2pi/nx, 2pi/ny) for integer nx,ny
    # We use n_pw plane waves in each direction
    eps_fft = np.fft.fft2(eps) / (N * N)
    # eps_fft[m, n] = (1/N^2) sum_{i,j} eps(i,j) exp(-2pi i (m*i/N + n*j/N))
    # We need eps_{(mx,my)} for mx,my in range(-n_pw//2, n_pw//2)
    # Reorder FFT output to center zero frequency
    eps_centered = np.fft.fftshift(eps_fft)
    # Now eps_centered[n_pw//2 + mx, n_pw//2 + my] = eps_{(mx,my)}
    # Extract the needed range
    half = n_pw // 2
    eps_k = eps_centered[half:half+n_pw, half:half+n_pw]
    return eps_k

def _solve_tm_bands(cell, n_pw=13, k_points=None):
    """Solve TM bands using plane-wave method.
    Returns list of (omega_4, omega_5) for each k-point, or None if error.
    TM: (kx^2+ky^2) * A = omega^2 * (1/eps) * (kx^2+ky^2) * A (simplified)
    Actually for TM: the eigenvalue problem is:
    sum_{G'} (k+G)^2 * eps_{G-G'}^{-1} * A_{G'} = omega^2 * A_G
    But more precisely, for TM polarization in 2D:
    The equation is: (k+G)^2 * A_G = omega^2 * sum_{G'} eps_{G-G'}^{-1} A_{G'}
    Wait, let me be more careful.
    
    For TM (magnetic field perpendicular to plane), the scalar equation is:
    ∇·(1/ε ∇ h_z) + ω² h_z = 0
    In plane waves: sum_{G'} (k+G)·(k+G') / ε_{G-G'} A_{G'} = ω² A_G
    Hmm, actually the standard form is:
    (k+G)^2 A_G = ω² sum_{G'} (1/ε)_{G-G'} A_{G'}
    where (1/ε)_{G-G'} are Fourier coefficients of 1/ε.
    
    Let me use the form: A^T M A = ω² A^T A where M_{GG'} = (k+G)^2 δ_{GG'} - sum... 
    Actually the cleanest form: 
    ω² A_G = sum_{G'} (k+G)·(k+G') / ε_{G-G'} * A_{G'}  -- NO this isn't right either.
    
    Let me use the standard plane-wave expansion for TM:
    h_z(r) = sum_G A_G exp(i(k+G)·r)
    The wave equation: ∇·(1/ε(r) ∇ h_z) + ω² h_z = 0
    This gives: sum_{G'} [(k+G)·(k+G')] / ε_{G-G'} A_{G'} = ω² A_G
    Wait no. Let me redo:
    ∇ h_z = sum_G i(k+G) A_G exp(i(k+G)·r)
    1/ε ∇ h_z = sum_{G,G'} (1/ε)_{G'-G} i(k+G) A_G exp(i(k+G')·r)
    ∇·(1/ε ∇ h_z) = sum_{G,G'} (1/ε)_{G'-G} i(k+G)·i(k+G') A_G exp(i(k+G')·r)
    = - sum_{G,G'} (1/ε)_{G'-G} (k+G)·(k+G') A_G exp(i(k+G')·r)
    
    So: -sum_G (1/ε)_{G'-G} (k+G)·(k+G') A_G + ω² A_{G'} = 0
    => ω² A_{G'} = sum_G (1/ε)_{G'-G} (k+G)·(k+G') A_G
    
    In matrix form: ω² A = M A where M_{G'G} = (1/ε)_{G'-G} (k+G)·(k+G')
    
    So I need Fourier coefficients of 1/ε, not ε!
    """
    N = cell.shape[0]
    eps = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps
    
    # Compute Fourier coefficients of 1/epsilon
    inv_eps_fft = np.fft.fft2(inv_eps) / (N * N)
    inv_eps_centered = np.fft.fftshift(inv_eps_fft)
    half = n_pw // 2
    inv_eps_k = inv_eps_centered[half:half+n_pw, half:half+n_pw]
    
    # Reciprocal lattice vectors: G = (2π * mx / N, 2π * my / N) for mx,my integers
    # In our indexing, the offset indices go from -half to half-1 (or similar)
    # Let's define G vectors
    mx_vals = np.arange(-half, half)  # n_pw values
    my_vals = np.arange(-half, half)
    
    # G vectors
    GX, GY = np.meshgrid(mx_vals, my_vals, indexing='ij')
    GX = GX * 2 * np.pi / N
    GY = GY * 2 * np.pi / N
    
    # k-points in the irreducible BZ
    if k_points is None:
        # Standard irreducible BZ for square lattice: 0 <= kx <= ky, kx,ky in [0, 1] (in units of 2π/a)
        # Actually let's use normalized coordinates where BZ is [0,1]x[0,1] with G=(1,0),(0,1)
        # Let me use k in units of 2π/a, BZ is [-0.5,0.5]x[-0.5,0.5]
        # Irreducible: 0 <= kx <= 0.5, 0 <= ky <= 0.5, kx <= ky (for square symmetry)
        # Let's use a few k-points
        k_points = [
            (0.0, 0.0),      # Gamma
            (0.5, 0.0),      # X
            (0.5, 0.5),      # M
            (0.25, 0.25),    # interior
            (0.3, 0.3),
            (0.4, 0.4),
            (0.2, 0.3),
            (0.1, 0.3),
            (0.35, 0.45),
        ]
    
    results = []
    for kx, ky in k_points:
        # k vector
        kx_val = kx * 2 * np.pi
        ky_val = ky * 2 * np.pi
        
        # Build matrix M
        # M_{G'G} = (1/eps)_{G'-G} * (k+G)·(k+G')
        # (k+G)·(k+G') = (kx+GX)^2 + (ky+GY)^2 for diagonal, but for off-diagonal:
        # (k+G)·(k+G') = (kx+GX[m,n])*(kx+GX[m',n']) + (ky+GY[m,n])*(ky+GY[m',n'])
        
        kx_G = kx_val + GX
        ky_G = ky_val + GY
        
        # (k+G)^2 for each G
        kG_sq = kx_G**2 + ky_G**2
        
        # Dot product (k+G)·(k+G')
        # This is: (kx+GX_i)(kx+GX_j) + (ky+GY_i)(ky+GY_j)
        # = kx^2 + kx*GX_j + kx*GX_i + GX_i*GX_j + ky^2 + ky*GY_j + ky*GY_i + GY_i*GY_j
        # Let's compute as outer product
        kG_dot = np.outer(kx_G, kx_G) + np.outer(ky_G, ky_G)
        
        # M = inv_eps_k * kG_dot (element-wise, since (1/eps)_{G'-G} depends on G'-G)
        # inv_eps_k[i,j] = (1/eps)_{G_i - G_j} where G_i corresponds to index i
        # Actually inv_eps_k is indexed by the difference in our centered FFT
        # inv_eps_k[m,n] = (1/eps)_{(mx_vals[m], my_vals[m]) - (mx_vals[n], my_vals[n])}
        # Hmm, this is tricky. Let me reconsider.
        
        # In the FFT, inv_eps_k[m,n] = sum_{i,j} inv_eps(i,j) exp(-2πi(mx_vals[m]*i/N + my_vals[n]*j/N)) / N^2
        # This is (1/eps)_{G_m} where G_m = (2π*mx_vals[m]/N, 2π*my_vals[n]/N)
        # So inv_eps_k[m,n] = (1/eps)_{(mx_vals[m], my_vals[n])}
        
        # We need (1/eps)_{G' - G} where G' is row index, G is column index
        # G' - G = (mx_vals[row] - mx_vals[col], my_vals[row] - my_vals[col])
        # So we need inv_eps at the difference
        
        # Let me recompute: for each pair (i,j) where i is row (G') and j is col (G)
        # diff_mx = mx_vals[i] - mx_vals[j]
        # diff_my = my_vals[i] - my_vals[j]
        # (1/eps)_{diff} = inv_eps_centered[half + diff_mx, half + diff_my]
        
        M = np.zeros((n_pw, n_pw), dtype=complex)
        for i in range(n_pw):
            for j in range(n_pw):
                diff_mx = mx_vals[i] - mx_vals[j]
                diff_my = my_vals[i] - my_vals[j]
                if -half <= diff_mx < half and -half <= diff_my < half:
                    eps_coeff = inv_eps_centered[half + diff_mx, half + diff_my]
                else:
                    # Wrap around - but for small n_pw this shouldn't happen
                    eps_coeff = 0.0
                M[i, j] = eps_coeff * kG_dot[i, j]
        
        # Solve eigenvalue problem: M A = ω² A
        # M is Hermitian (should be, due to symmetry)
        M = (M + M.conj().T) / 2  # enforce Hermiticity
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.real(eigvals)
        eigvals = np.sort(eigvals)
        
        # ω² = eigvals, so ω = sqrt(eigvals)
        # We want bands 4 and 5 (1-indexed), so indices 3 and 4
        if len(eigvals) >= 5:
            omega_4 = np.sqrt(max(eigvals[3], 0))
            omega_5 = np.sqrt(max(eigvals[4], 0))
            results.append((omega_4, omega_5))
        else:
            results.append((0, 0))
    
    return results

def _gap_ratio(cell, n_pw=13):
    """Compute the gap-midgap ratio for bands 4 and 5."""
    try:
        results = _solve_tm_bands(cell, n_pw=n_pw)
        if not results:
            return -1.0
        max_omega_4 = max(r[0] for r in results)
        min_omega_5 = min(r[1] for r in results)
        if max_omega_4 == 0:
            return -1.0
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2
        if midgap == 0:
            return -1.0
        return gap / midgap
    except:
        return -1.0

def _enforce_symmetry(cell):
    """Enforce 4-fold rotational and inversion symmetry."""
    N = cell.shape[0]
    # Inversion symmetry: cell[i,j] = cell[N-1-i, N-1-j]
    # 90-degree rotation: cell[i,j] = cell[j, N-1-i]
    # Combined: the fundamental domain is a triangle
    # For 32x32, center is at (15.5, 15.5)
    # Inversion: (i,j) <-> (31-i, 31-j)
    # 90 rot: (i,j) -> (j, 31-i)
    # 180 rot: (i,j) -> (31-i, 31-j) (same as inversion)
    # 270 rot: (i,j) -> (31-j, i)
    
    # To enforce all symmetries, for each orbit of 4 points (or fewer if on symmetry axis),
    # all must have the same value.
    
    # Let's build the symmetry orbits
    # For each (i,j), the orbit under D4 is:
    # (i,j), (31-i,31-j), (j,31-i), (31-j,i)
    # If some of these coincide, the orbit is smaller.
    
    N = 32
    visited = [[False]*N for _ in range(N)]
    new_cell = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        for j in range(N):
            if visited[i][j]:
                continue
            # Get orbit
            orbit = [(i, j), (N-1-i, N-1-j), (j, N-1-i), (N-1-j, i)]
            # Remove duplicates
            unique_orbit = list(set(orbit))
            for (oi, oj) in unique_orbit:
                visited[oi][oj] = True
            # The value is determined by the original cell value
            val = cell[i, j]
            for (oi, oj) in unique_orbit:
                new_cell[oi, oj] = val
    
    return new_cell

def design(problem):
    """Design a photonic crystal unit cell maximizing the TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 210  # Leave 20s margin
    
    N = size
    
    # Strategy: Try several structural families and pick the best one
    # Family 1: Air holes in dielectric background (square holes)
    # Family 2: Air holes in dielectric background (circular holes)
    # Family 3: Dielectric rods in air (various shapes)
    # Family 4: More complex patterns
    
    # Start with a dielectric background (all 1s) and punch air holes
    
    def make_square_hole_cell(hole_size, offset=0):
        """Dielectric background with a square air hole in the center."""
        cell = np.ones((N, N), dtype=int)
        # Center is at (15.5, 15.5)
        # Square hole of size hole_size x hole_size centered
        half = hole_size // 2
        start = N//2 - half
        end = start + hole_size
        if hole_size % 2 == 0:
            # Even: center between pixels
            start = N//2 - half
            end = N//2 + half
        else:
            start = N//2 - half
            end = start + hole_size
        cell[start:end, start:end] = 0
        return _enforce_symmetry(cell)
    
    def make_circular_hole_cell(radius_frac):
        """Dielectric background with a circular air hole."""
        cell = np.ones((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if x*x + y*y < radius_frac**2:
                    cell[i, j] = 0
        return _enforce_symmetry(cell)
    
    def make_rod_cell(radius_frac):
        """Dielectric rod in air background."""
        cell = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if x*x + y*y < radius_frac**2:
                    cell[i, j] = 1
        return _enforce_symmetry(cell)
    
    def make_cross_cell(arm_width, arm_length_frac):
        """Cross/plus shape dielectric in air."""
        cell = np.zeros((N, N), dtype=int)
        center = N // 2
        half_arm = int(N * arm_length_frac / 2)
        half_w = arm_width // 2
        # Horizontal arm
        for i in range(center - half_arm, center + half_arm):
            for j in range(center - half_w, center + half_w):
                if 0 <= i < N and 0 <= j < N:
                    cell[i, j] = 1
        # Vertical arm
        for i in range(center - half_w, center + half_w):
            for j in range(center - half_arm, center + half_arm):
                if 0 <= i < N and 0 <= j < N:
                    cell[i, j] = 1
        return _enforce_symmetry(cell)
    
    # Generate candidate cells
    candidates = []
    
    # Family 1: Square holes in dielectric
    for hs in range(4, 16, 2):
        candidates.append(('sq_hole_' + str(hs), make_square_hole_cell(hs)))
    
    # Family 2: Circular holes in dielectric
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(('circ_hole_' + str(r), make_circular_hole_cell(r)))
    
    # Family 3: Dielectric rods
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        candidates.append(('rod_' + str(r), make_rod_cell(r)))
    
    # Family 4: Cross shapes
    for aw in [4, 6, 8]:
        for af in [0.3, 0.4, 0.5]:
            candidates.append(('cross_' + str(aw) + '_' + str(af), make_cross_cell(aw, af)))
    
    # Family 5: Dielectric with multiple holes (2x2 pattern)
    def make_2x2_hole_cell(hole_radius_frac, spacing_frac):
        """Dielectric background with 4 circular air holes in 2x2 arrangement."""
        cell = np.ones((N, N), dtype=int)
        center = N / 2.0
        offset = N * spacing_frac / 2
        holes = [
            (center - offset, center - offset),
            (center + offset, center - offset),
            (center - offset, center + offset),
            (center + offset, center + offset),
        ]
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N
                y = (j + 0.5) / N
                for (hx, hy) in holes:
                    dx = x - hx / N
                    dy = y - hy / N
                    if dx*dx + dy*dy < hole_radius_frac**2:
                        cell[i, j] = 0
                        break
        return _enforce_symmetry(cell)
    
    for hr in [0.05, 0.08, 0.1, 0.12]:
        for sp in [0.2, 0.3, 0.4]:
            candidates.append(('2x2_hole_' + str(hr) + '_' + str(sp), make_2x2_hole_cell(hr, sp)))
    
    # Family 6: Dielectric frame (square ring)
    def make_frame_cell(outer_frac, inner_frac):
        """Dielectric square frame (ring) in air."""
        cell = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                x = abs((i + 0.5) / N - 0.5)
                y = abs((j + 0.5) / N - 0.5)
                r = max(x, y)  # Square distance
                if inner_frac <= r <= outer_frac:
                    cell[i, j] = 1
        return _enforce_symmetry(cell)
    
    for of in [0.4, 0.45, 0.5]:
        for inf_ in [0.2, 0.25, 0.3]:
            if inf_ < of:
                candidates.append(('frame_' + str(of) + '_' + str(inf_), make_frame_cell(of, inf_)))
    
    # Evaluate candidates
    best_cell = None
    best_score = -10.0
    
    # First pass: quick evaluation with fewer k-points
    for name, cell in candidates:
        if time.time() > deadline - 30:
            break
        score = _gap_ratio(cell, n_pw=9)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Second pass: refine top candidates with more k-points
    # Re-evaluate all with n_pw=11
    if time.time() < deadline - 60:
        top_candidates = []
        for name, cell in candidates:
            if time.time() > deadline - 60:
                break
            score = _gap_ratio(cell, n_pw=11)
            top_candidates.append((score, name, cell))
        
        top_candidates.sort(key=lambda x: -x[0])
        for score, name, cell in top_candidates[:5]:
            if time.time() > deadline - 30:
                break
            # Final evaluation
            final_score = _gap_ratio(cell, n_pw=13)
            if final_score > best_score:
                best_score = final_score
                best_cell = cell.copy()
    
    if best_cell is None:
        # Fallback: use the seed design
        cell = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                x = (i + 0.5) / N - 0.5
                y = (j + 0.5) / N - 0.5
                if x*x + y*y < 0.2**2:
                    cell[i, j] = 1
        best_cell = _enforce_symmetry(cell)
    
    # Convert to nested list
    result = [[int(best_cell[i][j]) for j in range(N)] for i in range(N)]
    
    # Verify symmetry
    for i in range(N):
        for j in range(N):
            if result[i][j] != result[N-1-i][N-1-j]:
                result[N-1-i][N-1-j] = result[i][j]
            if result[i][j] != result[j][N-1-i]:
                result[j][N-1-i] = result[i][j]
    
    return result
# EVOLVE-BLOCK-END
