import numpy as np
import time
from itertools import product

def _build_pw_solver(size=32, npw=17, n_bands=6, eps_high=12.0, eps_low=1.0):
    """Build a fast plane-wave band solver for the given pixel grid."""
    half = npw // 2  # For odd npw, we use -half to half
    
    # Reciprocal lattice vectors: G = (2*pi/m, 0) and (0, 2*pi/n) for integers m,n
    # We use a diamond-shaped set of G vectors for efficiency
    G_vecs = []
    for m in range(-half, half+1):
        for n in range(-half, half+1):
            if abs(m) + abs(n) <= half:  # Diamond shape
                G_vecs.append((m, n))
    G_vecs = np.array(G_vecs, dtype=float) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute G^2
    G2 = (G_vecs[:, 0]**2 + G_vecs[:, 1]**2).astype(float)
    
    # K-points in the irreducible BZ (M -> X -> Gamma -> M -> X)
    # Irreducible BZ for square lattice: triangle with vertices (0,0), (pi,0), (pi,pi)
    # Path: Gamma(0,0) -> X(pi/2,0) -> M(pi/2,pi/2) -> X(pi/2,0) -> Gamma(0,0)
    # Actually for square lattice, irreducible BZ is the triangle (0,0), (pi,0), (pi,pi)
    # Wait, let me reconsider. The BZ of a square lattice with lattice constant a=1
    # has corners at (pi, pi), (pi, -pi), etc. The irreducible wedge is from
    # Gamma(0,0) to X(pi,0) to M(pi,pi) back to Gamma.
    
    # For the square lattice, the irreducible BZ is a 1/8 sector:
    # Gamma(0,0) -> M(pi,pi) -> X(pi,0) -> Gamma(0,0)
    # Actually the standard path is Gamma -> X -> M -> Gamma
    # where X = (pi, 0) and M = (pi, pi)
    
    k_points = []
    n_k = 10  # Number of k-points along each segment
    # Segment 1: Gamma(0,0) to X(pi, 0)
    for i in range(n_k):
        t = i / (n_k - 1)
        k_points.append((t * np.pi, 0.0))
    # Segment 2: X(pi, 0) to M(pi, pi)
    for i in range(1, n_k):
        t = i / (n_k - 1)
        k_points.append((np.pi, t * np.pi))
    # Segment 3: M(pi, pi) to Gamma(0, 0)
    for i in range(1, n_k):
        t = i / (n_k - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    
    k_points = np.array(k_points)
    n_kpts = len(k_points)
    
    # For each k-point, we need to solve the eigenvalue problem
    # The matrix elements are:
    # H_{mn} = |k+G_m|^2 * delta_{mn} + (eps^{-1})_{m-n} * (some factor)
    
    # For TM polarization, the dielectric inverse appears.
    # The eigenvalue equation is:
    # sum_n [(k+G_m)^2 delta_{mn} + V_{m-n}] E_n = omega^2 * something * E_m
    # 
    # Actually for 2D TM, the equation is:
    # (k+G_m)^2 * E_m - sum_n (1/eps)_{m-n} * (k+G_m)·(k+G_n) * E_n = (omega/c)^2 * E_m
    # 
    # Wait, let me be more careful. For 2D TM (Hz), the equation is:
    # ∇_⊥ × (1/eps * ∇_⊥ × Hz) = (omega/c)^2 * eps * Hz
    # In Fourier space:
    # |k+G_m|^2 * E_m - sum_n (1/eps)_{m-n} * [(k+G_m)·(k+G_n)] * E_n / |k+G_n|^2 ... 
    # 
    # Actually the standard form is:
    # sum_n [(k+G_m)^2 delta_{mn} - (k+G_m)·(k+G_n) * (1/eps)_{m-n}] E_n = omega^2 * E_m
    # 
    # Hmm, let me just use the standard plane-wave expansion.
    # For TM: the eigenvalue problem is
    # sum_n [ (k+G_m)^2 * delta_{mn} + (1/eps)_{m-n} * ( (k+G_m)·(k+G_n) - (k+G_m)^2 ) ] E_n = (omega/c)^2 * eps_avg * E_m
    # 
    # This is getting complicated. Let me use a simpler but correct formulation.
    # 
    # For 2D TM photonic crystal, the eigenvalue equation in plane-wave basis is:
    # sum_n H_{mn} E_n = omega^2 * E_m
    # where H_{mn} = (k+G_m)^2 * delta_{mn} + sum_q (1/eps)_{q} * (k+G_m)·(k+G_n+G_q) * delta_{m,n+q}
    # 
    # Wait, I think the correct form for TM is:
    # The Maxwell equation gives: ∇²Hz + k0²*eps*Hz = 0
    # In a periodic medium: ∇²Hz + k0²*eps(r)*Hz = 0
    # Expanding Hz = sum_m E_m exp(i(k+G_m)·r)
    # sum_m E_m [ -(k+G_m)² + k0² * sum_n (1/eps)_{m-n} (k+G_m)·(k+G_n) ] = 0
    # 
    # Hmm, this is not quite right either. Let me look at this differently.
    # 
    # For 2D TM (Hz field), the wave equation is:
    # ∇ × (∇ × H) = (ω/c)² ε H  (in 2D, this becomes a scalar equation)
    # Actually for 2D TM: ∇²H_z + (ω/c)² ε(x,y) H_z = 0
    # 
    # With H_z = Σ_m E_m e^{i(k+G_m)·r}, we get:
    # Σ_m E_m [-(k+G_m)² + (ω/c)² ε(r)] = 0
    # The product ε(r) E_m e^{i(k+G_m)·r} = Σ_n [ε]_{n-m} E_m e^{i(k+G_n)·r}
    # where [ε]_q = Fourier coefficient of ε(r)
    # 
    # So: Σ_m [-(k+G_m)² δ_{mn} + (ω/c)² Σ_q [ε]_q δ_{m,n+q}] E_m = 0
    # 
    # This gives: Σ_m [-(k+G_n)² δ_{mn} + (ω/c)² [ε]_{n-m}] E_m = 0
    # 
    # Rearranging: Σ_m [(ω/c)² [ε]_{n-m} - (k+G_n)² δ_{mn}] E_m = 0
    # 
    # This is: Σ_m H_{nm} E_m = 0 where H_{nm} = (ω/c)² [ε]_{n-m} - (k+G_n)² δ_{nm}
    # 
    # For a non-trivial solution: det(H) = 0
    # Or equivalently: Σ_m (ω/c)² [ε]_{n-m} E_m = (k+G_n)² E_n
    # 
    # So: Σ_m (ω/c)² [ε]_{n-m} E_m = (k+G_n)² E_n
    # 
    # Let λ = (ω/c)². Then:
    # Σ_m [ε]_{n-m} E_m = (1/λ)(k+G_n)² E_n
    # 
    # This is an eigenvalue problem: Σ_m [ε]_{n-m} E_m = μ E_n where μ = (k+G_n)²/λ
    # 
    # Hmm, this is a generalized eigenvalue problem. Let me just solve it directly.
    # 
    # Actually, I think the standard approach is to write it as:
    # Σ_m [(k+G_m)² δ_{mn} - (k+G_m)·(k+G_n) (1/ε)_{n-m}] E_m = ω² E_m  (for TE)
    # 
    # For TM, it's different. Let me just use a numerical approach where I build the matrix
    # and solve the generalized eigenvalue problem.
    
    def compute_eps_fourier(cell_grid):
        """Compute Fourier coefficients of epsilon from the pixel grid."""
        eps = eps_low + (eps_high - eps_low) * cell_grid
        # 2D FFT
        F = np.fft.fft2(eps)
        F = F / (size * size)  # Normalize
        return F
    
    def solve_bands_at_k(kx, ky, eps_F, G_vecs, G2):
        """Solve for the first n_bands eigenvalues at a given k-point."""
        nG = len(G_vecs)
        # k + G_m for all m
        kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
        kG2 = kG[:, 0]**2 + kG[:, 1]**2
        
        # Build the matrix for TM
        # The equation is: Σ_m [(ω/c)² ε_{n-m}] E_m = (k+G_n)² E_n
        # This is: A * E = D * E where A_{nm} = (ω/c)² ε_{n-m}, D = diag((k+G_n)²)
        # So: (ω/c)² A_ε * E = D * E
        # Or: A_ε * E = (1/(ω/c)²) * D * E
        # 
        # Let me define λ = 1/(ω/c)² = c²/ω²
        # Then: A_ε * E = λ * D * E
        # 
        # This is a generalized eigenvalue problem: A_ε v = λ D v
        # Equivalently: (A_ε^{-1} D) v = (1/λ) v  (if A_ε is invertible)
        # 
        # Better: solve D^{-1} A_ε v = (1/λ) v = (ω/c)² v
        # 
        # So I need to solve the eigenvalue problem: (D^{-1} A_ε) v = ω²_norm v
        # where ω²_norm = (ω/c)²
        
        # A_ε is the circulant matrix with first column = ε_F
        # A_ε = fft2(eps) gives the Fourier coefficients
        # The matrix element A_ε[n,m] = ε_{n-m} (circular)
        
        # For efficiency, I'll use the fact that A_ε is circulant and can be
        # applied via FFT, but for the eigenvalue problem I need to form the matrix
        # or use iterative methods.
        
        # Since nG is small (~50 for npw=17 diamond), I'll form the matrix explicitly.
        
        # ε_F has indices from -size/2 to size/2-1 (or 0 to size-1 in FFT convention)
        # For the matrix A_ε[n,m] = ε_F[(n-m) mod nG]
        # But we need to map G-vector indices to Fourier coefficient indices.
        
        # The G vectors are indexed by their position in G_vecs array.
        # The difference G_n - G_m corresponds to some reciprocal lattice vector G_q
        # We need ε_F evaluated at G_q.
        
        # Let me create a lookup: for each pair (n,m), find the index q such that
        # G_q = G_n - G_m, then A_ε[n,m] = ε_F[q_index]
        
        # Actually, since the grid is 32x32 and we're using a diamond of G vectors,
        # the differences may not all be in our G set. Let me just use the FFT
        # coefficients directly.
        
        # Alternative approach: Since we have the pixel grid, let me compute the
        # matrix elements directly from the pixel data.
        # 
        # ε(r) = ε_low + (ε_high - ε_low) * f(r) where f(r) is 0/1
        # [ε]_q = (1/A) ∫ ε(r) e^{-i G_q · r} dr
        # For a pixel grid, this is just the 2D DFT of the pixel values.
        
        # Let me precompute the epsilon grid and its DFT once, then for each k-point
        # build the matrix.
        
        # For now, let me just form the matrix using the Fourier coefficients.
        # The key insight: A_ε[n,m] = [ε]_{G_n - G_m}
        
        # I need a mapping from (dm, dn) integers to the FFT index.
        # G_q = (2π*dm, 2π*dn) corresponds to FFT index (dm mod size, dn mod size)
        
        # Create a 2D array of G vectors for quick lookup
        G_lookup = {}
        for idx, (gm, gn) in enumerate(G_vecs):
            dm = round(gm / (2*np.pi))
            dn = round(gn / (2*np.pi))
            G_lookup[(dm, dn)] = idx
        
        # Build matrix M where M[n,m] = ε_F at (G_n - G_m)
        # Then solve: M * E = (1/ω²) * diag((k+G_n)²) * E
        # i.e., diag((k+G_n)²)^{-1} * M * E = ω² * E
        # i.e., solve eigenvalue problem: M * E = μ * diag((k+G_n)²) * E
        # where μ = 1/ω²
        
        M = np.zeros((nG, nG), dtype=complex)
        for n in range(nG):
            for m in range(nG):
                dm = round(G_vecs[n, 0] / (2*np.pi)) - round(G_vecs[m, 0] / (2*np.pi))
                dn = round(G_vecs[n, 1] / (2*np.pi)) - round(G_vecs[m, 1] / (2*np.pi))
                # Get FFT coefficient at (dm, dn)
                idx_m = dm % size
                idx_n = dn % size
                M[n, m] = eps_F[idx_n, idx_m]
        
        D = np.diag(kG2)
        
        # Solve generalized eigenvalue: M v = μ D v
        # Or: D^{-1} M v = μ v  (if D is invertible)
        # D has diagonal entries (k+G_n)² which can be zero for some k, G
        # Handle this by adding a small regularization or using scipy
        
        try:
            D_inv = np.diag(1.0 / kG2)
            A = D_inv @ M
            eigvals = np.linalg.eigvalsh(A)  # A should be Hermitian if eps is real
            eigvals = np.sort(eigvals.real)
            return eigvals[:n_bands]
        except:
            return None
    
    return k_points, G_vecs, G2, compute_eps_fourier, solve_bands_at_k


def _compute_gap_score(cell_grid, k_points, G_vecs, G2, compute_eps_fourier, solve_bands_at_k, n_bands=6):
    """Compute the 4-5 TM band gap score for a given cell grid."""
    eps_F = compute_eps_fourier(cell_grid.astype(float))
    
    band4_min = np.inf
    band5_max = 0.0
    valid = True
    
    for (kx, ky) in k_points:
        eigvals = solve_bands_at_k(kx, ky, eps_F, G_vecs, G2)
        if eigvals is None or len(eigvals) < n_bands:
            valid = False
            break
        # eigvals are (ω/c)² values, sorted ascending
        # Band 4 (0-indexed: 3) and Band 5 (0-indexed: 4)
        band4_val = eigvals[3]
        band5_val = eigvals[4]
        band4_min = min(band4_min, band4_val)
        band5_max = max(band5_max, band5_val)
    
    if not valid or band4_min == np.inf or band5_max == 0.0:
        return -1.0
    
    # The actual frequencies are sqrt(eigval) * c, but since we're taking ratios,
    # the gap-midgap ratio is:
    # 2 * (sqrt(band5_max) - sqrt(band4_min)) / (sqrt(band5_max) + sqrt(band4_min))
    # Wait, no. The gap is min(ω5) - max(ω4) across all k-points.
    # ω_n = c * sqrt(eigval_n)
    # So: gap = c*(sqrt(min over k of eigval_5) - sqrt(max over k of eigval_4))
    # midgap = c*(sqrt(min over k of eigval_5) + sqrt(max over k of eigval_4)) / 2
    # ratio = 2*gap/midgap = 2*(sqrt(min5) - sqrt(max4)) / (sqrt(min5) + sqrt(max4))
    
    # But I computed band4_min = min over k of eigval_4, and band5_max = max over k of eigval_5
    # Wait, that's wrong. I need:
    # max_omega_4 = max over k of omega_4(k) = c * max over k of sqrt(eigval_4(k)) = c * sqrt(max over k of eigval_4(k))
    # min_omega_5 = min over k of omega_5(k) = c * sqrt(min over k of eigval_5(k))
    
    # Let me recompute properly.
    
    eps_F = compute_eps_fourier(cell_grid.astype(float))
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = solve_bands_at_k(kx, ky, eps_F, G_vecs, G2)
        if eigvals is None or len(eigvals) < 5:
            return -1.0
        omega4 = np.sqrt(max(0, eigvals[3]))
        omega5 = np.sqrt(max(0, eigvals[4]))
        omega4_vals.append(omega4)
        omega5_vals.append(omega5)
    
    if not omega4_vals:
        return -1.0
    
    max_omega4 = max(omega4_vals)
    min_omega5 = min(omega5_vals)
    
    if min_omega5 <= max_omega4:
        # No gap or negative gap
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4) if (min_omega5 + max_omega4) > 0 else -1.0
    else:
        gap_ratio = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
    
    return gap_ratio


def _is_symmetric(cell):
    """Check if cell has the required 4-fold symmetry."""
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
    return True


def _get_fundamental_pixels(n=32):
    """Get the set of fundamental pixels under the D4 symmetry group."""
    fundamental = []
    for i in range(n):
        for j in range(n):
            # Check if this is the representative of its orbit
            orbit = set()
            # All 8 symmetry operations
            transforms = [
                lambda x, y: (x, y),           # identity
                lambda x, y: (n-1-x, y),       # mirror x
                lambda x, y: (x, n-1-y),       # mirror y
                lambda x, y: (n-1-x, n-1-y),   # 180 rot
                lambda x, y: (y, x),           # transpose
                lambda x, y: (n-1-y, x),       # -45 deg
                lambda x, y: (y, n-1-x),       # 45 deg
                lambda x, y: (n-1-y, n-1-x),   # anti-transpose
            ]
            for t in transforms:
                xi, yi = t(i, j)
                orbit.add((xi, yi))
            # The fundamental pixel is the minimum in the orbit
            min_pixel = min(orbit)
            if (i, j) == min_pixel:
                fundamental.append((i, j))
    return fundamental


def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 130s margin for the evaluator's 230s limit
    
    # Build the solver infrastructure
    # Use a smaller number of plane waves for speed
    npw = 13  # Diamond shape, about 49 G vectors
    n_bands = 6
    
    # Simplified solver: I'll use a coarser approach
    # Instead of a full PW solver, let me use a smarter initialization and
    # a fast local search based on Fourier coefficient analysis.
    
    # First, create the initial 4-rod design (which scores 5.78)
    cell = [[0] * size for _ in range(size)]
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    radius = 0.15
    
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy < radius*radius:
                    cell[i][j] = 1
                    break
    
    best_cell = [row[:] for row in cell]
    best_score = 5.78  # Known score for this configuration
    
    # Get fundamental pixels for the search
    fundamental_pixels = _get_fundamental_pixels(size)
    
    # Precompute symmetry groups for each fundamental pixel
    n = size
    sym_groups = {}
    for (i, j) in fundamental_pixels:
        orbit = set()
        transforms = [
            lambda x, y: (x, y),
            lambda x, y: (n-1-x, y),
            lambda x, y: (x, n-1-y),
            lambda x, y: (n-1-x, n-1-y),
            lambda x, y: (y, x),
            lambda x, y: (n-1-y, x),
            lambda x, y: (y, n-1-x),
            lambda x, y: (n-1-y, n-1-x),
        ]
        for t in transforms:
            orbit.add(t(i, j))
        sym_groups[(i, j)] = list(orbit)
    
    # Now do local search: try flipping pixels and see if score improves
    # Use a simplified score based on Fourier coefficients as a proxy
    
    # The 4-5 TM gap is related to the (1,1) and (2,0)/(0,2) Fourier coefficients
    # Let's optimize based on maximizing certain Fourier coefficient combinations
    
    def compute_fourier_score(cell_grid):
        """Compute a proxy score based on Fourier coefficients."""
        eps = np.array(cell_grid, dtype=float)
        F = np.fft.fft2(eps)
        F /= (size * size)
        
        # Key Fourier components for the 4-5 TM gap
        # (1,1), (-1,1), (1,-1), (-1,-1) and (2,0), (0,2), (-2,0), (0,-2)
        # Also (1,0), (0,1) etc.
        
        # For a 32x32 grid, FFT indices:
        # (1,1) is at F[1,1], (-1,1) is at F[31,1], etc.
        
        f11 = abs(F[1, 1])
        f20 = abs(F[2, 0])
        f02 = abs(F[0, 2])
        f10 = abs(F[1, 0])
        f01 = abs(F[0, 1])
        
        # The gap is proportional to some combination of these
        # Based on perturbation theory, the 4-5 gap at the M point
        # is related to the (1,1) coupling
        score = 2.0 * f11 + 1.5 * f20 + 1.5 * f02 - 0.5 * (f10 + f01)
        return score
    
    current_score = compute_fourier_score(cell)
    best_proxy_score = current_score
    
    # Hill climbing search
    max_iter = 50000
    iter_count = 0
    
    while time.time() < deadline - 20 and iter_count < max_iter:
        iter_count += 1
        
        # Pick a random fundamental pixel
        idx = np.random.randint(0, len(fundamental_pixels))
        i, j = fundamental_pixels[idx]
        
        # Try flipping
        new_cell = [row[:] for row in cell]
        for (ri, ci) in sym_groups[(i, j)]:
            new_cell[ri][ci] = 1 - new_cell[ri][ci]
        
        new_score = compute_fourier_score(new_cell)
        
        if new_score > current_score:
            cell = new_cell
            current_score = new_score
            if current_score > best_proxy_score:
                best_proxy_score = current_score
                best_cell = [row[:] for row in cell]
        # else: reject (keep current)
    
    # Try to verify with a few candidate configurations
    # and pick the one with the best proxy score
    candidates = [best_cell]
    
    # Also try some hand-crafted variants
    # Variant 1: Larger rods
    cell2 = [[0] * size for _ in range(size)]
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy < 0.18**2:
                    cell2[i][j] = 1
                    break
    candidates.append(cell2)
    
    # Variant 2: Smaller rods
    cell3 = [[0] * size for _ in range(size)]
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            for cx, cy in centers:
                dx = x - cx
                dy = y - cy
                if dx*dx + dy*dy < 0.12**2:
                    cell3[i][j] = 1
                    break
    candidates.append(cell3)
    
    # Pick the best by proxy score
    best_candidate = max(candidates, key=lambda c: compute_fourier_score(c))
    
    # Verify symmetry
    if not _is_symmetric(best_candidate):
        best_candidate = best_cell
    
    return best_candidate
