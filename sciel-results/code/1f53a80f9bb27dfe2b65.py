"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 210.0  # Safety margin before 230s kill

def _elapsed():
    return time.time() - _START_TIME

def _check_deadline():
    return _elapsed() > TIME_LIMIT

def _make_cell_from_rods(size, cx, cy, r, n_rods=4, offset=0.25):
    """Create a cell with 4-fold symmetric rod placement.
    cx, cy define the primary rod center offset from center.
    The 4 rods are at (cx,cy), (cy,cx), (-cx,-cy), (-cy,-cx) in normalized coords.
    """
    cell = np.zeros((size, size), dtype=np.int8)
    half = size / 2.0
    # Pixel centers in [-0.5, 0.5)
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    
    # Four rod centers (4-fold symmetric)
    # Primary at (cx, cy), then rotate 90, 180, 270
    centers = [
        (cx, cy),
        (cy, -cx),
        (-cx, -cy),
        (-cy, cx)
    ]
    
    for (rcx, rcy) in centers:
        dx = xs - rcx
        dy = ys - rcy
        dist2 = dx[None, :] ** 2 + dy[:, None] ** 2
        mask = dist2 < r ** 2
        cell |= mask.astype(np.int8)
    
    return cell

def _make_cell_single_rod(size, r):
    """Single centered rod."""
    cell = np.zeros((size, size), dtype=np.int8)
    xs = (np.arange(size) + 0.5) / size - 0.5
    ys = (np.arange(size) + 0.5) / size - 0.5
    dist2 = xs[None, :] ** 2 + ys[:, None] ** 2
    cell = (dist2 < r ** 2).astype(np.int8)
    return cell

def _compute_fourier_coefficients(cell, n_pw):
    """Compute Fourier coefficients of 1/eps for the unit cell.
    n_pw: number of plane waves in each dimension (total n_pw x n_pw)
    Returns eps_inv_coeffs: 2D array of shape (n_pw, n_pw)
    """
    size = cell.shape[0]
    cell_f = cell.astype(np.float64)
    eps = 1.0 + (12.0 - 1.0) * cell_f  # eps = 1 + 11*cell
    
    # Compute 1/eps
    eps_inv = 1.0 / eps
    
    # Fourier coefficients: c_{m,n} = (1/N^2) sum_{i,j} eps_inv[i,j] exp(-2πi(mi/N + nj/N))
    # Use 2D FFT
    # We need coefficients for wavevectors G = (2π/a)(m, n) where m,n are integers
    # The plane wave expansion uses G vectors with |G| <= Gmax
    
    # For n_pw plane waves in each direction, we need G components from -n_pw//2 to n_pw//2
    # But we need to select those with |G| <= Gmax. For simplicity, use a square cutoff.
    
    # Actually, let's compute all Fourier coefficients up to order n_pw//2
    # The FFT gives us coefficients at frequencies 0, 1, ..., N-1
    # We need to map to -N/2, ..., N/2-1
    
    fft2 = np.fft.fft2(eps_inv)
    # Shift so that zero frequency is at center
    fft2_shifted = np.fft.fftshift(fft2)
    
    # The coefficient for G=(m,n) is fft2_shifted[m+N/2, n+N/2] / N^2
    # But we need to be careful about the indexing
    
    # For the plane wave method, we need eps_inv(G) for G in our basis
    # Let's return the full shifted FFT scaled by 1/N^2
    return fft2_shifted / (size * size)

def _tm_band_structure(cell, n_pw, k_points, eps_high=12.0):
    """Compute TM band structure using plane-wave method.
    
    For TM: eigenvalue problem is (G+k) · eps^{-1}(G-G') · (G'+k) = ω^2/a^2 * δ
    
    Where eps^{-1}(G) are Fourier coefficients of 1/epsilon.
    
    Returns array of eigenvalues (ω^2/a^2) for each k-point, sorted.
    """
    size = cell.shape[0]
    cell_f = cell.astype(np.float64)
    eps = 1.0 + (eps_high - 1.0) * cell_f
    eps_inv = 1.0 / eps
    
    # Compute Fourier coefficients of eps_inv
    fft2 = np.fft.fft2(eps_inv) / (size * size)
    fft2_shifted = np.fft.fftshift(fft2)
    
    # Select plane wave basis: G vectors with |G| <= Gmax
    # n_pw total in 2D means we select G vectors within a circle of radius ~n_pw/2
    half = n_pw // 2
    
    # Build list of G vectors (m, n) with m^2 + n^2 <= half^2
    g_vectors = []
    for m in range(-half, half + 1):
        for n in range(-half, half + 1):
            if m * m + n * n <= half * half:
                g_vectors.append((m, n))
    
    # Sort by magnitude for consistency
    g_vectors.sort(key=lambda v: (v[0]**2 + v[1]**2, v[0], v[1]))
    
    n_g = len(g_vectors)
    if n_g == 0:
        return np.zeros((len(k_points), 1))
    
    # Get Fourier coefficients for these G vectors
    # fft2_shifted is indexed as [m+half_size, n+half_size] where half_size = size//2
    # But our G indices m,n are in range [-half, half], not [-size//2, size//2]
    # We need to map: coefficient for G=(m,n) is fft2[m % size, n % size] / size^2
    
    eps_inv_coeffs = np.zeros(n_g, dtype=np.complex128)
    for idx, (m, n) in enumerate(g_vectors):
        # FFT index: m mod size, n mod size
        im = m % size
        in_ = n % size
        eps_inv_coeffs[idx] = fft2[im, in_]
    
    # For each k-point, build and solve the eigenvalue problem
    results = np.zeros((len(k_points), n_g))
    
    for ki, (kx, ky) in enumerate(k_points):
        # Build matrix A where A[i,j] = (Gi+k) · eps_inv(Gi-Gj) · (Gj+k)
        # Gi = (2π/a)(mi, ni), so (Gi+k) = (2π*mi + kx, 2π*ni + ky)
        
        # Precompute (Gi+k) vectors
        gk = np.array([(2.0 * math.pi * m + kx, 2.0 * math.pi * n + ky) 
                       for (m, n) in g_vectors])
        
        # Build matrix
        A = np.zeros((n_g, n_g), dtype=np.float64)
        for i in range(n_g):
            gi = g_vectors[i]
            for j in range(n_g):
                gj = g_vectors[j]
                # G_diff = Gi - Gj
                gm = gi[0] - gj[0]
                gn = gi[1] - gj[1]
                # Get eps_inv(G_diff)
                # Need to find index of (gm, gn) in g_vectors, or compute directly
                # For efficiency, precompute a dictionary
                # But for now, compute directly from FFT
                im = gm % size
                in_ = gn % size
                e_inv = fft2[im, in_].real  # Should be real for symmetric cell
                
                # (Gi+k) · (Gj+k)
                dot = gk[i][0] * gk[j][0] + gk[i][1] * gk[j][1]
                
                A[i, j] = e_inv * dot
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(A)
            # Keep only positive eigenvalues
            eigvals = eigvals[eigvals > 0]
            results[ki, :len(eigvals)] = eigvals
        except:
            results[ki, :] = np.inf
    
    return results

def _tm_band_structure_fast(cell, n_g, k_points, eps_high=12.0):
    """Faster version using precomputed structure and vectorization."""
    size = cell.shape[0]
    cell_f = cell.astype(np.float64)
    eps = 1.0 + (eps_high - 1.0) * cell_f
    eps_inv = 1.0 / eps
    
    fft2 = np.fft.fft2(eps_inv) / (size * size)
    
    # Build G vector list
    half = int(math.sqrt(n_g))
    g_vectors = []
    for m in range(-half, half + 1):
        for n in range(-half, half + 1):
            if m * m + n * n <= half * half:
                g_vectors.append((m, n))
    g_vectors.sort(key=lambda v: (v[0]**2 + v[1]**2, v[0], v[1]))
    n_g_actual = min(n_g, len(g_vectors))
    g_vectors = g_vectors[:n_g_actual]
    
    # Precompute eps_inv coefficients for all needed G differences
    # We need eps_inv(Gi - Gj) for all i, j
    # Precompute a lookup
    g_set = {}
    for idx, (m, n) in enumerate(g_vectors):
        g_set[(m, n)] = idx
    
    # Precompute eps_inv for all G differences
    eps_inv_matrix = np.zeros((n_g_actual, n_g_actual), dtype=np.float64)
    for i in range(n_g_actual):
        for j in range(n_g_actual):
            gm = g_vectors[i][0] - g_vectors[j][0]
            gn = g_vectors[i][1] - g_vectors[j][1]
            im = gm % size
            in_ = gn % size
            eps_inv_matrix[i, j] = fft2[im, in_].real
    
    results = np.zeros((len(k_points), n_g_actual))
    
    for ki, (kx, ky) in enumerate(k_points):
        gk = np.array([(2.0 * math.pi * m + kx, 2.0 * math.pi * n + ky) 
                       for (m, n) in g_vectors[:n_g_actual]])
        
        # A[i,j] = eps_inv_matrix[i,j] * dot(gk[i], gk[j])
        # dot(gk[i], gk[j]) = gk[i] @ gk[j]
        dots = gk @ gk.T
        A = eps_inv_matrix * dots
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = eigvals[eigvals > 1e-10]
            results[ki, :len(eigvals)] = eigvals
        except:
            results[ki, :] = np.inf
    
    return results

def _get_irreducible_k_points(n_k=30):
    """Get k-points in the irreducible Brillouin zone of square lattice.
    The irreducible BZ is the triangle (0,0)-(π/a,0)-(π/a,π/a).
    We parameterize along the boundary and interior.
    """
    k_points = []
    # High symmetry points and lines
    # X = (π, 0), M = (π, π), Γ = (0, 0)
    # Irreducible BZ: 0 <= kx <= ky, kx + ky <= π (in units of 2π/a)
    # Actually for square lattice, the first BZ is [-π/a, π/a]^2
    # Irreducible sector: 0 <= kx <= π/a, 0 <= ky <= π/a, kx >= ky (or similar)
    
    # Let's use: kx in [0, π], ky in [0, π], with kx >= ky
    # This covers the irreducible BZ
    
    # Sample points along the boundaries and interior
    # Line Γ-X: (t, 0) for t in [0, π]
    # Line X-M: (π, t) for t in [0, π]
    # Line M-Γ: (t, t) for t in [0, π]
    # Interior: random points with 0 <= ky <= kx <= π
    
    n_boundary = n_k // 2
    n_interior = n_k - n_boundary
    
    # Boundary points
    for i in range(n_boundary // 3):
        t = i / max(1, n_boundary // 3 - 1) * math.pi
        k_points.append((t, 0.0))  # Γ-X
    for i in range(n_boundary // 3):
        t = i / max(1, n_boundary // 3 - 1) * math.pi
        k_points.append((math.pi, t))  # X-M
    for i in range(n_boundary - 2 * (n_boundary // 3)):
        t = i / max(1, n_boundary - 2 * (n_boundary // 3) - 1) * math.pi
        k_points.append((t, t))  # M-Γ
    
    # Interior points (deterministic grid)
    n_int_side = int(math.ceil(math.sqrt(n_interior)))
    for i in range(n_int_side):
        for j in range(n_int_side):
            if len(k_points) >= n_k:
                break
            kx = (i + 0.5) / n_int_side * math.pi
            ky = (j + 0.5) / n_int_side * math.pi
            if ky <= kx:  # In irreducible sector
                k_points.append((kx, ky))
    
    # Pad if needed
    while len(k_points) < n_k:
        k_points.append((math.pi / 2, math.pi / 4))
    
    return k_points[:n_k]

def _score_gap(eigenvalues, band_low=4, band_high=5):
    """Compute gap-midgap ratio from eigenvalues.
    eigenvalues: 2D array (n_k, n_bands)
    Returns the gap ratio.
    """
    # Find max of band_low across all k-points
    # Find min of band_high across all k-points
    # Only consider k-points that have enough bands
    
    valid = []
    for row in eigenvalues:
        if len(row) >= band_high and all(np.isfinite(row[:band_high])):
            valid.append(row)
    
    if not valid:
        return -1.0
    
    valid = np.array(valid)
    
    # Band indices are 0-based, so band 4 is index 3, band 5 is index 4
    max_omega4 = np.max(valid[:, band_low - 1])
    min_omega5 = np.min(valid[:, band_high - 1])
    
    if max_omega4 <= 0 or min_omega5 <= 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap <= 0:
        return -1.0
    
    return gap / midgap

def _surrogate_evaluate(cell, n_g=49, n_k=12):
    """Fast surrogate evaluation with fewer plane waves and k-points."""
    k_points = _get_irreducible_k_points(n_k)
    eigvals = _tm_band_structure_fast(cell, n_g, k_points)
    return _score_gap(eigvals, 4, 5)

def _full_evaluate(cell, n_g=289, n_k=30):
    """Full evaluation matching the evaluator."""
    k_points = _get_irreducible_k_points(n_k)
    eigvals = _tm_band_structure_fast(cell, n_g, k_points)
    return _score_gap(eigvals, 4, 5)

def _is_deadline():
    return _elapsed() > TIME_LIMIT

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    
    _START_TIME_LOCAL = time.time()
    TIME_LIMIT_LOCAL = 200.0
    
    def _time_left():
        return TIME_LIMIT_LOCAL - (time.time() - _START_TIME_LOCAL)
    
    # Start with the four-rod configuration (known to generate (1,1) Fourier components)
    # and refine using local search.
    
    best_cell = None
    best_score = -1.0
    
    # Initial candidates: various four-rod configurations
    # Parameters: cx, cy (rod center offset), r (rod radius)
    # 4-fold symmetric: rods at (cx,cy), (cy,-cx), (-cx,-cy), (-cy,cx)
    
    candidates = []
    
    # Grid of initial candidates
    for cx in [0.15, 0.2, 0.25, 0.3, 0.35]:
        for cy in [0.05, 0.1, 0.15, 0.2, 0.25]:
            for r in [0.12, 0.15, 0.18, 0.2, 0.22]:
                if cx * cx + cy * cy + r * r < 0.25:  # Rods don't overlap center
                    candidates.append((cx, cy, r))
    
    # Also try single rod as baseline
    candidates.append(('single', 0.2, 0.0))
    
    # Evaluate candidates with fast surrogate
    evaluated = []
    for idx, cand in enumerate(candidates):
        if time.time() - _START_TIME_LOCAL > TIME_LIMIT_LOCAL - 60:
            break
        
        if isinstance(cand[0], str):
            # Single rod
            cell = _make_cell_single_rod(size, cand[1])
        else:
            cx, cy, r = cand
            cell = _make_cell_from_rods(size, cx, cy, r)
        
        score = _surrogate_evaluate(cell, n_g=49, n_k=12)
        evaluated.append((score, cell, cand))
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
    
    # Sort by score and take top candidates for refinement
    evaluated.sort(key=lambda x: -x[0])
    
    # Local refinement of top candidates
    top_n = min(10, len(evaluated))
    for score, cell, cand in evaluated[:top_n]:
        if time.time() - _START_TIME_LOCAL > TIME_LIMIT_LOCAL - 45:
            break
        
        if isinstance(cand[0], str):
            continue
        
        cx, cy, r = cand
        
        # Local search with small perturbations
        for _ in range(5):
            if time.time() - _START_TIME_LOCAL > TIME_LIMIT_LOCAL - 40:
                break
            
            dx = np.random.uniform(-0.03, 0.03)
            dy = np.random.uniform(-0.03, 0.03)
            dr = np.random.uniform(-0.02, 0.02)
            
            new_cx = max(0.05, min(0.4, cx + dx))
            new_cy = max(0.02, min(0.35, cy + dy))
            new_r = max(0.08, min(0.28, r + dr))
            
            if new_cx * new_cx + new_cy * new_cy + new_r * new_r >= 0.25:
                continue
            
            new_cell = _make_cell_from_rods(size, new_cx, new_cy, new_r)
            new_score = _surrogate_evaluate(new_cell, n_g=49, n_k=12)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
                cx, cy, r = new_cx, new_cy, new_r
    
    # If we have time, do a more careful evaluation with more plane waves
    if best_cell is not None and time.time() - _START_TIME_LOCAL < TIME_LIMIT_LOCAL - 30:
        # Verify with more plane waves
        verify_score = _surrogate_evaluate(best_cell, n_g=100, n_k=20)
        # The verify score might be different; trust the surrogate for ranking
        # but use verify for final decision
    
    # Final: if we have time, try to improve with even more careful search
    if best_cell is not None and time.time() - _START_TIME_LOCAL < TIME_LIMIT_LOCAL - 20:
        # Try a few more random candidates
        for _ in range(5):
            if time.time() - _START_TIME_LOCAL > TIME_LIMIT_LOCAL - 15:
                break
            cx = np.random.uniform(0.1, 0.35)
            cy = np.random.uniform(0.05, 0.3)
            r = np.random.uniform(0.12, 0.22)
            if cx * cx + cy * cy + r * r >= 0.25:
                continue
            cell = _make_cell_from_rods(size, cx, cy, r)
            score = _surrogate_evaluate(cell, n_g=64, n_k=15)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
    
    if best_cell is None:
        # Fallback: single centered rod
        best_cell = _make_cell_single_rod(size, 0.2)
    
    # Convert to nested list
    result = best_cell.astype(int).tolist()
    
    # Ensure symmetry (the construction should guarantee this, but verify)
    # The 4-fold symmetric construction guarantees:
    # cell[i][j] == cell[size-1-i][size-1-j] (180° rotation)
    # cell[i][j] == cell[j][i] (transpose) - this requires the rod placement to be symmetric
    # Actually, let's verify and fix if needed
    
    arr = np.array(result)
    # Check transpose symmetry
    if not np.array_equal(arr, arr.T):
        # Force symmetry by averaging
        arr = ((arr + arr.T) // 2).astype(int)
    # Check 180° rotation
    arr_rot = arr[::-1, ::-1]
    if not np.array_equal(arr, arr_rot):
        arr = ((arr + arr_rot) // 2).astype(int)
    
    result = arr.astype(int).tolist()
    
    return result
# EVOLVE-BLOCK-END
