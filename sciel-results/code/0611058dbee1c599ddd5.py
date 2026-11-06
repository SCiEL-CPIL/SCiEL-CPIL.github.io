"""Photonic crystal band-gap optimizer with internal PWE solver."""
import numpy as np
from scipy.linalg import eigh
import time

def _build_pwe_matrices(cell, n_pw=17, n_bands=6):
    """Build PWE matrices for TM mode. cell is 32x32 binary array."""
    size = cell.shape[0]
    eps = 1.0 + 11.0 * cell.astype(float)
    
    # Fourier coefficients of 1/eps
    inv_eps = 1.0 / eps
    # Compute FFT-based Fourier coefficients
    N = 2 * n_pw + 1
    # Pad to next power of 2 for FFT efficiency
    N_fft = 1
    while N_fft < N:
        N_fft *= 2
    
    # Compute 2D FFT of inv_eps
    # We need Fourier coefficients G where G[m,n] = (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i + n*j)/size)
    # Using FFT: F = FFT2(inv_eps), then G[m,n] = F[m mod size, n mod size] / size^2
    F = np.fft.fft2(inv_eps, s=(N_fft, N_fft))
    F = F / (size * size)
    
    # Build the G index array: indices from -n_pw to n_pw
    g_idx = np.arange(-n_pw, n_pw + 1)
    G = np.zeros((N, N), dtype=complex)
    for im, m in enumerate(g_idx):
        for in_, n in enumerate(g_idx):
            # Wrap to [0, size)
            G[im, in_] = F[m % size, n % size]
    
    # Build A and B matrices
    # A[G1, G2] = (k+G1)^2 * delta(G1,G2)  (identity in G-space scaled by (k+G)^2)
    # B[G1, G2] = G[G1-G2]
    # Eigenvalue problem: A * c = omega^2 * B * c
    
    # For a given k, we solve: (k+G)^2 * c_G - omega^2 * sum_{G'} G[G-G'] c_{G'} = 0
    # i.e., sum_{G'} [(k+G)^2 delta(G,G') - omega^2 G[G-G']] c_{G'} = 0
    
    # We'll build this as a generalized eigenvalue problem:
    # M * c = omega^2 * N_mat * c
    # where M[G,G'] = G[G-G'] (the inv_eps convolution)
    # and N_mat[G,G'] = (k+G)^2 delta(G,G')
    
    # Actually, the standard PWE formulation for TM is:
    # sum_{G'} (k+G)^2 delta(G,G') c_{G'} = omega^2 sum_{G'} G[G-G'] c_{G'}
    # So: N_mat * c = omega^2 * M * c
    # where N_mat is diagonal with (k+G)^2 and M is the circulant matrix of G
    
    # M is circulant, so M[G,G'] = G[(G-G') mod N]
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            diff_m = g_idx[i] - g_idx[j]
            diff_n = 0  # This is 2D, need to handle properly
    # Actually for 2D, M is a 2D circulant. Let me restructure.
    
    # For 2D: the matrix is indexed by pairs (m,n). M[(m1,n1),(m2,n2)] = G[(m1-m2),(n1-n2)]
    # This is a 2D circulant matrix.
    
    # Let's build it properly:
    M2d = np.zeros((N, N, N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    dm = g_idx[i] - g_idx[k]
                    dn = g_idx[j] - g_idx[l]
                    M2d[i, j, k, l] = G[i, j] if (dm == 0 and dn == 0) else 0
    # That's wrong. Let me think again.
    
    # M[(m1,n1),(m2,n2)] = G[m1-m2, n1-n2]
    # where G is indexed by (m,n) with m,n in [-n_pw, n_pw]
    
    M_mat = np.zeros((N*N, N*N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    dm = g_idx[i] - g_idx[k]
                    dn = g_idx[j] - g_idx[l]
                    if -n_pw <= dm <= n_pw and -n_pw <= dn <= n_pw:
                        M_mat[i*N+j, k*N+l] = G[i + n_pw, j + n_pw]  # This is wrong
    # I'm overcomplicating. Let me use a cleaner approach.
    
    # G[dm, dn] for dm,dn in range(-n_pw, n_pw+1)
    # M[(m1,n1),(m2,n2)] = G[m1-m2, n1-n2]
    
    M_mat = np.zeros((N*N, N*N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    dm = g_idx[i] - g_idx[k]
                    dn = g_idx[j] - g_idx[l]
                    if -n_pw <= dm <= n_pw and -n_pw <= dn <= n_pw:
                        M_mat[i*N+j, k*N+l] = G[i + n_pw, j + n_pw]
    # Still wrong. G is indexed by [0..N-1] corresponding to g_idx[0..N-1]
    # G[a, b] = Fourier coeff for (g_idx[a], g_idx[b])
    # M[(i,j),(k,l)] = G[dm_idx, dn_idx] where dm = g_idx[i]-g_idx[k], dn = g_idx[j]-g_idx[l]
    
    M_mat = np.zeros((N*N, N*N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    dm = g_idx[i] - g_idx[k]
                    dn = g_idx[j] - g_idx[l]
                    if -n_pw <= dm <= n_pw and -n_pw <= dn <= n_pw:
                        dm_idx = dm + n_pw
                        dn_idx = dn + n_pw
                        M_mat[i*N+j, k*N+l] = G[dm_idx, dn_idx]
    
    return M_mat, g_idx, n_pw

def _compute_bands(cell, kx, ky, n_pw=17, n_bands=6):
    """Compute TM bands at wavevector (kx, ky) in units of 2*pi/a."""
    size = cell.shape[0]
    eps = 1.0 + 11.0 * cell.astype(float)
    inv_eps = 1.0 / eps
    
    N = 2 * n_pw + 1
    g_idx = np.arange(-n_pw, n_pw + 1)
    
    # Fourier coefficients of inv_eps
    N_fft = 1
    while N_fft < N:
        N_fft *= 2
    F = np.fft.fft2(inv_eps, s=(N_fft, N_fft))
    F = F / (size * size)
    
    G = np.zeros((N, N), dtype=complex)
    for im, m in enumerate(g_idx):
        for in_, n in enumerate(g_idx):
            G[im, in_] = F[m % size, n % size]
    
    # Build M matrix: M[(i,j),(k,l)] = G[g_idx[i]-g_idx[k], g_idx[j]-g_idx[l]]
    M_mat = np.zeros((N*N, N*N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    dm = g_idx[i] - g_idx[k]
                    dn = g_idx[j] - g_idx[l]
                    if -n_pw <= dm <= n_pw and -n_pw <= dn <= n_pw:
                        M_mat[i*N+j, k*N+l] = G[dm + n_pw, dn + n_pw]
    
    # N_mat is diagonal: N_mat[(i,j),(i,j)] = (kx+g_idx[i])^2 + (ky+g_idx[j])^2
    diag = np.zeros(N*N, dtype=float)
    for i in range(N):
        for j in range(N):
            diag[i*N+j] = (kx + g_idx[i])**2 + (ky + g_idx[j])**2
    
    # Solve: N_mat * c = omega^2 * M_mat * c
    # => M_mat^{-1} * N_mat * c = omega^2 * c
    # Generalized eigenvalue: N_mat * c = omega^2 * M_mat * c
    
    # Use scipy to solve generalized eigenvalue problem
    # M_mat is Hermitian (since G is from FFT of real function, G[-m,-n] = conj(G[m,n]))
    # N_mat is real diagonal, positive
    
    try:
        eigvals, eigvecs = eigh(M_mat, N_mat, subset_by_index=[0, n_bands-1])
        # eigvals are omega^2, so omega = sqrt(eigvals)
        omega_sq = eigvals.real
        omega = np.sqrt(np.maximum(omega_sq, 0))
        return omega
    except Exception:
        return None

def _evaluate_gap(cell, n_pw=13, n_bands=6, n_kpts=8):
    """Evaluate the gap-midgap ratio between bands 4 and 5."""
    # Irreducible BZ for square lattice: kx in [0, 0.5], ky in [0, kx] (or similar)
    # Key points: Gamma(0,0), M(0.5,0), X(0.5,0.5), and edges
    kpts = [
        (0.0, 0.0),      # Gamma
        (0.25, 0.0),
        (0.5, 0.0),      # M
        (0.25, 0.25),
        (0.5, 0.25),
        (0.5, 0.5),      # X
        (0.125, 0.0),
        (0.375, 0.0),
    ]
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx, ky in kpts:
        bands = _compute_bands(cell, kx, ky, n_pw=n_pw, n_bands=n_bands)
        if bands is None or len(bands) < n_bands:
            continue
        # bands[0] is lowest, bands[3] is 4th band, bands[4] is 5th band
        if len(bands) >= 5:
            max_omega4 = max(max_omega4, bands[3])
            min_omega5 = min(min_omega5, bands[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return 0.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    if midgap <= 0:
        return 0.0
    
    return gap / midgap

def _make_cell(size, radius, shape_param, center_offset=0.0):
    """Create a symmetric cell with deformed circular rod."""
    cell = np.zeros((size, size), dtype=int)
    cx = size / 2.0
    cy = size / 2.0
    
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) / size - 0.5 + center_offset
            dy = (j + 0.5) / size - 0.5 + center_offset
            r = np.sqrt(dx*dx + dy*dy)
            # Deformed radius: r_eff = radius * (1 + shape_param * cos(4*theta))
            if r > 1e-10:
                theta = np.arctan2(dy, dx)
                r_eff = radius * (1.0 + shape_param * np.cos(4 * theta))
            else:
                r_eff = radius
            if r < r_eff:
                cell[i, j] = 1
    
    # Enforce C4v symmetry
    cell = np.maximum(cell, cell[::-1, :])
    cell = np.maximum(cell, cell[:, ::-1])
    cell = np.maximum(cell, cell.T)
    cell = np.maximum(cell, cell[::-1, ::-1].T)
    # Actually, for proper symmetry: cell[i,j] = cell[size-1-i, j] = cell[i, size-1-j] = cell[j, i]
    # Let's enforce by averaging (but it's binary, so use max or min)
    # Better: enforce by requiring all four symmetric positions to agree
    # For binary, we can set cell[i,j] = 1 if any of the symmetric positions is 1
    # But that may break the intended geometry. Let's just ensure symmetry by construction.
    
    # Re-do with proper symmetry enforcement
    cell2 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) / size - 0.5
            dy = (j + 0.5) / size - 0.5
            r = np.sqrt(dx*dx + dy*dy)
            if r > 1e-10:
                theta = np.arctan2(dy, dx)
                r_eff = radius * (1.0 + shape_param * np.cos(4 * theta))
            else:
                r_eff = radius
            if r < r_eff:
                cell2[i, j] = 1
    
    # Enforce symmetry: cell[i,j] must equal cell[size-1-i, j], cell[i, size-1-j], cell[j, i]
    # Since our formula is already C4v symmetric (cos(4*theta) is invariant under 90° rotation and reflection),
    # the cell should already be symmetric. But pixel discretization may break it slightly.
    # Enforce by taking the AND of all symmetric copies (conservative) or OR (aggressive)
    # For a rod, AND is safer (keeps only pixels that are definitely inside)
    cell_sym = cell2.copy()
    cell_sym = np.minimum(cell_sym, cell2[::-1, :])
    cell_sym = np.minimum(cell_sym, cell2[:, ::-1])
    cell_sym = np.minimum(cell_sym, cell2.T)
    cell_sym = np.minimum(cell_sym, cell2[::-1, ::-1].T)
    
    return cell_sym.tolist()

def design(problem):
    """Optimize photonic crystal unit cell for maximum TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    deadline = time.time() + 210  # Safety margin before 230s kill
    
    # Start with the seed design
    best_cell = None
    best_score = -1.0
    
    # Initial candidate: seed (radius 0.2)
    cell_seed = _make_cell(size, 0.2, 0.0)
    best_cell = cell_seed
    best_score = _evaluate_gap(np.array(cell_seed), n_pw=13, n_bands=6, n_kpts=8)
    
    # Grid search over radius and shape parameter
    radii = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    shapes = [0.0, 0.05, 0.10, 0.15, 0.20]
    
    for r in radii:
        if time.time() > deadline - 30:
            break
        for s in shapes:
            if time.time() > deadline - 15:
                break
            cell = _make_cell(size, r, s)
            score = _evaluate_gap(np.array(cell), n_pw=13, n_bands=6, n_kpts=8)
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # If we have time, do a finer local search around the best
    if time.time() < deadline - 60:
        # Find the best r and s from the coarse grid
        best_r, best_s = 0.2, 0.0
        best_coarse = -1.0
        for r in radii:
            for s in shapes:
                cell = _make_cell(size, r, s)
                score = _evaluate_gap(np.array(cell), n_pw=13, n_bands=6, n_kpts=8)
                if score > best_coarse:
                    best_coarse = score
                    best_r, best_s = r, s
        
        # Fine search around best
        fine_radii = np.linspace(max(0.1, best_r - 0.05), min(0.4, best_r + 0.05), 5)
        fine_shapes = np.linspace(max(-0.1, best_s - 0.05), min(0.3, best_s + 0.05), 5)
        
        for r in fine_radii:
            if time.time() > deadline - 30:
                break
            for s in fine_shapes:
                if time.time() > deadline - 15:
                    break
                cell = _make_cell(size, float(r), float(s))
                score = _evaluate_gap(np.array(cell), n_pw=15, n_bands=6, n_kpts=10)
                if score > best_score:
                    best_score = score
                    best_cell = cell
    
    # Return the best cell found
    if best_cell is None:
        # Fallback to seed
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    return best_cell
