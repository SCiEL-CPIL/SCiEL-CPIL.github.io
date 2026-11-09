"""Optimized photonic crystal design for TM band gap maximization."""
import time
import numpy as np
from scipy.linalg import eigh

def _fourier_coefficients(cell, N=17):
    """Compute exact Fourier coefficients of the pixel map for N x N plane waves.
    N=17 gives 17^2 = 289 plane waves."""
    size = len(cell)
    eps = np.array(cell, dtype=float)
    # We need Fourier coefficients of 1/eps for TM mode
    # eps_inv[k] = (1/eps) * exp(-i k.r) integrated over cell
    # For pixel map, this is a sum over pixels
    kmax = N - 1  # -8 to 8 for N=17
    kx = np.arange(-kmax, kmax + 1)
    ky = np.arange(-kmax, kmax + 1)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.stack([KX.ravel(), KY.ravel()], axis=-1)  # (289, 2)
    
    # Pixel centers
    px = (np.arange(size) + 0.5) / size - 0.5
    py = (np.arange(size) + 0.5) / size - 0.5
    PX, PY = np.meshgrid(px, py, indexing='ij')
    PX = PX.ravel()
    PY = PY.ravel()
    EPS_FLAT = eps.ravel()
    
    # Fourier coefficient: f[k] = sum_j eps_inv[j] * exp(-i 2pi (kx*px_j + ky*py_j)) * (1/size^2)
    # Actually for the Galerkin method, we need the inverse dielectric
    # Let's compute 1/eps for each pixel
    eps_inv = 1.0 / EPS_FLAT  # eps is 0 or 1, but 0 means air (eps=1), 1 means dielectric (eps=12)
    # Wait: 0 = air (eps=1), 1 = dielectric (eps=12)
    eps_actual = np.where(EPS_FLAT > 0.5, 12.0, 1.0)
    eps_inv = 1.0 / eps_actual
    
    # Compute Fourier coefficients
    # f[k] = (1/N_pixels) * sum_j eps_inv[j] * exp(-i 2pi (kx*px_j + ky*py_j))
    phase = 2 * np.pi * (K[:, 0:1] * PX[None, :] + K[:, 1:2] * PY[None, :])
    # This is (289, 1024) - might be large but manageable
    exp_phase = np.exp(-1j * phase)
    fourier_coeffs = (exp_phase @ eps_inv) / (size * size)
    
    return K, fourier_coeffs

def _solve_tm_bands(K, fourier_coeffs, q, n_bands=6):
    """Solve TM band structure at wave vector q using plane-wave Galerkin method.
    TM mode: eigenvalue problem involves 1/eps * (|k+K|^2) and coupling via Fourier coeffs of 1/eps.
    For TM: (k+K)^2 / eps * E = omega^2 E
    In plane wave basis: sum_K' [delta_KK' |k+K|^2/eps0 + (1-eps0/eps_avg)*V_K-K'] E_K' = omega^2 E_K
    Actually the standard formulation:
    For TM: (k+K)^2 E_K / eps(r) = omega^2 E_K
    Discretize: sum_K' [delta_KK' (k+K)^2/eps_avg + V_K-K'] E_K' = omega^2 E_K
    where V is related to Fourier coefficients of (1/eps - 1/eps_avg)
    """
    # q is (qx, qy) in units of 2pi/a
    n_pw = len(K)
    
    # Average dielectric
    # From Fourier coefficient at K=0
    idx0 = np.argmin(np.sum(K**2, axis=1))
    eps_avg_inv = fourier_coeffs[idx0].real
    eps_avg = 1.0 / eps_avg_inv
    
    # Build the matrix
    # For TM: M[K,K'] = delta_KK' * |k+K|^2 / eps_avg + (1/eps - 1/eps_avg)_Fourier[K-K']
    # Actually the standard form is:
    # sum_K' [delta_KK' |k+K|^2 + (1/eps - 1/eps_avg)_{K-K'} |k+K|^2] E_K' / eps_avg = omega^2 E_K
    # Let me use the more standard formulation:
    # The eigenvalue problem for TM is:
    # sum_K' [delta_KK' |k+K|^2/eps0 + V_K-K'] E_K' = omega^2 E_K
    # where V_K-K' = (1/eps)_{K-K'} * |k+K|^2 ... no this isn't quite right.
    
    # Let me use the formulation from Joannopoulos et al.:
    # For TM: sum_K' delta_KK' |k+K|^2 E_K' / eps_avg + sum_K' V_K-K' |k+K|^2 E_K' / eps_avg = omega^2 E_K
    # where V_K = Fourier coefficient of (1/eps - 1/eps_avg)
    # So: M_KK' = |k+K|^2/eps_avg * (delta_KK' + V_K-K')
    # But V_0 = 0 by definition, so M_KK = |k+K|^2/eps_avg
    
    # Fourier coefficients of (1/eps - 1/eps_avg)
    V = fourier_coeffs.copy()
    V[idx0] = 0.0  # V_0 = 0
    
    # Build matrix
    M = np.zeros((n_pw, n_pw), dtype=complex)
    for i in range(n_pw):
        k_plus_K = q + K[i]
        kp2 = k_plus_K[0]**2 + k_plus_K[1]**2
        for j in range(n_pw):
            dK = K[i] - K[j]
            # Find index of dK in K array
            idx_match = np.argmin(np.sum((K - dK[None, :])**2, axis=1))
            if np.allclose(K[idx_match], dK, atol=1e-10):
                V_dK = V[idx_match]
            else:
                V_dK = 0.0
            M[i, j] = kp2 / eps_avg * (1.0 + V_dK)
    
    # Solve eigenvalue problem
    # M is Hermitian (should be, up to numerical error)
    M = (M + M.conj().T) / 2.0
    eigvals, eigvecs = eigh(M)
    eigvals = np.real(eigvals)
    
    return eigvals[:n_bands]

def _compute_score(cell):
    """Compute the gap-midgap ratio for bands 4 and 5."""
    size = len(cell)
    K, fourier_coeffs = _fourier_coefficients(cell, N=17)
    
    # Define 30 wave vectors in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= qx <= qy <= 1 (in units of 2pi/a)
    # Key points: Gamma(0,0), M(0.5,0.5), X(0.5,0), and lines between them
    kpoints = []
    # Along Gamma to X: (t, 0) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 7):
        kpoints.append((t, 0.0))
    # Along X to M: (0.5, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 7):
        kpoints.append((0.5, t))
    # Along M to Gamma: (t, t) for t in [0.5, 0] -> (t, t) for t in [0, 0.5]
    for t in np.linspace(0, 0.5, 7):
        kpoints.append((t, t))
    # Along Gamma to M: already covered
    # Add some interior points
    kpoints.append((0.25, 0.1))
    kpoints.append((0.3, 0.15))
    kpoints.append((0.4, 0.2))
    kpoints.append((0.35, 0.35))
    kpoints.append((0.2, 0.2))
    # Trim to 30
    kpoints = kpoints[:30]
    
    omega4_max = -np.inf
    omega5_min = np.inf
    
    for qx, qy in kpoints:
        q = np.array([qx, qy])
        eigvals = _solve_tm_bands(K, fourier_coeffs, q, n_bands=6)
        if len(eigvals) >= 5:
            omega4_max = max(omega4_max, eigvals[3])  # 4th band (0-indexed: 3)
            omega5_min = min(omega5_min, eigvals[4])  # 5th band (0-indexed: 4)
    
    if omega5_min <= 0 or omega4_max < 0:
        return -1.0
    
    gap = omega5_min - omega4_max
    midgap = (omega5_min + omega4_max) / 2.0
    if midgap <= 0:
        return -1.0
    
    return 2.0 * gap / midgap

def _apply_symmetry(cell):
    """Apply C4v symmetry to a 32x32 cell.
    Symmetries: transpose, row reversal, column reversal, and combinations."""
    size = len(cell)
    cell = np.array(cell, dtype=float)
    
    # Average over all 8 symmetry operations
    # 1. Identity
    c1 = cell
    # 2. Transpose
    c2 = cell.T
    # 3. Row reversal (flip vertically)
    c3 = cell[::-1, :]
    # 4. Column reversal (flip horizontally)
    c4 = cell[:, ::-1]
    # 5. Transpose + row rev = flip along anti-diagonal
    c5 = cell.T[::-1, :]
    # 6. Transpose + col rev = flip along main diagonal... wait
    # Let me be more careful.
    # The 8 symmetries of the square:
    # E, C4, C4^2, C4^3, sigma_v(x), sigma_v(y), sigma_d(1), sigma_d(2)
    
    # For a 2D array indexed [i,j] where i=row, j=col:
    # E: [i,j]
    # C4 (90 deg CCW): [j, size-1-i]
    # C4^2 (180 deg): [size-1-i, size-1-j]
    # C4^3 (270 deg CCW): [size-1-j, i]
    # sigma_v (vertical axis, x->-x): [i, size-1-j]
    # sigma_h (horizontal axis, y->-y): [size-1-i, j]
    # sigma_d1 (diagonal x=y): [j, i] (transpose)
    # sigma_d2 (anti-diagonal): [size-1-j, size-1-i]
    
    symmetries = [
        lambda c: c,
        lambda c: c[:, ::-1],           # sigma_v
        lambda c: c[::-1, :],           # sigma_h
        lambda c: c[::-1, ::-1],        # C4^2
        lambda c: c.T,                  # sigma_d1
        lambda c: c[:, ::-1].T,         # C4 (90 deg)
        lambda c: c[::-1, :].T,         # C4^3
        lambda c: c[::-1, ::-1].T,      # sigma_d2
    ]
    
    averaged = np.zeros_like(cell)
    for sym in symmetries:
        averaged += sym(cell)
    averaged /= 8.0
    
    # Threshold to 0/1
    result = (averaged > 0.5).astype(int)
    return result.tolist()

def design(problem):
    """Design a 32x32 photonic crystal unit cell maximizing TM band 4-5 gap."""
    start_time = time.time()
    deadline = start_time + 220.0  # Leave 10s margin
    
    size = problem["pixels"]  # 32
    
    # Start with a good initial guess: a dielectric disk with some structure
    # Try a ring-like structure which often gives better gaps
    cell = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            # Ring: dielectric where 0.15 < r < 0.35
            if 0.15 < r < 0.35:
                row.append(1)
            else:
                row.append(0)
        cell.append(row)
    
    # Apply symmetry
    cell = _apply_symmetry(cell)
    
    # Evaluate initial score
    best_cell = [row[:] for row in cell]
    best_score = _compute_score(cell)
    
    # Local search: flip pixels in the fundamental domain (1/8 of the cell)
    # Fundamental domain: i >= j, i + j <= size-1 (upper right triangle of the octant)
    # Actually for C4v, the fundamental domain is 1/8 of the cell
    # Let's use: 0 <= j <= i < size, and i + j <= size - 1
    # Simpler: just use the upper-left triangle where i >= j
    # For 32x32, that's 32*33/2 = 528 pixels, but with C4v we need 1/8
    
    # Let's use a simpler approach: iterate over all pixels but only in a representative region
    # and apply symmetry after each flip
    
    np.random.seed(42)
    
    # Get the fundamental domain indices
    # For C4v, we can use pixels where 0 <= j <= i and i + j < size
    # This gives a triangle in the upper-left
    fund_indices = []
    for i in range(size):
        for j in range(size):
            if j <= i and i + j < size:
                fund_indices.append((i, j))
    
    # Shuffle for random order
    np.random.shuffle(fund_indices)
    
    # Hill climbing with occasional random restarts
    current_cell = [row[:] for row in cell]
    current_score = best_score
    
    flip_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Check time
        if time.time() > deadline - 5:
            break
        
        # Pick a random pixel from fundamental domain
        idx = flip_count % len(fund_indices)
        i, j = fund_indices[idx]
        flip_count += 1
        
        # Try flipping this pixel
        old_val = current_cell[i][j]
        current_cell[i][j] = 1 - old_val
        
        # Apply symmetry
        sym_cell = _apply_symmetry(current_cell)
        
        # Evaluate
        score = _compute_score(sym_cell)
        
        if score > current_score:
            current_cell = [row[:] for row in sym_cell]
            current_score = score
            if score > best_score:
                best_score = score
                best_cell = [row[:] for row in sym_cell]
            no_improve_count = 0
        else:
            # Revert
            current_cell[i][j] = old_val
            no_improve_count += 1
        
        # Occasionally do a random restart
        if no_improve_count > 500:
            # Random perturbation
            for _ in range(20):
                ri, rj = np.random.randint(0, size, 2)
                if rj <= ri and ri + rj < size:
                    current_cell[ri][rj] = 1 - current_cell[ri][rj]
            sym_cell = _apply_symmetry(current_cell)
            current_cell = [row[:] for row in sym_cell]
            current_score = _compute_score(current_cell)
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
            no_improve_count = 0
    
    return best_cell
