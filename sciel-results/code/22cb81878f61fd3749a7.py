"""Design a symmetric 32x32 photonic crystal unit cell for a large TM band gap (bands 4-5).

Approach: continuous permittivity field via D4-symmetric Fourier modes,
optimized by gradient ascent on the gap-midgap ratio using Hellmann-Feynman
sensitivity of the plane-wave eigenvalues.
"""
import numpy as np

def _d4_symmetrize_2d(arr):
    """Enforce D4 symmetry about the center."""
    n = arr.shape[0]
    rot = arr[::-1, ::-1].copy()
    mir_x = arr[:, ::-1].copy()
    mir_y = arr[::-1, :].copy()
    trans = arr.T.copy()
    sym = (arr + rot + mir_x + mir_y + trans + rot.T + mir_x.T + mir_y.T) / 8.0
    return sym

def _fourier_coeffs_eps_inv(v_field, size, nmax):
    """Compute Fourier coefficients of 1/eps = 1/(1+11*v) on a size x size grid."""
    inv_eps = 1.0 / (1.0 + 11.0 * v_field)
    fft2 = np.fft.fft2(inv_eps)
    coeffs = {}
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            coeffs[(m, n)] = fft2[m % size, n % size] / (size * size)
    return coeffs

def _band_structure(coeffs, kx, ky, nmax):
    """Solve the TM plane-wave eigenvalue problem at wave vector (kx, ky)."""
    G_list = []
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            G_list.append((m, n))
    N = len(G_list)
    G_idx = {g: i for i, g in enumerate(G_list)}
    
    M = np.zeros((N, N), dtype=np.complex128)
    for i, Gi in enumerate(G_list):
        for j, Gj in enumerate(G_list):
            q = (Gi[0] - Gj[0], Gi[1] - Gj[1])
            if abs(q[0]) > nmax or abs(q[1]) > nmax:
                continue
            c = coeffs.get(q, 0.0)
            if c == 0:
                continue
            kxj = kx + Gj[0]
            kyj = ky + Gj[1]
            M[i, j] = c * (kxj * kxj + kyj * kyj)
    
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals.real)
    vals = vals[order]
    vecs = vecs[:, order]
    return vals.real, vecs, G_list, G_idx

def _irreducible_kpoints(n=30):
    """Generate n wave vectors in the irreducible BZ: 0 <= kx <= ky, kx+ky <= 0.5."""
    kpts = []
    for t in np.linspace(0, 0.5, 8):
        kpts.append((0.0, t))
    for t in np.linspace(0, 0.25, 6):
        kpts.append((t, 0.5 - t))
    for t in np.linspace(0, 0.25, 6):
        kpts.append((t, t))
    rng = np.random.RandomState(42)
    while len(kpts) < n:
        kx = rng.uniform(0, 0.25)
        ky = rng.uniform(kx, 0.5 - kx)
        if kx <= ky and kx + ky <= 0.5:
            kpts.append((kx, ky))
    return kpts[:n]

def _hf_gradient_band(coeffs, kx, ky, nmax, band_idx, vecs, G_list, G_idx):
    """Compute Hellmann-Feynman gradient of eigenvalue w.r.t. Fourier coeffs of 1/eps."""
    N = len(G_list)
    c = vecs[:, band_idx]
    grad_q = np.zeros((2 * nmax + 1, 2 * nmax + 1), dtype=np.complex128)
    
    for i, Gi in enumerate(G_list):
        for j, Gj in enumerate(G_list):
            q = (Gi[0] - Gj[0], Gi[1] - Gj[1])
            if abs(q[0]) > nmax or abs(q[1]) > nmax:
                continue
            kxj = kx + Gj[0]
            kyj = ky + Gj[1]
            grad_q[q[0] + nmax, q[1] + nmax] += np.conj(c[i]) * (kxj * kxj + kyj * kyj) * c[j]
    
    return grad_q

def design(problem):
    size = problem["pixels"]
    nmax = 8  # 17x17 = 289 plane waves
    
    # D4-symmetric Fourier modes
    mode_defs = [
        (0, 0),
        (1, 0), (0, 1),
        (1, 1),
        (2, 0), (0, 2),
        (2, 1), (1, 2),
        (2, 2),
        (3, 0), (0, 3),
        (3, 1), (1, 3),
        (3, 2), (2, 3),
        (3, 3),
        (4, 0), (0, 4),
        (4, 1), (1, 4),
        (4, 2), (2, 4),
        (4, 3), (3, 4),
        (4, 4),
    ]
    mode_groups = []
    seen = set()
    for m, n in mode_defs:
        if (m, n) in seen:
            continue
        seen.add((m, n))
        seen.add((n, m))
        if m == n:
            mode_groups.append([(m, n)])
        else:
            mode_groups.append([(m, n), (n, m)])
    
    n_modes = len(mode_groups)
    
    coefs_opt = np.zeros(n_modes)
    coefs_opt[0] = 0.5
    
    i_arr = np.arange(size)
    x_grid = (i_arr + 0.5) / size - 0.5
    X, Y = np.meshgrid(x_grid, x_grid, indexing='ij')
    
    mode_vals = []
    for group in mode_groups:
        val = np.zeros((size, size))
        for (m, n) in group:
            val += np.cos(2 * np.pi * m * X) * np.cos(2 * np.pi * n * Y)
        mode_vals.append(val)
    mode_vals = np.array(mode_vals)
    
    kpts = _irreducible_kpoints(30)
    
    def compute_score_and_grad(coefs):
        v = np.sum(coefs[:, None, None] * mode_vals, axis=0)
        v_clipped = np.clip(v, 0.0, 1.0)
        coeffs = _fourier_coeffs_eps_inv(v_clipped, size, nmax)
        
        omega4_max = -np.inf
        omega5_min = np.inf
        k4 = None
        k5 = None
        vecs4 = None
        vecs5 = None
        G_list = None
        G_idx = None
        
        for kx, ky in kpts:
            vals, vecs, G_list, G_idx = _band_structure(coeffs, kx, ky, nmax)
            if len(vals) < 5:
                continue
            w4 = np.sqrt(max(vals[3], 0))
            w5 = np.sqrt(max(vals[4], 0))
            if w4 > omega4_max:
                omega4_max = w4
                k4 = (kx, ky)
                vecs4 = vecs
            if w5 < omega5_min:
                omega5_min = w5
                k5 = (kx, ky)
                vecs5 = vecs
        
        if omega4_max < 0 or omega5_min < 0:
            return -1.0, np.zeros(n_modes)
        
        gap = omega5_min - omega4_max
        midgap = (omega5_min + omega4_max) / 2
        if midgap == 0:
            score = -1.0
        else:
            score = 2 * gap / midgap
        
        # HF gradient
        grad_1over_eps_q4 = _hf_gradient_band(coeffs, k4[0], k4[1], nmax, 3, vecs4, G_list, G_idx)
        grad_1over_eps_q5 = _hf_gradient_band(coeffs, k5[0], k5[1], nmax, 4, vecs5, G_list, G_idx)
        
        # Convert q-space gradient to pixel space
        # The q-space gradient is on a grid of (2*nmax+1) x (2*nmax+1), centered at (nmax, nmax)
        # We need to map this to the full size x size FFT grid
        grad_pixel4 = np.zeros((size, size), dtype=np.complex128)
        grad_pixel5 = np.zeros((size, size), dtype=np.complex128)
        for i in range(2 * nmax + 1):
            for j in range(2 * nmax + 1):
                m = i - nmax
                n = j - nmax
                grad_pixel4[m % size, n % size] = grad_1over_eps_q4[i, j]
                grad_pixel5[m % size, n % size] = grad_1over_eps_q5[i, j]
        
        # Inverse FFT to get pixel-space gradient
        grad_pixel4 = np.fft.ifft2(grad_pixel4).real
        grad_pixel5 = np.fft.ifft2(grad_pixel5).real
        
        # d(score)/d(1/eps) at each pixel
        # score = 2*(w5 - w4)/(w5 + w4)
        # d(score)/d(w5) = 2*w4/(w5+w4)^2
        # d(score)/d(w4) = -2*w5/(w5+w4)^2
        # d(w)/d(1/eps) = (1/(2*w)) * d(w^2)/d(1/eps)
        # So d(score)/d(1/eps) = d(score)/d(w5) * d(w5)/d(1/eps) + d(score)/d(w4) * d(w4)/d(1/eps)
        # = [2*w4/(w5+w4)^2] * [1/(2*w5)] * d(w5^2)/d(1/eps) + [-2*w5/(w5+w4)^2] * [1/(2*w4)] * d(w4^2)/d(1/eps)
        # = [w4/(w5*(w5+w4)^2)] * d(w5^2)/d(1/eps) - [w5/(w4*(w5+w4)^2)] * d(w4^2)/d(1/eps)
        
        if omega4_max == 0 or omega5_min == 0:
            grad_pixel = np.zeros((size, size))
        else:
            coeff5 = omega4_max / (omega5_min * (omega5_min + omega4_max) ** 2)
            coeff4 = -omega5_min / (omega4_max * (omega5_min + omega4_max) ** 2)
            grad_pixel = coeff5 * grad_pixel5 + coeff4 * grad_pixel4
        
        # Project onto mode basis
        grad_coefs = np.zeros(n_modes)
        for i in range(n_modes):
            grad_coefs[i] = np.sum(grad_pixel * mode_vals[i])
        
        return score, grad_coefs
    
    best_score = -np.inf
    best_coefs = coefs_opt.copy()
    step_size = 0.05
    
    for it in range(40):
        score, grad = compute_score_and_grad(coefs_opt)
        if score > best_score:
            best_score = score
            best_coefs = coefs_opt.copy()
        
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-8:
            grad_norm = grad / gnorm
            coefs_opt = coefs_opt + step_size * grad_norm
            coefs_opt[0] = np.clip(coefs_opt[0], 0.0, 1.0)
            for i in range(1, n_modes):
                coefs_opt[i] = np.clip(coefs_opt[i], -0.3, 0.3)
        
        if it % 10 == 9:
            step_size *= 0.8
    
    v_final = np.sum(best_coefs[:, None, None] * mode_vals, axis=0)
    v_final = np.clip(v_final, 0.0, 1.0)
    cell = (v_final > 0.5).astype(int)
    cell = _d4_symmetrize_2d(cell.astype(float)).astype(int)
    cell = (cell > 0.5).astype(int)
    return cell.tolist()
