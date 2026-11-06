"""Photonic-crystal unit cell design: maximize TM 4-5 band gap via surrogate-guided search."""
import time
import math
import numpy as np

# --- Deadline guard: evaluator kills at 230 s wall-clock; we return by ~190 s ---
_DEADLINE = 190.0

def _symmetrize(cell):
    """Enforce 4-fold square symmetry: cell == cell[::-1] rows/cols and cell == cell.T"""
    size = len(cell)
    c = np.array(cell, dtype=np.float64)
    # Average over the 4 symmetry operations to enforce exact symmetry
    c1 = c
    c2 = c[::-1, ::-1]  # 180 deg rotation
    c3 = c.T            # transpose (mirror across main diagonal)
    c4 = c[::-1, ::-1].T  # other diagonal mirror
    c_sym = (c1 + c2 + c3 + c4) / 4.0
    # Threshold to 0/1
    c_sym = (c_sym >= 0.5).astype(int)
    # Verify and fix: enforce exact symmetry by majority vote
    result = np.zeros_like(c_sym)
    for i in range(size):
        for j in range(size):
            vals = [c_sym[i, j], c_sym[size-1-i, size-1-j], c_sym[j, i], c_sym[size-1-j, size-1-i]]
            result[i, j] = 1 if sum(vals) >= 2 else 0
    return result.tolist()


def _compute_gap_surrogate(cell_2d):
    """
    Fast surrogate: compute the TM 4-5 gap using a reduced plane-wave basis
    and a small set of k-points. This is approximate but much faster than
    the full 289-PW evaluator.
    """
    size = len(cell_2d)
    eps_high = 12.0
    eps_low = 1.0
    eps_cell = np.array(cell_2d, dtype=np.float64) * eps_high + (1 - np.array(cell_2d, dtype=np.float64)) * eps_low
    inv_eps = 1.0 / eps_cell

    # Use a smaller PW basis: 15x15 = 225 PWs (close to 289 but faster)
    # Actually let's use 13x13 = 169 for speed
    n_pw = 13
    half = n_pw // 2
    # Reciprocal lattice vectors G = (m, n) in units of 2π
    G_list = []
    for m in range(-half, half + 1):
        for n in range(-half, half + 1):
            G_list.append((m, n))
    n_pw_actual = len(G_list)

    # Fourier coefficients of 1/eps
    # inv_eps[m,n] = (1/N^2) sum_{i,j} inv_eps[i,j] exp(-i 2π (m i/N + n j/N))
    # Use FFT for speed
    inv_eps_2d = np.array(cell_2d, dtype=np.float64) * eps_high + (1 - np.array(cell_2d, dtype=np.float64)) * eps_low
    inv_eps_2d = 1.0 / inv_eps_2d
    fft_inv = np.fft.fft2(inv_eps_2d) / (size * size)
    # fft_inv[m,n] gives coefficient for G=(m,n) with m,n in [-N/2, N/2-1]
    # We need to shift to get proper ordering
    fft_inv_shifted = np.fft.fftshift(fft_inv)

    def get_coeff(m, n):
        """Get Fourier coefficient for G=(m,n)"""
        # Map to FFT index
        im = (m + size) % size
        in_ = (n + size) % size
        return fft_inv[im, in_]

    # k-points in the irreducible BZ (fractional coordinates, in units of 2π)
    # Irreducible BZ of square lattice: 0 <= kx <= ky <= 0.5
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.25),     # interior
        (0.1, 0.1),
        (0.4, 0.2),
        (0.3, 0.3),
        (0.5, 0.25),
        (0.25, 0.0),
        (0.4, 0.4),
    ]

    max_omega_4 = 0.0
    min_omega_5 = float('inf')

    for kx, ky in k_points:
        # Build the TM eigenvalue problem
        # For TM: (G+k)^2 * (1/eps[G'-G]) * c' = omega^2 * c
        # Matrix: A[G,G'] = (G+k)^2 * inv_eps[G'-G]
        # But we need to be careful: the equation is
        # sum_{G'} (G+k)^2 * inv_eps[G'-G] * c_{G'} = omega^2 * c_G
        # Wait, the standard form for TM is:
        # omega^2 c_G = sum_{G'} (G+k)^2 * inv_eps[G'-G] * c_{G'}
        
        # Actually the correct form: for TM polarization,
        # (G+k)^2 * c_G = omega^2 * sum_{G'} eps[G-G'] * c_{G'}
        # which gives omega^2 = (G+k)^2 / <eps> approximately for uniform
        # More precisely, we solve:
        # sum_{G'} eps[G-G'] c_{G'} = (1/omega^2) (G+k)^2 c_G
        # Or equivalently for eigenvalue lambda = 1/omega^2:
        # A c = lambda c where A[G,G'] = (1/eps)[G-G'] * (G+k)^2 delta... 
        # 
        # Let me use the standard formulation:
        # For TM: the eigenvalue problem is
        # sum_{G'} (G+k)^2 * inv_eps[G'-G] * c_{G'} = omega^2 * c_G
        # This is NOT a standard eigenvalue problem because (G+k)^2 depends on G, not G'.
        #
        # Correct formulation:
        # omega^2 * c_G = sum_{G'} (G+k)^2 * inv_eps[G'-G] * c_{G'}
        # This can be written as: D * E * c = omega^2 * c
        # where D is diagonal with D[G,G] = (G+k)^2 and E[G,G'] = inv_eps[G'-G]
        # So we solve: D * E * c = omega^2 * c
        # This is a generalized eigenvalue problem: E * c = (1/omega^2) * D^{-1} * c
        # Or: (D^{-1/2} * E * D^{-1/2}) * (D^{1/2} c) = (1/omega^2) * (D^{1/2} c)
        
        # Build matrices
        kx_2pi = 2 * math.pi * kx
        ky_2pi = 2 * math.pi * ky
        
        D_diag = np.zeros(n_pw_actual)
        for idx, (gm, gn) in enumerate(G_list):
            gx = kx_2pi + 2 * math.pi * gm
            gy = ky_2pi + 2 * math.pi * gn
            D_diag[idx] = gx * gx + gy * gy
        
        # Avoid zero division
        D_diag[D_diag < 1e-12] = 1e-12
        
        # Build E matrix: E[i,j] = inv_eps[G_i - G_j]
        E = np.zeros((n_pw_actual, n_pw_actual), dtype=np.complex128)
        for i, (mi, ni) in enumerate(G_list):
            for j, (mj, nj) in enumerate(G_list):
                dm = mi - mj
                dn = ni - nj
                E[i, j] = get_coeff(dm, dn)
        
        # Form symmetric matrix: S = D^{-1/2} * E * D^{-1/2}
        D_inv_sqrt = 1.0 / np.sqrt(D_diag)
        S = (D_inv_sqrt[:, None] * E * D_inv_sqrt[None, :])
        
        # S should be Hermitian (approximately)
        S = (S + S.conj().T) / 2.0
        
        # Solve eigenvalue problem
        try:
            eigvals, eigvecs = np.linalg.eigh(S)
            # omega^2 = 1 / eigval
            # We want the 4th and 5th smallest omega, i.e., 4th and 5th largest 1/omega^2
            # Sort eigvals in descending order (largest lambda = smallest omega)
            eigvals = np.sort(eigvals)[::-1]
            # omega^2 = 1/lambda, so omega = 1/sqrt(lambda)
            omegas = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
            omegas = np.sort(omegas)
            
            if len(omegas) >= 5:
                w4 = omegas[3]  # 4th band (0-indexed: 3)
                w5 = omegas[4]  # 5th band (0-indexed: 4)
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        except np.linalg.LinAlgError:
            pass

    if min_omega_5 == float('inf'):
        return -1.0
    
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap < 1e-12:
        return -1.0
    return gap / midgap


def _random_symmetric_cell(size, density):
    """Generate a random symmetric cell with given dielectric density."""
    # Generate a fundamental domain and mirror
    # For 4-fold symmetry, the fundamental domain is a quarter of the cell
    # But we need to be careful with the center lines
    cell = np.zeros((size, size), dtype=int)
    # Fill the top-left quadrant and mirror
    half = size // 2
    for i in range(half):
        for j in range(half):
            val = 1 if np.random.random() < density else 0
            cell[i, j] = val
            cell[size-1-i, size-1-j] = val
            cell[j, i] = val
            cell[size-1-j, size-1-i] = val
    # Handle center lines if size is even
    if size % 2 == 0:
        for i in range(half):
            val = 1 if np.random.random() < density else 0
            cell[i, half] = val
            cell[i, size-1-half] = val
            cell[half, i] = val
            cell[size-1-half, i] = val
            # Center 2x2
            cval = 1 if np.random.random() < density else 0
            cell[half, half] = cval
            cell[half, size-1-half] = cval
            cell[size-1-half, half] = cval
            cell[size-1-half, size-1-half] = cval
    else:
        # Odd size: center pixel
        c = size // 2
        val = 1 if np.random.random() < density else 0
        cell[c, c] = val
        # Center row/col
        for i in range(c):
            val = 1 if np.random.random() < density else 0
            cell[i, c] = val
            cell[size-1-i, c] = val
            cell[c, i] = val
            cell[c, size-1-i] = val
    return cell


def _hill_climb(cell, size, max_iters=50, time_limit=60.0):
    """Local search: flip random pixels (maintaining symmetry) and accept if gap improves."""
    start_time = time.time()
    best_cell = np.array(cell, dtype=int)
    best_score = _compute_gap_surrogate(best_cell.tolist())
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        # Pick a random position in the fundamental domain
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        # Flip the pixel and its symmetric partners
        partners = [(i, j), (size-1-i, size-1-j), (j, i), (size-1-j, size-1-i)]
        old_vals = [best_cell[p[0], p[1]] for p in partners]
        for p in partners:
            best_cell[p[0], p[1]] = 1 - best_cell[p[0], p[1]]
        
        new_score = _compute_gap_surrogate(best_cell.tolist())
        
        if new_score > best_score:
            best_score = new_score
        else:
            # Revert
            for idx, p in enumerate(partners):
                best_cell[p[0], p[1]] = old_vals[idx]
    
    return best_cell.tolist(), best_score


def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    np.random.seed(42)
    start_time = time.time()
    
    # Initial diverse set of designs
    candidates = []
    
    # 1. Textbook: centered dielectric disk
    cell_disk = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(1 if x*x + y*y < 0.2**2 else 0)
        cell_disk.append(row)
    candidates.append(cell_disk)
    
    # 2. Air disk in dielectric (inverted)
    cell_inv = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(0 if x*x + y*y < 0.2**2 else 1)
        cell_inv.append(row)
    candidates.append(cell_inv)
    
    # 3. Square hole
    cell_sq = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            row.append(0 if abs(x) < 0.15 and abs(y) < 0.15 else 1)
        cell_sq.append(row)
    candidates.append(cell_sq)
    
    # 4. Four small disks at corners of the cell
    cell_4disk = []
    for i in range(size):
        row = []
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            val = 0
            for cx, cy in [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]:
                if (x-cx)**2 + (y-cy)**2 < 0.1**2:
                    val = 1
            row.append(val)
        cell_4disk.append(row)
    candidates.append(cell_4disk)
    
    # 5-8. Random designs with various densities
    for density in [0.3, 0.4, 0.5, 0.6]:
        cell_rand = _random_symmetric_cell(size, density)
        candidates.append(cell_rand)
    
    # Score all candidates
    best_cell = None
    best_score = -float('inf')
    
    for cell in candidates:
        cell = _symmetrize(cell)
        score = _compute_gap_surrogate(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Hill-climb from the best candidate
    best_cell, best_score = _hill_climb(best_cell, size, max_iters=80, 
                                         time_limit=min(90.0, _DEADLINE - (time.time() - start_time) - 20.0))
    
    # Try hill-climbing from a few other candidates if time allows
    remaining = _DEADLINE - (time.time() - start_time) - 10.0
    if remaining > 20.0:
        for cell in candidates[:4]:
            if time.time() - start_time > _DEADLINE - 15.0:
                break
            cell = _symmetrize(cell)
            cell, score = _hill_climb(cell, size, max_iters=30,
                                       time_limit=min(30.0, _DEADLINE - (time.time() - start_time) - 10.0))
            if score > best_score:
                best_score = score
                best_cell = cell
    
    # Final check: ensure deadline
    if time.time() - start_time > _DEADLINE - 5.0:
        # Return immediately
        return best_cell
    
    # One more round of local search if we have time
    remaining = _DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 15.0:
        best_cell, best_score = _hill_climb(best_cell, size, max_iters=40,
                                             time_limit=remaining)
    
    return best_cell
