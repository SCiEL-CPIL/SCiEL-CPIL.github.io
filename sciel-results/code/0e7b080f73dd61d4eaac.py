"""Photonic crystal design: symmetric cell search with fast plane-wave evaluator."""
import numpy as np
import time
import math
import random

# EVOLVE-BLOCK-START

def _build_fast_evaluator(size, eps_low, eps_high):
    """
    Build a fast plane-wave evaluator for TM bands 4 and 5 gap ratio.
    Uses 17x17=289 plane waves and a reduced set of k-points for speed.
    """
    n_pw = 17
    half = n_pw // 2  # 8
    idx = np.arange(-half, half + 1)
    Gx, Gy = np.meshgrid(idx, idx, indexing='ij')
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    nG = n_pw * n_pw
    
    # Precompute G index mapping: (dx, dy) -> linear index in G array
    # G values range from -half to half
    g_index = {}
    for i in range(nG):
        g_index[(int(Gx_flat[i]), int(Gy_flat[i]))] = i
    
    # k-points: use a reduced set for speed during search
    # Irreducible BZ: 0 <= ky <= kx, kx,ky in [0, 0.5]
    # Use ~12 key points covering boundaries and interior
    k_points = []
    # Along X: (k, 0)
    for i in range(5):
        k_points.append((i / 4.0 * 0.5, 0.0))
    # Along diagonal: (k, k)
    for i in range(1, 5):
        k_points.append((i / 4.0 * 0.5, i / 4.0 * 0.5))
    # Along M-X boundary: (0.5, k)
    for i in range(1, 5):
        k_points.append((0.5, i / 4.0 * 0.5))
    # A few interior points
    k_points.append((0.25, 0.1))
    k_points.append((0.35, 0.15))
    k_points.append((0.45, 0.2))
    
    k_points = np.array(k_points)
    n_k = len(k_points)
    
    # Precompute (k+G)^2 for each k-point
    # k in units of 2*pi (lattice constant = 1)
    kvecs = k_points * 2.0 * math.pi  # (n_k, 2)
    # k_plus_G_sq[k_idx, g_idx] = (kx + Gx[g])^2 + (ky + Gy[g])^2
    k_plus_G_sq = np.zeros((n_k, nG))
    for ki in range(n_k):
        kx = kvecs[ki, 0]
        ky = kvecs[ki, 1]
        k_plus_G_sq[ki] = (Gx_flat + kx)**2 + (Gy_flat + ky)**2
    
    def evaluate(cell_arr):
        """
        cell_arr: 32x32 array of 0/1.
        Returns gap-midgap ratio for TM bands 4 and 5.
        """
        # Compute 1/eps field
        eps_field = eps_low + (eps_high - eps_low) * cell_arr.astype(np.float64)
        inv_eps = 1.0 / eps_field
        
        # FFT to get Fourier coefficients of 1/eps
        N_fft = size  # 32, which is power of 2
        inv_eps_fft = np.fft.fft2(inv_eps) / (N_fft * N_fft)
        
        # Extract coefficients for our G vectors
        # G vectors are -half..half = -8..8
        # FFT indices: 0..31 correspond to frequencies 0,1,...,15,-16,...,-1 (mod 32)
        # So frequency f corresponds to FFT index f % 32
        eps_coeffs = np.zeros(nG, dtype=np.complex128)
        for gi in range(nG):
            gx = int(Gx_flat[gi])
            gy = int(Gy_flat[gi])
            fft_ix = gx % N_fft
            fft_iy = gy % N_fft
            eps_coeffs[gi] = inv_eps_fft[fft_ix, fft_iy]
        
        # For TM mode: eigenvalue problem
        # sum_{G'} eps_{G-G'} |G'+k|^2 c_{G'} = omega^2 c_G
        # Matrix M[i,j] = eps_{G_i - G_j} * |G_j + k|^2
        # M = C * D where C is circulant (C[i,j] = eps_{G_i-G_j}) and D = diag(|G+k|^2)
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for ki in range(n_k):
            # Build matrix M
            ksq = k_plus_G_sq[ki]
            
            # Build circulant matrix C: C[i,j] = eps_coeffs[index(G_i - G_j)]
            # Precompute difference indices
            # For efficiency, build C using the fact that it's circulant-like
            C = np.zeros((nG, nG), dtype=np.complex128)
            for i in range(nG):
                gx_i = int(Gx_flat[i])
                gy_i = int(Gy_flat[i])
                for j in range(nG):
                    dx = gx_i - int(Gx_flat[j])
                    dy = gy_i - int(Gy_flat[j])
                    if abs(dx) <= half and abs(dy) <= half:
                        C[i, j] = eps_coeffs[g_index[(dx, dy)]]
            
            # M = C * diag(ksq)
            # M[i,j] = C[i,j] * ksq[j]
            M = C * ksq[None, :]
            
            # Symmetrize (should be real symmetric for real eps)
            M_sym = (M + M.conj().T) / 2.0
            M_real = M_sym.real  # Should be real
            
            try:
                eigvals = np.linalg.eigvalsh(M_real)
                eigvals = np.sort(eigvals)
            except:
                eigvals = np.linalg.eigvals(M)
                eigvals = np.sort(np.real(eigvals))
            
            if len(eigvals) >= 5:
                w4 = math.sqrt(max(0, eigvals[3]))
                w5 = math.sqrt(max(0, eigvals[4]))
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)
        
        gap = min_omega_5 - max_omega_4
        midgap = min_omega_5 + max_omega_4
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap
    
    return evaluate


def _apply_d4_symmetry(cell, size):
    """
    Enforce D4 symmetry (square symmetry group) on the cell.
    The cell must be invariant under:
    - 90-degree rotation
    - Reflection about x-axis (row reversal)
    - Reflection about y-axis (column reversal)
    - Transpose (reflection about diagonal)
    
    We enforce by averaging over all 8 symmetry operations and thresholding.
    """
    sym_sum = np.zeros((size, size), dtype=np.float64)
    
    # 8 symmetry operations of D4
    ops = [
        lambda c: c,                          # identity
        lambda c: c[::-1, :],                  # flip rows
        lambda c: c[:, ::-1],                  # flip cols
        lambda c: c[::-1, ::-1],               # 180 rotation
        lambda c: c.T,                         # transpose
        lambda c: c.T[::-1, :],                # transpose + flip rows
        lambda c: c.T[:, ::-1],                # transpose + flip cols
        lambda c: c.T[::-1, ::-1],             # transpose + 180
    ]
    
    for op in ops:
        sym_sum += op(cell)
    
    sym_avg = sym_sum / 8.0
    return (sym_avg > 0.5).astype(int)


def _get_symmetric_orbit(i, j, size):
    """
    Get all positions in the D4 orbit of (i, j).
    Returns list of unique (r, c) positions.
    """
    points = set()
    # D4 group elements applied to (i, j)
    # Using 0-indexed coordinates
    candidates = [
        (i, j),
        (size - 1 - i, j),      # flip x
        (i, size - 1 - j),      # flip y
        (size - 1 - i, size - 1 - j),  # flip both (180 rot)
        (j, i),                 # transpose
        (size - 1 - j, i),      # transpose + flip x
        (j, size - 1 - j),      # transpose + flip y
        (size - 1 - j, size - 1 - i),  # transpose + flip both
    ]
    for (r, c) in candidates:
        if 0 <= r < size and 0 <= c < size:
            points.add((r, c))
    return list(points)


def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell maximizing the TM band gap
    between bands 4 and 5.
    """
    t_start = time.time()
    DEADLINE = 210.0  # Safety margin before 230s limit
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    lower_band = problem["lower_band"]  # 4
    
    # Build fast evaluator
    evaluate = _build_fast_evaluator(size, eps_low, eps_high)
    
    # Initialize with a centered dielectric rod (the seed solution)
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        x = (i + 0.5) / size - 0.5
        for j in range(size):
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i, j] = 1
    
    # Ensure symmetry
    cell = _apply_d4_symmetry(cell, size)
    
    best_score = evaluate(cell)
    best_cell = cell.copy()
    
    # Random seed for reproducibility
    rng = random.Random(42)
    
    # Local search: flip symmetric orbits of pixels
    # Track how many evaluations we can do
    eval_count = 1
    
    while time.time() - t_start < DEADLINE:
        # Pick a random pixel
        i = rng.randint(0, size - 1)
        j = rng.randint(0, size - 1)
        
        orbit = _get_symmetric_orbit(i, j, size)
        
        # Skip if orbit is too large (inefficient) or trivial (all same value)
        if len(orbit) > 8:
            continue
        
        # Check if flipping would change anything
        current_vals = [cell[r, c] for (r, c) in orbit]
        if all(v == 0 for v in current_vals) or all(v == 1 for v in current_vals):
            pass  # Still flip, it's a valid move
        else:
            # Mixed values - this shouldn't happen with proper symmetry enforcement
            # but handle gracefully
            pass
        
        # Flip the orbit
        for (r, c) in orbit:
            cell[r, c] = 1 - cell[r, c]
        
        score = evaluate(cell)
        eval_count += 1
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        else:
            # Revert
            for (r, c) in orbit:
                cell[r, c] = 1 - cell[r, c]
        
        # Periodic check: try a multi-pixel move every N steps
        if eval_count % 50 == 0 and time.time() - t_start < DEADLINE:
            # Try flipping two random orbits simultaneously
            i1 = rng.randint(0, size - 1)
            j1 = rng.randint(0, size - 1)
            i2 = rng.randint(0, size - 1)
            j2 = rng.randint(0, size - 1)
            
            orbit1 = _get_symmetric_orbit(i1, j1, size)
            orbit2 = _get_symmetric_orbit(i2, j2, size)
            
            # Combine orbits
            all_orbit = set(orbit1) | set(orbit2)
            if len(all_orbit) <= 16:
                for (r, c) in all_orbit:
                    cell[r, c] = 1 - cell[r, c]
                
                score = evaluate(cell)
                eval_count += 1
                
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                else:
                    for (r, c) in all_orbit:
                        cell[r, c] = 1 - cell[r, c]
        
        # Occasionally try a larger structural change:
        # Modify the rod radius by flipping a ring of pixels
        if eval_count % 100 == 0 and time.time() - t_start < DEADLINE:
            # Try changing the effective radius
            # Pick a random radius and flip all pixels in a thin annulus
            r_target = rng.uniform(0.1, 0.4)
            changed = False
            for i in range(size):
                x = (i + 0.5) / size - 0.5
                for j in range(size):
                    y = (j + 0.5) / size - 0.5
                    r = math.sqrt(x * x + y * y)
                    if abs(r - r_target) < 0.02:
                        # Flip this pixel and its symmetric partners
                        orbit = _get_symmetric_orbit(i, j, size)
                        for (r2, c2) in orbit:
                            cell[r2, c2] = 1 - cell[r2, c2]
                        changed = True
            
            if changed:
                score = evaluate(cell)
                eval_count += 1
                if score > best_score:
                    best_score = score
                    best_cell = cell.copy()
                else:
                    # Revert: we need to track what we changed
                    # Simpler: just regenerate from best_cell
                    cell = best_cell.copy()
    
    # Final symmetry enforcement
    best_cell = _apply_d4_symmetry(best_cell, size)
    
    return best_cell.tolist()

# EVOLVE-BLOCK-END
