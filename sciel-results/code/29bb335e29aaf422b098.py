"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _make_symmetric(cell):
    """Enforce D4 symmetry (transpose, row-reverse, col-reverse) by averaging/rounding."""
    cell = np.array(cell, dtype=float)
    cell = (cell + cell.T) / 2.0
    cell = (cell + cell[::-1, :]) / 2.0
    cell = (cell + cell[:, ::-1]) / 2.0
    cell = (cell + cell[::-1, :]) / 2.0
    cell = (cell + cell[:, ::-1]) / 2.0
    cell = (cell + cell.T) / 2.0
    cell = np.round(cell)
    cell = np.clip(cell, 0, 1)
    return cell.astype(int)


def _gen_rod(radius, size=32):
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                cell[i, j] = 1
    return cell


def _gen_hole(radius, size=32):
    cell = np.ones((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < radius * radius:
                cell[i, j] = 0
    return cell


def _gen_cross(arm_width, arm_len, size=32):
    cell = np.zeros((size, size), dtype=int)
    c = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            di = abs(i - c)
            dj = abs(j - c)
            if di < arm_len and dj < arm_width / 2.0 + 0.5:
                cell[i, j] = 1
            if dj < arm_len and di < arm_width / 2.0 + 0.5:
                cell[i, j] = 1
    return cell


def _gen_diamond(r, size=32):
    cell = np.zeros((size, size), dtype=int)
    c = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            di = abs(i - c)
            dj = abs(j - c)
            if di + dj < r:
                cell[i, j] = 1
    return cell


def _gen_square(half, size=32):
    cell = np.zeros((size, size), dtype=int)
    c = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if abs(i - c) < half and abs(j - c) < half:
                cell[i, j] = 1
    return cell


def _gen_ring(outer, inner, size=32):
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r2 = x * x + y * y
            if r2 < outer * outer and r2 >= inner * inner:
                cell[i, j] = 1
    return cell


def _gen_diamond_hole(r, size=32):
    cell = np.ones((size, size), dtype=int)
    c = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            di = abs(i - c)
            dj = abs(j - c)
            if di + dj < r:
                cell[i, j] = 0
    return cell


def _pw_tm_bands(cell, n_pw, kx, ky, eps_high=12.0, eps_low=1.0):
    """Compute TM eigenvalues (omega^2 * a^2) via plane-wave method.
    Returns sorted list of eigenvalues."""
    N = cell.shape[0]
    eps = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
    # Fourier coefficients of epsilon
    G_list = []
    for mx in range(-n_pw, n_pw + 1):
        for my in range(-n_pw, n_pw + 1):
            G_list.append((mx, my))
    G_list = np.array(G_list, dtype=float)
    nG = len(G_list)
    # Compute Fourier coefficients of epsilon at each G
    # eps_G = (1/N^2) sum_{i,j} eps[i,j] exp(-i 2pi (Gx i/N + Gy j/N))
    # Use FFT for speed: reshape eps to N x N, do 2D FFT
    eps_FFT = np.fft.fft2(eps)
    eps_FFT = eps_FFT / (N * N)
    # Map G indices to FFT bin indices
    # G = (mx, my) in units of 2pi/N, FFT bin index = (mx mod N, my mod N)
    eps_G = np.zeros(nG, dtype=complex)
    for idx, (mx, my) in enumerate(G_list):
        ix = int(mx) % N
        iy = int(my) % N
        eps_G[idx] = eps_FFT[iy, ix]
    # For TM: matrix element M_{G,G'} = (k+G)^2 * delta_{G,G'} / eps_{G-G'}
    # Actually for TM: (k+G)^2 * A_G / eps = sum_{G'} (1/eps_{G-G'}) (k+G')^2 A_{G'}
    # Let's use: for each G, (k+G)^2 A_G = sum_{G'} eps_{G-G'}^{-1} (k+G')^2 A_{G'}
    # Wait, need to be careful. TM: (k+G)^2 A_G / eps = sum (k+G')^2 A_{G'} / eps_{G-G'}
    # Standard form: (k+G)^2 A_G = sum_{G'} (1/eps_{G-G'}) (k+G')^2 A_{G'}
    # But 1/eps is not simply 1/eps_G. For pixelated, we need 1/eps field.
    # Actually the correct formulation: for TM, the equation is
    # sum_{G'} (1/eps_{G-G'}) (k+G')^2 A_{G'} = (k+G)^2 A_G
    # where 1/eps is the Fourier transform of 1/epsilon(x).
    inv_eps = 1.0 / eps
    inv_eps_FFT = np.fft.fft2(inv_eps) / (N * N)
    inv_eps_G = np.zeros(nG, dtype=complex)
    for idx, (mx, my) in enumerate(G_list):
        ix = int(mx) % N
        iy = int(my) % N
        inv_eps_G[idx] = inv_eps_FFT[iy, ix]
    # Build matrix
    # M_{G,G'} = delta_{G,G'} * (k+G)^2 - (1/eps_{G-G'}) (k+G')^2
    # Actually: (k+G)^2 A_G - sum_{G'} inv_eps_{G-G'} (k+G')^2 A_{G'} = 0
    # So M_{G,G'} = (k+G)^2 delta_{G,G'} - inv_eps_{G-G'} (k+G')^2
    kG = np.zeros((nG, 2))
    for idx, (mx, my) in enumerate(G_list):
        kG[idx, 0] = kx + 2 * math.pi * mx
        kG[idx, 1] = ky + 2 * math.pi * my
    kG2 = np.sum(kG * kG, axis=1)
    # inv_eps_{G-G'} for G, G' in G_list
    # G-G' = (mx1-mx2, my1-my2)
    # We need inv_eps at these differences
    # Build a dict
    inv_eps_dict = {}
    for idx, (mx, my) in enumerate(G_list):
        inv_eps_dict[(int(mx), int(my))] = inv_eps_G[idx]
    M = np.zeros((nG, nG), dtype=complex)
    for i in range(nG):
        M[i, i] = kG2[i]
        for j in range(nG):
            dm = int(G_list[i, 0]) - int(G_list[j, 0])
            dn = int(G_list[i, 1]) - int(G_list[j, 1])
            if (dm, dn) in inv_eps_dict:
                M[i, j] -= inv_eps_dict[(dm, dn)] * kG2[j]
    # Solve generalized eigenvalue: M A = 0, but this is a standard eigenvalue problem
    # Actually the equation is sum_j M_{ij} A_j = 0, so M is the matrix and we find eigenvectors with eigenvalue 0?
    # No. Let me reconsider. The equation is:
    # (k+G)^2 A_G = sum_{G'} inv_eps_{G-G'} (k+G')^2 A_{G'}
    # This is an eigenvalue problem where the eigenvalue is omega^2.
    # Let me rewrite: omega^2 A_G = sum_{G'} inv_eps_{G-G'} (k+G')^2 A_{G'}
    # So the matrix is B_{G,G'} = inv_eps_{G-G'} (k+G')^2 and we find eigenvalues of B.
    B = np.zeros((nG, nG), dtype=complex)
    for i in range(nG):
        for j in range(nG):
            dm = int(G_list[i, 0]) - int(G_list[j, 0])
            dn = int(G_list[i, 1]) - int(G_list[j, 1])
            if (dm, dn) in inv_eps_dict:
                B[i, j] = inv_eps_dict[(dm, dn)] * kG2[j]
    # B is not Hermitian in general, but eigenvalues should be real
    eigvals = np.linalg.eigvalsh((B + B.conj().T) / 2.0)
    eigvals = np.sort(np.real(eigvals))
    return eigvals


def _estimate_gap(cell, n_pw=12, n_k=8):
    """Estimate the gap-midgap ratio between bands 4 and 5."""
    # Irreducible BZ of square lattice: kx >= ky >= 0, kx + ky <= pi (in units of 2pi/a, so k in [0, 0.5])
    # Actually in units where G = 2pi/a * (m, n), the BZ is [-pi, pi]^2, irreducible is 0 <= ky <= kx <= pi
    # Let's use kx, ky in [0, 0.5] in units of 2pi/a, so actual k = 2pi * (kx, ky)
    # The irreducible wedge: 0 <= ky <= kx, kx + ky <= 0.5 (in units of 2pi/a)
    # Wait, the BZ boundary is at |k| = pi in each direction, i.e., kx = ky = 0.5 in our units.
    # The irreducible wedge for square lattice D4: 0 <= ky <= kx, and kx <= 0.5, ky <= 0.5
    # But also the BZ is the square [-0.5, 0.5]^2 in our units.
    # Irreducible: 0 <= ky <= kx <= 0.5
    k_points = []
    # Sample the wedge
    for i in range(n_k):
        for j in range(n_k):
            kx = (i + 0.5) / n_k * 0.5
            ky = (j + 0.5) / n_k * 0.5
            if ky <= kx:
                k_points.append((kx, ky))
    # Add high-symmetry points
    k_points.append((0.5, 0.5))  # M point
    k_points.append((0.5, 0.0))  # X point
    k_points.append((0.0, 0.0))  # Gamma
    k_points.append((0.25, 0.25))  # center of wedge
    
    max_omega4 = 0.0
    min_omega5 = float('inf')
    for kx, ky in k_points:
        # k in units of 2pi/a, so actual wavevector = 2pi * (kx, ky)
        eig = _pw_tm_bands(cell, n_pw, 2 * math.pi * kx, 2 * math.pi * ky)
        if len(eig) >= 5:
            max_omega4 = max(max_omega4, eig[3])
            min_omega5 = min(min_omega5, eig[4])
    if min_omega5 == float('inf') or max_omega4 <= 0:
        return -1.0
    gap = min_omega5 - max_omega4
    midgap = min_omega5 + max_omega4
    if midgap <= 0:
        return -1.0
    return 2.0 * gap / midgap


def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM band gap 4-5."""
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 180  # safety margin before 200s limit
    
    # Generate a diverse set of candidate designs
    candidates = []
    
    # Rods
    for r in [0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3]:
        candidates.append(_gen_rod(r, size))
    
    # Holes (dielectric slab with air hole)
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        candidates.append(_gen_hole(r, size))
    
    # Crosses
    for aw in [3, 5, 7, 9, 11]:
        for al in [5, 8, 11, 14, 16]:
            if al <= size // 2:
                candidates.append(_gen_cross(aw, al, size))
    
    # Diamonds
    for r in [4, 6, 8, 10, 12, 14]:
        candidates.append(_gen_diamond(r, size))
    
    # Diamond holes
    for r in [3, 5, 7, 9, 11, 13]:
        candidates.append(_gen_diamond_hole(r, size))
    
    # Squares
    for h in [3, 5, 7, 9, 11, 13]:
        candidates.append(_gen_square(h, size))
    
    # Rings
    for outer in [0.3, 0.35, 0.4]:
        for inner in [0.1, 0.15, 0.2]:
            if inner < outer:
                candidates.append(_gen_ring(outer, inner, size))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(c.flatten().tolist())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Evaluate candidates with a fast PW solver
    best_cell = unique_candidates[0]
    best_gap = -1.0
    
    # First pass: quick evaluation with small n_pw
    for c in unique_candidates:
        if time.time() > deadline - 60:
            break
        gap = _estimate_gap(c, n_pw=8, n_k=5)
        if gap > best_gap:
            best_gap = gap
            best_cell = c
    
    # Refine around the best candidate
    if time.time() < deadline - 30:
        # Try finer PW resolution on the top candidates
        top_candidates = []
        scored = []
        for c in unique_candidates[:20]:
            gap = _estimate_gap(c, n_pw=10, n_k=6)
            scored.append((gap, c))
        scored.sort(key=lambda x: -x[0])
        top_candidates = [c for _, c in scored[:5]]
        
        for c in top_candidates:
            if time.time() > deadline - 20:
                break
            gap = _estimate_gap(c, n_pw=12, n_k=8)
            if gap > best_gap:
                best_gap = gap
                best_cell = c
    
    # Local search: perturb the best cell
    if time.time() < deadline - 15:
        best_cell_arr = np.array(best_cell)
        current_best = best_gap
        current_best_cell = best_cell_arr.copy()
        
        # Try flipping small patches
        np.random.seed(42)
        for iteration in range(50):
            if time.time() > deadline - 10:
                break
            # Pick a random symmetric position to flip
            i = np.random.randint(0, size // 2)
            j = np.random.randint(0, size // 2)
            # Flip the symmetric group of this pixel
            new_cell = current_best_cell.copy()
            for (ii, jj) in [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                             (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)]:
                if 0 <= ii < size and 0 <= jj < size:
                    new_cell[ii, jj] = 1 - new_cell[ii, jj]
            gap = _estimate_gap(new_cell, n_pw=8, n_k=5)
            if gap > current_best:
                current_best = gap
                current_best_cell = new_cell.copy()
        
        if current_best > best_gap:
            best_gap = current_best
            best_cell = current_best_cell
    
    # Convert to nested list
    result = _make_symmetric(best_cell)
    return result.tolist()

# EVOLVE-BLOCK-END
