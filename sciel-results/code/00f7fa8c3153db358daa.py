import numpy as np
import time
from itertools import product

def design(problem):
    """
    Direct pixel-level search over the C4v fundamental domain for the 32x32
    TM photonic crystal unit cell, maximizing the band-4/5 gap-midgap ratio.
    """
    start_time = time.time()
    deadline = start_time + 215.0  # safety margin before 230s kill

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # ------------------------------------------------------------------
    # 1. Build the C4v fundamental-domain mapping
    #    For a pixel (i, j) in [0, size), the 8 symmetric images under
    #    the square symmetry group about the cell centre are:
    #      (i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
    #      (j, i), (size-1-j, i), (j, size-1-i), (size-1-j, size-1-i)
    #    The fundamental domain is the set of pixels where (i, j) is the
    #    lexicographically smallest among its 8 images.
    # ------------------------------------------------------------------
    def sym8(i, j):
        s = size
        return sorted([
            (i, j),
            (s - 1 - i, j),
            (i, s - 1 - j),
            (s - 1 - i, s - 1 - j),
            (j, i),
            (s - 1 - j, i),
            (j, s - 1 - i),
            (s - 1 - j, s - 1 - i),
        ])

    # Map each fundamental pixel to its full 8-pixel orbit
    full_orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            orbit = sym8(i, j)
            rep = orbit[0]  # lexicographic representative
            if rep not in seen:
                seen.add(rep)
                full_orbits.append(orbit)

    num_fund = len(full_orbits)  # number of independent pixels

    # ------------------------------------------------------------------
    # 2. Fast evaluator using the exact same PW basis as the real evaluator
    #    The real evaluator uses 289 PWs = 17x17, i.e. G indices -8..8.
    #    TM band equation:  sum_j  (k+G_i)·(k+G_j) / eps(G_i - G_j)  a_j = w^2 a_i
    #    The matrix M = diag(|k+G|^2) @ D_inv_eps is NOT Hermitian in general,
    #    but it is similar to a Hermitian matrix. We use the standard trick:
    #    M = D_k @ D_eps_inv  where D_k = diag(|k+G|^2), D_eps_inv is the
    #    Toeplitz matrix of 1/eps Fourier coefficients.
    #    The eigenvalues of M are real and positive (for TM).
    #    We can compute them as eigenvalues of D_k^{1/2} @ D_eps_inv @ D_k^{1/2}
    #    which IS Hermitian, or simply use np.linalg.eigvals on M and take
    #    the real parts. For robustness we use the Hermitian form.
    # ------------------------------------------------------------------
    N_pw = 8  # gives 17x17 = 289 PWs, matching the evaluator
    G_vals = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_vals, G_vals, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # shape (289, 2)
    num_pw = len(G_vectors)

    # Precompute G differences for the Toeplitz structure
    # diff[i, j] = G_vectors[i] - G_vectors[j]
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (289, 289, 2)

    # k-points: cover the irreducible BZ boundary and interior
    # Irreducible BZ of square lattice: 0 <= kx <= ky <= 0.5  (in units of pi/a)
    # We use a set of representative k-points similar to the evaluator's 30.
    k_points = []
    # Gamma-M edge: (t, 0), t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, 0.0))
    # M-X edge: (0.5, t), t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((0.5, t))
    # Gamma-X edge: (t, t), t in [0, 0.5]
    for t in np.linspace(0, 0.5, 8):
        k_points.append((t, t))
    # Interior points
    interior = [
        (0.1, 0.3), (0.3, 0.1), (0.2, 0.4), (0.4, 0.2),
        (0.15, 0.25), (0.25, 0.15), (0.35, 0.35), (0.2, 0.2),
        (0.4, 0.4), (0.1, 0.1),
    ]
    for k in interior:
        k_points.append(k)

    # Remove duplicates
    seen_k = set()
    unique_k = []
    for k in k_points:
        key = (round(k[0], 10), round(k[1], 10))
        if key not in seen_k:
            seen_k.add(key)
            unique_k.append(k)

    def evaluate_cell_fast(fund_pixels):
        """
        Compute the gap-midgap ratio for a C4v symmetric cell defined by
        fundamental-domain pixels. fund_pixels is an array of 0/1 of length
        num_fund. Returns the ratio (higher is better).
        """
        # Build the full 32x32 epsilon grid
        eps_grid = np.zeros((size, size), dtype=float)
        for idx, orbit in enumerate(full_orbits):
            val = eps_high if fund_pixels[idx] == 1 else eps_low
            for (oi, oj) in orbit:
                eps_grid[oi, oj] = val

        # FFT of 1/eps
        inv_eps = 1.0 / eps_grid
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)

        # Build the coefficient matrix C[i, j] = fft_inv_eps[(G_i - G_j) mod size]
        # This is the Fourier transform of 1/eps at G_i - G_j
        coeff = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            gi = G_vectors[i]
            for j in range(num_pw):
                gj = G_vectors[j]
                dx = int(gi[0] - gj[0])
                dy = int(gi[1] - gj[1])
                ix = dx % size
                iy = dy % size
                coeff[i, j] = fft_inv_eps[ix, iy]

        min_omega_5 = np.inf
        max_omega_4 = -np.inf

        for kx, ky in unique_k:
            k = np.array([kx, ky], dtype=float)
            k_plus_G = k + G_vectors  # (289, 2)
            omega_sq = np.sum(k_plus_G ** 2, axis=1)  # (289,)

            # M[i, j] = omega_sq[i] * coeff[i, j]
            # Eigenvalues of M give w^2. M is similar to a Hermitian matrix.
            # Use the Hermitian form: H = D^{1/2} @ C @ D^{1/2} where D = diag(omega_sq)
            # H[i, j] = sqrt(omega_sq[i]) * coeff[i, j] * sqrt(omega_sq[j])
            # This is Hermitian because coeff[i,j] = conj(coeff[j,i]) (FT of real function).
            # Actually coeff[i,j] is the FT of a real function, so coeff[i,j] = conj(coeff[j,i]).
            # Let's verify: coeff[i,j] = sum 1/eps(r) exp(i(G_i-G_j).r)
            # coeff[j,i] = sum 1/eps(r) exp(i(G_j-G_i).r) = conj(coeff[i,j]) since 1/eps is real.
            # So H = D^{1/2} C D^{1/2} is Hermitian.
            sqrt_wsq = np.sqrt(np.maximum(omega_sq, 0))
            H = np.outer(sqrt_wsq, sqrt_wsq) * coeff  # H[i,j] = sqrt(w_i)*C[i,j]*sqrt(w_j)

            # Ensure Hermitian (numerical cleanup)
            H = (H + H.conj().T) / 2.0

            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(np.real(eigvals))

            if len(eigvals) >= 5:
                w4 = np.sqrt(max(eigvals[3], 0))
                w5 = np.sqrt(max(eigvals[4], 0))
                max_omega_4 = max(max_omega_4, w4)
                min_omega_5 = min(min_omega_5, w5)

        if min_omega_5 == np.inf or max_omega_4 == -np.inf or max_omega_4 == 0:
            return -1.0

        gap = min_omega_5 - max_omega_4
        midgap = 0.5 * (min_omega_5 + max_omega_4)
        if midgap <= 0:
            return -1.0
        return 2.0 * gap / midgap

    # ------------------------------------------------------------------
    # 3. Initial candidate: a solid dielectric circle in the centre
    # ------------------------------------------------------------------
    def make_initial_cell(radius_frac=0.3):
        fund = np.zeros(num_fund, dtype=int)
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        r_pix = radius_frac * size
        for idx, orbit in enumerate(full_orbits):
            # Check if the representative pixel is inside the circle
            (ri, rj) = orbit[0]
            dist = np.sqrt((ri - cx) ** 2 + (rj - cy) ** 2)
            fund[idx] = 1 if dist < r_pix else 0
        return fund

    # Try several initial shapes
    best_pixels = None
    best_score = -np.inf
    for frac in [0.2, 0.25, 0.3, 0.35, 0.4]:
        p = make_initial_cell(frac)
        s = evaluate_cell_fast(p)
        if s > best_score:
            best_score = s
            best_pixels = p.copy()

    # Also try a ring / annulus shape
    for inner_frac, outer_frac in [(0.1, 0.35), (0.15, 0.4), (0.2, 0.45)]:
        fund = np.zeros(num_fund, dtype=int)
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for idx, orbit in enumerate(full_orbits):
            (ri, rj) = orbit[0]
            dist = np.sqrt((ri - cx) ** 2 + (rj - cy) ** 2)
            r_in = inner_frac * size
            r_out = outer_frac * size
            fund[idx] = 1 if (r_in < dist < r_out) else 0
        s = evaluate_cell_fast(fund)
        if s > best_score:
            best_score = s
            best_pixels = fund.copy()

    # Also try a square dielectric block
    for half_size in [6, 7, 8, 9]:
        fund = np.zeros(num_fund, dtype=int)
        cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
        for idx, orbit in enumerate(full_orbits):
            (ri, rj) = orbit[0]
            if abs(ri - cx) < half_size and abs(rj - cy) < half_size:
                fund[idx] = 1
        s = evaluate_cell_fast(fund)
        if s > best_score:
            best_score = s
            best_pixels = fund.copy()

    # ------------------------------------------------------------------
    # 4. Adaptive multi-flip local search
    #    At each step, try flipping 1, 2, or 3 random fundamental pixels
    #    and keep the best improvement. Also try "grow/shrink" moves that
    #    flip all pixels at a certain distance from the centre.
    # ------------------------------------------------------------------
    current_pixels = best_pixels.copy()
    current_score = best_score

    # Precompute distance from centre for each fundamental pixel
    cx, cy = (size - 1) / 2.0, (size - 1) / 2.0
    fund_dists = np.zeros(num_fund)
    for idx, orbit in enumerate(full_orbits):
        (ri, rj) = orbit[0]
        fund_dists[idx] = np.sqrt((ri - cx) ** 2 + (rj - cy) ** 2)

    # Sort fundamental pixels by distance from centre for radial moves
    dist_order = np.argsort(fund_dists)

    num_iterations = 0
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 10:
        num_iterations += 1

        # --- Try single flips ---
        best_candidate = None
        best_candidate_score = current_score

        # Try a batch of single flips
        num_single_tries = 20
        for _ in range(num_single_tries):
            if time.time() > deadline - 10:
                break
            idx = np.random.randint(0, num_fund)
            trial = current_pixels.copy()
            trial[idx] = 1 - trial[idx]
            s = evaluate_cell_fast(trial)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = trial.copy()

        # --- Try double flips ---
        num_double_tries = 8
        for _ in range(num_double_tries):
            if time.time() > deadline - 10:
                break
            i1, i2 = np.random.choice(num_fund, size=2, replace=False)
            trial = current_pixels.copy()
            trial[i1] = 1 - trial[i1]
            trial[i2] = 1 - trial[i2]
            s = evaluate_cell_fast(trial)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = trial.copy()

        # --- Try triple flips ---
        num_triple_tries = 4
        for _ in range(num_triple_tries):
            if time.time() > deadline - 10:
                break
            idxs = np.random.choice(num_fund, size=3, replace=False)
            trial = current_pixels.copy()
            for idx in idxs:
                trial[idx] = 1 - trial[idx]
            s = evaluate_cell_fast(trial)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = trial.copy()

        # --- Try radial grow/shrink: flip all pixels in a distance band ---
        num_radial_tries = 3
        for _ in range(num_radial_tries):
            if time.time() > deadline - 10:
                break
            # Pick a random radius and flip all pixels within a thin band
            r0 = np.random.uniform(0, size * 0.5)
            band_width = np.random.uniform(1.0, 3.0)
            mask = np.abs(fund_dists - r0) < band_width
            if np.sum(mask) > 0 and np.sum(mask) < num_fund:
                trial = current_pixels.copy()
                trial[mask] = 1 - trial[mask]
                s = evaluate_cell_fast(trial)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = trial.copy()

        # Accept the best candidate
        if best_candidate is not None and best_candidate_score > current_score:
            current_pixels = best_candidate
            current_score = best_candidate_score
            no_improve_count = 0

            if current_score > best_score:
                best_score = current_score
                best_pixels = current_pixels.copy()
        else:
            no_improve_count += 1

        # --- Occasional random restart from a new shape ---
        if no_improve_count >= max_no_improve and time.time() < deadline - 60:
            # Try a completely new initial shape
            np.random.seed(int(time.time() * 1000) % (2**31))
            for frac in [np.random.uniform(0.15, 0.45)]:
                p = make_initial_cell(frac)
                s = evaluate_cell_fast(p)
                if s > best_score:
                    best_score = s
                    best_pixels = p.copy()
            current_pixels = best_pixels.copy()
            current_score = best_score
            no_improve_count = 0

        # If we've been stuck for a long time, try a more aggressive restart
        if no_improve_count >= max_no_improve * 2 and time.time() < deadline - 40:
            # Random binary pattern with some structure
            p = np.random.randint(0, 2, num_fund)
            # Add a central bias
            central_mask = fund_dists < 0.3 * size
            p[central_mask] = np.random.randint(0, 2, np.sum(central_mask))
            s = evaluate_cell_fast(p)
            if s > best_score:
                best_score = s
                best_pixels = p.copy()
            current_pixels = best_pixels.copy()
            current_score = best_score
            no_improve_count = 0

    # ------------------------------------------------------------------
    # 5. Final local polish: single-flip hill climbing
    # ------------------------------------------------------------------
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(num_fund):
            if time.time() > deadline - 5:
                break
            trial = best_pixels.copy()
            trial[idx] = 1 - trial[idx]
            s = evaluate_cell_fast(trial)
            if s > best_score:
                best_score = s
                best_pixels = trial.copy()
                improved = True

    # ------------------------------------------------------------------
    # 6. Build the final 32x32 symmetric cell
    # ------------------------------------------------------------------
    final_grid = np.zeros((size, size), dtype=int)
    for idx, orbit in enumerate(full_orbits):
        val = best_pixels[idx]
        for (oi, oj) in orbit:
            final_grid[oi, oj] = val

    # Verify symmetry
    for i in range(size):
        for j in range(size):
            assert final_grid[i, j] == final_grid[j, i], "Transpose symmetry violated"
            assert final_grid[i, j] == final_grid[size - 1 - i, j], "X-reflection violated"
            assert final_grid[i, j] == final_grid[i, size - 1 - j], "Y-reflection violated"

    return final_grid.tolist()
