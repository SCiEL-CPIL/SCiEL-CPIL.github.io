"""Photonic crystal unit cell design for large TM 4-5 band gap."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM 4-5 band gap.
    
    Combines diverse geometric candidate generation with direct stochastic
    pixel search and a fast plane-wave evaluator for local refinement.
    """
    t_start = time.time()
    DEADLINE = 215.0  # safety margin before 230s kill

    size = problem["pixels"]
    eps_high = problem["eps_high"]
    eps_low = problem["eps_low"]

    # --- Fast TM plane-wave solver (17x17 = 289 plane waves) ---

    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # K-points in irreducible BZ (30 points)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)

    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = int(Gx[i]) - int(Gx[j])
            dGy = int(Gy[i]) - int(Gy[j])
            idx_x = dGx + half
            idx_y = dGy + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def compute_band_ratio_fast(coeffs):
        """Fast approximate evaluation using subset of k-points."""
        k_subset = [0, 4, 9, 14, 19, 24, 29]
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in k_subset:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def evaluate_cell(cell):
        """Full evaluation of a cell."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio_full(coeffs)

    def evaluate_cell_fast(cell):
        """Fast evaluation for search guidance."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio_fast(coeffs)

    # --- Symmetry enforcement ---

    def enforce_symmetry(cell):
        """Enforce full square symmetry (D4 group) by averaging over all 8 operations."""
        cell = np.array(cell, dtype=int)
        # Correct D4 group: identity, rot90, rot180, rot270, hflip, vflip, diag, anti-diag
        c = cell.copy()
        rot90 = np.rot90(c, 1)
        rot180 = np.rot90(c, 2)
        rot270 = np.rot90(c, 3)
        hflip = c[:, ::-1]
        vflip = c[::-1, :]
        diag = c.T
        anti_diag = c[::-1, ::-1].T
        ops = [
            c,
            rot90,
            rot180,
            rot270,
            hflip,
            vflip,
            diag,
            anti_diag,
        ]
        vflip = c[::-1, :]
        diag = c.T
        anti_diag = np.rot90(c, 1).T
        
        result = (c + rot90 + rot180 + rot270 + hflip + vflip + diag + anti_diag) / 8.0
        return (result >= 0.5).astype(int)

    # --- Candidate generation ---

    def generate_rod(radius):
        """Single centered rod."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rods(radius, offset):
        """Four rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_ring(inner_r, outer_r):
        """Dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_cross(arm_width, arm_length):
        """Cross/plus shape."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                    cell[i, j] = 1
                if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_four_rings(inner_r, outer_r, offset):
        """Four small rings at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        r2 = dx * dx + dy * dy
                        if inner_r * inner_r < r2 < outer_r * outer_r:
                            cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_moire(f1, f2, phase1, phase2, threshold):
        """Moiré pattern from two counter-rotating square waves."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                v1 = np.sin(2 * np.pi * f1 * (x + y) + phase1)
                sq1 = 1.0 if v1 > 0 else 0.0
                v2 = np.sin(2 * np.pi * f2 * (x - y) + phase2)
                sq2 = 1.0 if v2 > 0 else 0.0
                val = sq1 * sq2
                if val >= threshold:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    # --- Mutation for stochastic search ---

    def mutate_cell(cell, n_mutations=3):
        """Randomly flip full D4 symmetry orbits of pixels."""
        cell = cell.copy()
        for _ in range(n_mutations):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            # Generate all 8 symmetric equivalents
            orbit = set()
            orbit.add((i, j))
            orbit.add((i, size - 1 - j))
            orbit.add((size - 1 - i, j))
            orbit.add((size - 1 - i, size - 1 - j))
            orbit.add((j, i))
            orbit.add((j, size - 1 - i))
            orbit.add((size - 1 - i, j))
            orbit.add((size - 1 - i, size - 1 - j))
            new_val = 1 - cell[i, j]
            for (oi, oj) in orbit:
                cell[oi, oj] = new_val
        return cell

    # --- Main search ---

    best_score = -np.inf
    best_cell = None

    def update_best(cell, score):
        nonlocal best_score, best_cell
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    print(f"Phase 1: Diverse candidate generation...")

    # Generate and evaluate diverse candidates
    candidates = []

    # Single rods
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        if time.time() > DEADLINE:
            break
        cell = generate_rod(r)
        candidates.append(cell)

    # Four-rod patterns
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        if time.time() > DEADLINE:
            break
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            if time.time() > DEADLINE:
                break
            cell = generate_four_rods(r, off)
            candidates.append(cell)

    # Rings
    for ri in [0.05, 0.08, 0.10, 0.12]:
        if time.time() > DEADLINE:
            break
        for ro in [0.15, 0.18, 0.20, 0.22, 0.25]:
            if time.time() > DEADLINE:
                break
            cell = generate_ring(ri, ro)
            candidates.append(cell)

    # Crosses
    for aw in [0.05, 0.08, 0.10]:
        if time.time() > DEADLINE:
            break
        for al in [0.30, 0.35, 0.40]:
            if time.time() > DEADLINE:
                break
            cell = generate_cross(aw, al)
            candidates.append(cell)

    # Four rings
    for ri in [0.03, 0.04, 0.05]:
        if time.time() > DEADLINE:
            break
        for ro in [0.08, 0.10, 0.12]:
            if time.time() > DEADLINE:
                break
            for off in [0.20, 0.25, 0.30]:
                if time.time() > DEADLINE:
                    break
                cell = generate_four_rings(ri, ro, off)
                candidates.append(cell)

    # Moiré patterns
    for f1 in [3, 4, 5]:
        if time.time() > DEADLINE:
            break
        for f2 in [3, 4, 5]:
            if time.time() > DEADLINE:
                break
            if f1 == f2:
                continue
            for p1 in [0, np.pi / 2]:
                if time.time() > DEADLINE:
                    break
                for p2 in [0, np.pi / 2]:
                    if time.time() > DEADLINE:
                        break
                    for thr in [0.5, 0.7]:
                        if time.time() > DEADLINE:
                            break
                        cell = generate_moire(f1, f2, p1, p2, thr)
                        candidates.append(cell)

    # Evaluate all candidates with fast evaluator first
    print(f"Evaluating {len(candidates)} candidates...")
    fast_scores = []
    for cell in candidates:
        if time.time() > DEADLINE:
            break
        score = evaluate_cell_fast(cell)
        fast_scores.append((score, cell))

    # Sort by fast score and evaluate top candidates with full evaluator
    fast_scores.sort(key=lambda x: -x[0])
    top_n = min(20, len(fast_scores))
    for score, cell in fast_scores[:top_n]:
        if time.time() > DEADLINE:
            break
        full_score = evaluate_cell(cell)
        update_best(cell, full_score)

    print(f"Best after Phase 1: {best_score:.4f}")

    # Phase 2: Intensive stochastic search with simulated annealing
    if best_cell is not None and time.time() < DEADLINE - 40:
        print(f"Phase 2: Stochastic search with annealing...")
        current = best_cell.copy()
        current_score = best_score
        T = max(0.05, abs(current_score) * 0.3)
        T_min = 0.01
        cooling = 0.995

        for iteration in range(2000):
            if time.time() > DEADLINE - 20:
                break
            mutated = mutate_cell(current, n_mutations=random.randint(1, 3))
            fast_score = evaluate_cell_fast(mutated)
            
            # Simulated annealing acceptance
            if fast_score > current_score:
                accept = True
            else:
                delta = current_score - fast_score
                accept = random.random() < np.exp(-delta / max(T, 1e-10))
            
            if accept:
                full_score = evaluate_cell(mutated)
                if full_score > current_score:
                    current = mutated
                    current_score = full_score
                    update_best(current, current_score)
                else:
                    # Keep current but still update T
                    pass
            T *= cooling
            T = max(T, T_min)

    # Phase 3: Local refinement with single-pixel flips
    if best_cell is not None and time.time() < DEADLINE - 20:
        print(f"Phase 3: Local refinement...")
        current = best_cell.copy()
        current_score = best_score
        
        # Try flipping each orbit position once
        for i in range(size):
            if time.time() > DEADLINE - 10:
                break
            for j in range(i + 1):  # only upper triangle to avoid duplicates
                if time.time() > DEADLINE - 10:
                    break
                # Generate orbit
                orbit = set()
                orbit.add((i, j))
                orbit.add((i, size - 1 - j))
                orbit.add((size - 1 - i, j))
                orbit.add((size - 1 - i, size - 1 - j))
                orbit.add((j, i))
                orbit.add((j, size - 1 - i))
                orbit.add((size - 1 - j, i))
                orbit.add((size - 1 - j, size - 1 - i))
                
                # Try flipping this orbit
                test_cell = current.copy()
                new_val = 1 - current[i, j]
                for (oi, oj) in orbit:
                    test_cell[oi, oj] = new_val
                
                fast_score = evaluate_cell_fast(test_cell)
                if fast_score > current_score * 0.98:
                    full_score = evaluate_cell(test_cell)
                    if full_score > current_score:
                        current = test_cell
                        current_score = full_score
                        update_best(current, current_score)

    # Fallback
    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
        best_score = evaluate_cell(best_cell)

    # Final symmetry enforcement and validation
    best_cell = enforce_symmetry(best_cell)
    
    # Verify symmetry
    assert np.array_equal(best_cell, best_cell[::-1, :]), "Vertical symmetry failed"
    assert np.array_equal(best_cell, best_cell[:, ::-1]), "Horizontal symmetry failed"
    assert np.array_equal(best_cell, best_cell.T), "Transpose symmetry failed"
    
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
