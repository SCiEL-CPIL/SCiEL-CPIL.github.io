"""Rewritten program with a stronger PWE solver and parameterized geometric search."""
import time
import math
import numpy as np

def design(problem):
    """Design a symmetric 32x32 unit cell maximizing the TM band-4/5 gap-midgap ratio."""
    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12)

    t0 = time.time()
    deadline = t0 + 215.0  # Safety margin before 230s limit

    N = size
    half = N // 2  # 16

    # --- PWE solver with a moderate basis size for accuracy/speed balance ---
    max_g = 4  # 9x9 = 81 plane waves
    g_indices = [(mx, my) for mx in range(-max_g, max_g + 1) for my in range(-max_g, max_g + 1)]
    num_g = len(g_indices)
    G_vecs = np.array([(2 * math.pi * mx, 2 * math.pi * my) for mx, my in g_indices])

    # K-points covering the irreducible BZ: Gamma(0,0), X(0,pi), M(pi/2,pi/2) and along edges
    k_points = [
        (0.0, 0.0),
        (0.0, math.pi),
        (math.pi / 2, math.pi / 2),
        (0.0, math.pi * 0.5),
        (math.pi / 4, math.pi / 4),
        (0.0, math.pi * 0.25),
        (0.0, math.pi * 0.75),
        (math.pi / 8, math.pi / 8),
        (math.pi / 6, math.pi / 3),
    ]

    def compute_eps_inv_fourier(cell):
        """Compute Fourier coefficients of 1/epsilon using FFT."""
        eps_inv = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                eps_inv[i, j] = 1.0 / eps_high if cell[i][j] == 1 else 1.0
        return np.fft.fft2(eps_inv) / (N * N)

    def evaluate_gap(cell, fft_coeff=None):
        """Evaluate the gap-midgap ratio between TM bands 4 and 5."""
        if fft_coeff is None:
            fft_coeff = compute_eps_inv_fourier(cell)

        # Build eps_inv difference matrix
        eps_inv_mat = np.zeros((num_g, num_g))
        for p in range(num_g):
            mx_p, my_p = g_indices[p]
            for q in range(num_g):
                mx_q, my_q = g_indices[q]
                dm = (mx_p - mx_q) % N
                dn = (my_p - my_q) % N
                eps_inv_mat[p, q] = fft_coeff[dn, dm]

        max_omega4 = 0.0
        min_omega5 = float('inf')

        for kx, ky in k_points:
            kG_sq = (kx + G_vecs[:, 0]) ** 2 + (ky + G_vecs[:, 1]) ** 2
            M = eps_inv_mat * kG_sq[None, :]

            try:
                eigvals = np.linalg.eigvalsh(M)
            except Exception:
                continue

            eigvals = np.sort(eigvals)
            if len(eigvals) < 5:
                continue

            omega4 = math.sqrt(max(eigvals[3], 0))
            omega5 = math.sqrt(max(eigvals[4], 0))

            max_omega4 = max(max_omega4, omega4)
            min_omega5 = min(min_omega5, omega5)

        if min_omega5 == float('inf') or max_omega4 == 0:
            return -1.0

        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0

        return gap / midgap

    def build_cell_from_quadrant(quadrant):
        """Build full 32x32 cell from 16x16 upper-triangle quadrant (i<=j)."""
        cell = [[0] * N for _ in range(N)]
        for i in range(half):
            for j in range(i, half):
                val = quadrant[i][j]
                cell[i][j] = val
                cell[i][N - 1 - j] = val
                cell[N - 1 - i][j] = val
                cell[N - 1 - i][N - 1 - j] = val
                cell[j][i] = val
                cell[j][N - 1 - i] = val
                cell[N - 1 - j][i] = val
                cell[N - 1 - j][N - 1 - i] = val
        return cell

    # --- Parameterized geometric designs: rings and squares ---
    def make_ring_quadrant(r_inner, r_outer, center=(0.5, 0.5)):
        q = [[0] * half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                dx = x - center[0]
                dy = y - center[1]
                r = math.sqrt(dx * dx + dy * dy)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    def make_square_ring_quadrant(r_inner, r_outer, center=(0.5, 0.5)):
        """Square-shaped ring (axis-aligned)."""
        q = [[0] * half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                dx = abs(x - center[0])
                dy = abs(y - center[1])
                r = max(dx, dy)
                if r_inner <= r <= r_outer:
                    q[i][j] = 1
        return q

    def make_cross_quadrant(width, arm_length, center=(0.5, 0.5)):
        """Cross/plus shape."""
        q = [[0] * half for _ in range(half)]
        for i in range(half):
            for j in range(i, half):
                x = (j + 0.5) / N
                y = (i + 0.5) / N
                dx = abs(x - center[0])
                dy = abs(y - center[1])
                if dx <= width / 2 and dy <= arm_length / 2:
                    q[i][j] = 1
                elif dy <= width / 2 and dx <= arm_length / 2:
                    q[i][j] = 1
        return q

    # --- Initial parameter sweep ---
    best_score = -1.0
    best_quadrant = None

    # Ring designs
    ring_params = [
        (0.10, 0.30), (0.15, 0.35), (0.20, 0.40), (0.25, 0.45),
        (0.12, 0.32), (0.18, 0.38), (0.22, 0.42), (0.28, 0.48),
        (0.08, 0.28), (0.14, 0.34), (0.20, 0.40), (0.26, 0.46),
    ]
    for ri, ro in ring_params:
        if time.time() > deadline - 80:
            break
        q = make_ring_quadrant(ri, ro)
        cell = build_cell_from_quadrant(q)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]

    # Square ring designs
    sq_params = [
        (0.10, 0.30), (0.15, 0.35), (0.20, 0.40), (0.25, 0.45),
        (0.12, 0.32), (0.18, 0.38), (0.22, 0.42),
    ]
    for ri, ro in sq_params:
        if time.time() > deadline - 60:
            break
        q = make_square_ring_quadrant(ri, ro)
        cell = build_cell_from_quadrant(q)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]

    # Cross designs
    cross_params = [
        (0.10, 0.35), (0.15, 0.40), (0.20, 0.45), (0.12, 0.38),
    ]
    for w, al in cross_params:
        if time.time() > deadline - 50:
            break
        q = make_cross_quadrant(w, al)
        cell = build_cell_from_quadrant(q)
        score = evaluate_gap(cell)
        if score > best_score:
            best_score = score
            best_quadrant = [row[:] for row in q]

    if best_quadrant is None:
        # Fallback: simple ring
        best_quadrant = make_ring_quadrant(0.2, 0.4)
        best_score = evaluate_gap(build_cell_from_quadrant(best_quadrant))

    # --- Local search: flip pixels in the quadrant ---
    current_quadrant = [row[:] for row in best_quadrant]
    current_score = best_score

    positions = [(i, j) for i in range(half) for j in range(i, half)]
    import random
    rng = random.Random(42)

    # Multiple passes of local search with random restarts
    for sweep in range(30):
        if time.time() > deadline - 15:
            break

        shuffled_pos = positions[:]
        rng.shuffle(shuffled_pos)

        improved_this_sweep = False
        for (i, j) in shuffled_pos:
            if time.time() > deadline - 12:
                break

            old_val = current_quadrant[i][j]
            current_quadrant[i][j] = 1 - old_val
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_score = evaluate_gap(new_cell)

            if new_score > current_score:
                current_score = new_score
                improved_this_sweep = True
                if new_score > best_score:
                    best_score = new_score
                    best_quadrant = [row[:] for row in current_quadrant]
            else:
                current_quadrant[i][j] = old_val

        if not improved_this_sweep and sweep > 10:
            # Random restart: flip a few random pixels
            for _ in range(8):
                if time.time() > deadline - 12:
                    break
                i, j = rng.choice(positions)
                current_quadrant[i][j] = 1 - current_quadrant[i][j]
            new_cell = build_cell_from_quadrant(current_quadrant)
            new_score = evaluate_gap(new_cell)
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_quadrant = [row[:] for row in current_quadrant]
            else:
                # Revert
                for _ in range(8):
                    i, j = rng.choice(positions)
                    current_quadrant[i][j] = 1 - current_quadrant[i][j]

    # Build final cell
    final_cell = build_cell_from_quadrant(best_quadrant)

    # Verify and enforce symmetry
    for i in range(N):
        for j in range(N):
            if final_cell[i][j] != final_cell[N - 1 - i][j]:
                final_cell[N - 1 - i][j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[i][N - 1 - j]:
                final_cell[i][N - 1 - j] = final_cell[i][j]
            if final_cell[i][j] != final_cell[j][i]:
                final_cell[j][i] = final_cell[i][j]

    return final_cell
