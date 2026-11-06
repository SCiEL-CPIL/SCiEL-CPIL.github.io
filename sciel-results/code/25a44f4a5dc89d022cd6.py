"""Rewritten photonic-crystal unit-cell designer with a faster, more accurate local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    size = problem["pixels"]          # 32
    eps_low = problem["eps_low"]      # 1
    eps_high = problem["eps_high"]    # 12

    start_time = time.time()
    deadline = start_time + 115.0     # keep a margin inside the 120 s budget

    # ------------------------------------------------------------------
    # Symmetry helpers
    # ------------------------------------------------------------------
    def quadrant_to_full(q):
        """q is a 16x16 top-left quadrant; return the full 32x32 D4-symmetric cell."""
        q = np.asarray(q, dtype=np.int8)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])

    def full_to_quadrant(cell):
        return np.asarray(cell[:16, :16], dtype=np.int8)

    # ------------------------------------------------------------------
    # Plane-wave constants
    # ------------------------------------------------------------------
    N = 32
    npw = 8                              # (2*npw+1)^2 = 289 plane waves, matching evaluator
    mx = np.arange(-npw, npw + 1, dtype=np.int64)
    my = np.arange(-npw, npw + 1, dtype=np.int64)
    Gx = np.tile(mx, len(my))            # shape (nG,)
    Gy = np.repeat(my, len(mx))
    G2 = (2 * np.pi) ** 2 * (Gx ** 2 + Gy ** 2)
    nG = len(Gx)

    # Precompute the difference-index map: for each (i, j) in the G grid,
    # find the index of (G_i - G_j) if it lies inside the window, else -1.
    Gx_grid = Gx.reshape(len(my), len(mx))
    Gy_grid = Gy.reshape(len(my), len(mx))
    diff_idx = np.full((nG, nG), -1, dtype=np.int64)
    for ii in range(nG):
        for jj in range(nG):
            dx = Gx[ii] - Gx[jj]
            dy = Gy[ii] - Gy[jj]
            if -npw <= dx <= npw and -npw <= dy <= npw:
                # index in the flattened (my-major, mx-minor) array
                row = dy + npw
                col = dx + npw
                diff_idx[ii, jj] = row * len(mx) + col

    # 30 k-points in the irreducible BZ (0 <= ky <= kx <= pi)
    k_points = [
        (0.0, 0.0),
        (np.pi, 0.0),
        (np.pi / 2, np.pi / 2),
        (np.pi / 2, 0.0),
        (np.pi / 4, np.pi / 4),
        (3 * np.pi / 8, 0.0),
        (3 * np.pi / 8, np.pi / 8),
        (np.pi / 3, np.pi / 3),
        (np.pi / 3, 0.0),
        (2 * np.pi / 3, 0.0),
        (np.pi / 2, np.pi / 4),
        (np.pi / 2, np.pi / 6),
        (np.pi / 2, np.pi / 3),
        (5 * np.pi / 8, 0.0),
        (5 * np.pi / 8, np.pi / 8),
        (7 * np.pi / 8, 0.0),
        (np.pi, np.pi / 4),
        (np.pi, np.pi / 2),
        (np.pi, np.pi / 3),
        (3 * np.pi / 4, np.pi / 4),
        (3 * np.pi / 4, 0.0),
        (np.pi / 4, 0.0),
        (np.pi / 6, 0.0),
        (np.pi / 3, np.pi / 6),
        (np.pi / 2, np.pi / 8),
        (np.pi / 2, 3 * np.pi / 8),
        (3 * np.pi / 4, np.pi / 8),
        (5 * np.pi / 6, 0.0),
        (np.pi, np.pi / 6),
        (np.pi, 3 * np.pi / 4),
    ]

    # Precompute exp(-i G . r_ij) for the pixel grid, used for Fourier coefficients
    x = (np.arange(N, dtype=np.float64) + 0.5) / N
    y = (np.arange(N, dtype=np.float64) + 0.5) / N
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Phase matrix: shape (nG, N, N)
    phase = np.exp(-1j * (2 * np.pi * (Gx[:, None, None] * X[None, :, :] +
                                       Gy[:, None, None] * Y[None, :, :])))

    # ------------------------------------------------------------------
    # Fast gap evaluation
    # ------------------------------------------------------------------
    def compute_gap(cell_2d):
        """Return the gap-midgap ratio 2*(min_w5 - max_w4)/(min_w5 + max_w4)."""
        eps = np.where(cell_2d == 1, eps_high, eps_low, dtype=np.float64)
        inv_eps = 1.0 / eps

        # Fourier coefficients of 1/eps: shape (nG,)
        inv_eps_F = np.sum(inv_eps[None, :, :] * phase, axis=(1, 2)) / (N * N)

        max_w4 = 0.0
        min_w5 = np.inf

        for kx, ky in k_points:
            kG2 = (kx + 2 * np.pi * Gx) ** 2 + (ky + 2 * np.pi * Gy) ** 2

            # Build the matrix M[i,j] = kG2[i] * inv_eps_F[diff_idx[i,j]]
            # Only entries where diff_idx >= 0 are non-zero.
            valid = diff_idx >= 0
            coeffs = np.where(valid, inv_eps_F[diff_idx], 0.0)
            M = kG2[:, None] * coeffs

            # Ensure Hermitian
            M = 0.5 * (M + M.conj().T)

            eigvals = np.linalg.eigvalsh(M)
            eigvals = np.real(eigvals)
            omega = np.sqrt(np.maximum(eigvals, 0.0))

            if len(omega) >= 5:
                w4 = omega[3]
                w5 = omega[4]
                if w4 > max_w4:
                    max_w4 = w4
                if w5 < min_w5:
                    min_w5 = w5

        if max_w4 <= 0 or min_w5 <= 0:
            return 0.0
        gap = min_w5 - max_w4
        midgap = 0.5 * (min_w5 + max_w4)
        if midgap <= 0:
            return 0.0
        return 2.0 * gap / midgap

    # ------------------------------------------------------------------
    # Initial design: centered dielectric disk in the quadrant
    # ------------------------------------------------------------------
    q = np.zeros((16, 16), dtype=np.int8)
    cx, cy = 7.5, 7.5
    for i in range(16):
        for j in range(16):
            dx = i - cx
            dy = j - cy
            if dx * dx + dy * dy < 3.5 ** 2:
                q[i, j] = 1

    cell = quadrant_to_full(q)
    current_score = compute_gap(cell)
    best_score = current_score
    best_cell = cell.copy()

    # ------------------------------------------------------------------
    # Local search with simulated annealing
    # ------------------------------------------------------------------
    rng = np.random.RandomState(12345)
    q_flat = q.flatten()
    n_q = q_flat.size  # 256

    temperature = 0.02
    min_temp = 0.0005
    cool_rate = 0.997
    max_no_improve = 300
    no_improve = 0
    eval_count = 0

    while time.time() < deadline - 3.0:
        # Choose 1 or 2 random pixels to flip
        n_flips = 1 if rng.random() < 0.7 else 2
        flip_idx = rng.choice(n_q, size=n_flips, replace=False)
        orig_vals = q_flat[flip_idx].copy()
        q_flat[flip_idx] = 1 - q_flat[flip_idx]

        q_new = q_flat.reshape(16, 16)
        cell_new = quadrant_to_full(q_new)
        score_new = compute_gap(cell_new)
        eval_count += 1

        delta = score_new - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-9)):
            current_score = score_new
            if score_new > best_score:
                best_score = score_new
                best_cell = cell_new.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            q_flat[flip_idx] = orig_vals
            no_improve += 1

        temperature *= cool_rate
        if temperature < min_temp:
            temperature = min_temp

        if no_improve > max_no_improve:
            temperature = max(temperature, 0.01)
            no_improve = 0

    # ------------------------------------------------------------------
    # Final polish: exhaustive single-flip scan from the best cell
    # ------------------------------------------------------------------
    if time.time() < deadline - 1.0:
        best_q = full_to_quadrant(best_cell).copy()
        best_q_score = compute_gap(quadrant_to_full(best_q))

        for i in range(16):
            for j in range(16):
                if time.time() > deadline - 0.5:
                    break
                best_q[i, j] = 1 - best_q[i, j]
                cell_try = quadrant_to_full(best_q)
                score_try = compute_gap(cell_try)
                eval_count += 1
                if score_try > best_q_score:
                    best_q_score = score_try
                    best_score = max(best_score, score_try)
                    best_cell = cell_try.copy()
                else:
                    best_q[i, j] = 1 - best_q[i, j]

    return best_cell.tolist()
# EVOLVE-BLOCK-END
