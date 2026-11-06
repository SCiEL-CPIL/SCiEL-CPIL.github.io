"""Optimized design: super-ellipse dielectric rod with fast band computation."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def _build_cell(size, p, r):
    """Build a symmetric super-ellipse dielectric rod."""
    half = size / 2.0
    xs = np.abs(np.arange(size) + 0.5 - half) / size
    ys = np.abs(np.arange(size) + 0.5 - half) / size
    xx = xs[:, None] ** p
    yy = ys[None, :] ** p
    mask = (xx + yy) < (r ** p)
    cell = mask.astype(np.int8)
    # Enforce 4-fold symmetry
    cell = np.minimum.reduce([cell, cell[::-1].copy(), cell[:, ::-1].copy(), cell[::-1, ::-1].copy()])
    return cell


def _compute_bands_fast(cell, size, n_pw=15):
    """Compute TM bands at Gamma point using vectorized plane-wave method.
    Returns (omega_4, omega_5) as 1-indexed band indices.
    """
    eps = np.where(cell == 1, 12.0, 1.0)
    inv_eps = 1.0 / eps

    N = n_pw
    G_vals = np.arange(-N, N + 1, dtype=np.float64)
    nG = len(G_vals)
    Gm, Gn = np.meshgrid(G_vals, G_vals, indexing='ij')
    G2 = Gm * Gm + Gn * Gn

    # Fourier coefficients of 1/epsilon
    i_idx = (np.arange(size) + 0.5) / size
    j_idx = (np.arange(size) + 0.5) / size
    # f_G = (1/size^2) sum_{i,j} inv_eps[i,j] exp(-2pi i (Gm*x_i + Gn*y_j))
    # Vectorized: use matrix multiply
    # phase = -2pi * (Gm[:,None,None]*i_idx[None,:,None] + Gn[:,:,None]*j_idx[None,None,:])
    # To save memory, do in chunks
    f_G = np.zeros((nG, nG), dtype=np.complex128)
    chunk = 32
    for start in range(0, nG * nG, chunk):
        end = min(start + chunk, nG * nG)
        gm_idx = np.arange(start, end) // nG
        gn_idx = np.arange(start, end) % nG
        gm_vals = G_vals[gm_idx]
        gn_vals = G_vals[gn_idx]
        # For each (gm, gn) in this chunk, compute f
        for k in range(end - start):
            gm = gm_vals[k]
            gn = gn_vals[k]
            phase_x = np.exp(-2j * np.pi * gm * i_idx)
            phase_y = np.exp(-2j * np.pi * gn * j_idx)
            f_G[gm_idx[k], gn_idx[k]] = np.sum(inv_eps * phase_x[:, None] * phase_y[None, :]) / (size * size)

    # Build TM matrix at Gamma point
    # A_{G,G'} = (delta_{G,G'} + G^2) delta_{G,G'} + (G-G')^2 f_{G-G'}
    A = np.zeros((nG * nG, nG * nG), dtype=np.complex128)
    # Diagonal
    diag_idx = np.arange(nG * nG)
    for gi in range(nG):
        for gj in range(nG):
            idx = gi * nG + gj
            A[idx, idx] = 1.0 + G2[gi, gj]
    # Off-diagonal: vectorized
    # For each pair (G, G'), contribution is (G-G')^2 * f_{G-G'}
    # Let's build it properly
    Gm_flat = Gm.ravel()
    Gn_flat = Gn.ravel()
    G2_flat = G2.ravel()
    # (G-G')^2 for all pairs
    dGm = Gm_flat[:, None] - Gm_flat[None, :]
    dGn = Gn_flat[:, None] - Gn_flat[None, :]
    dG2 = dGm * dGm + dGn * dGn
    # f_{G-G'}: need f at index (G-G')
    # G-G' = (Gm[i]-Gm[j], Gn[i]-Gn[j])
    # Index in f_G: (gm_i - gm_j + N, gn_i - gn_j + N)
    gm_idx_flat = np.arange(nG * nG) // nG
    gn_idx_flat = np.arange(nG * nG) % nG
    dgm = gm_idx_flat[:, None] - gm_idx_flat[None, :]
    dgn = gn_idx_flat[:, None] - gn_idx_flat[None, :]
    f_diff = np.zeros_like(dG2)
    valid = (np.abs(dgm) <= N) & (np.abs(dgn) <= N)
    f_diff[valid] = f_G[dgm[valid] + N, dgn[valid] + N]
    A = A + dG2 * f_diff
    # Make symmetric (should be Hermitian)
    A = (A + A.conj().T) / 2.0

    try:
        eigvals = np.linalg.eigvalsh(A)
        eigvals = np.sort(np.real(eigvals))
        omega = np.sqrt(np.maximum(eigvals, 0.0))
        if len(omega) >= 5:
            return omega[3], omega[4]
    except:
        pass
    return np.inf, -np.inf


def _score_design(size, p, r):
    """Compute the gap-midgap ratio for a given super-ellipse parameter."""
    cell = _build_cell(size, p, r)
    w4, w5 = _compute_bands_fast(cell, size)
    if w5 <= 0 or w4 <= 0 or not np.isfinite(w4) or not np.isfinite(w5):
        return -1.0
    gap = w5 - w4
    mid = w5 + w4
    if mid <= 0:
        return -1.0
    return 2.0 * gap / mid


def design(problem):
    size = problem["pixels"]
    deadline = time.time() + 200  # safety margin before 230s kill

    # Start with a reasonable super-ellipse (squarish rod)
    best_p = 3.0
    best_r = 0.35
    best_score = -1.0

    # Coarse grid search over p and r
    p_vals = [2.0, 2.5, 3.0, 3.5, 4.0]
    r_vals = [0.32, 0.34, 0.36, 0.38, 0.40]

    for p in p_vals:
        if time.time() > deadline - 40:
            break
        for r in r_vals:
            if time.time() > deadline - 40:
                break
            try:
                s = _score_design(size, p, r)
                if s > best_score:
                    best_score = s
                    best_p = p
                    best_r = r
            except:
                pass

    # Local refinement around the best
    if time.time() < deadline - 50:
        # Fine-tune p
        p_step = 0.25
        for _ in range(3):
            improved = False
            for dp in [-p_step, p_step]:
                p_try = best_p + dp
                if p_try < 1.5 or p_try > 8:
                    continue
                try:
                    s = _score_design(size, p_try, best_r)
                    if s > best_score:
                        best_score = s
                        best_p = p_try
                        improved = True
                except:
                    pass
            p_step *= 0.5
            if not improved:
                break

        # Fine-tune r
        r_step = 0.015
        for _ in range(3):
            improved = False
            for dr in [-r_step, r_step]:
                r_try = best_r + dr
                if r_try < 0.2 or r_try > 0.45:
                    continue
                try:
                    s = _score_design(size, best_p, r_try)
                    if s > best_score:
                        best_score = s
                        best_r = r_try
                        improved = True
                except:
                    pass
            r_step *= 0.5
            if not improved:
                break

    # Build final cell
    cell_np = _build_cell(size, best_p, best_r)
    cell = [[int(cell_np[i][j]) for j in range(size)] for i in range(size)]
    return cell
# EVOLVE-BLOCK-END
