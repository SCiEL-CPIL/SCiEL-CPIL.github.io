"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Symmetric 32x32 TM photonic-crystal unit cell maximizing the 4-5 band gap.

    Patterns are built from the top-left 16x16 quadrant and reflected, guaranteeing
    full square (D4) symmetry: cell == row-reverse == col-reverse == transpose.
    A fast plane-wave TM estimator scores candidates and a hill-climb refines the
    best under an explicit deadline guard.
    """
    size = problem["pixels"]          # 32
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    start = time.time()
    DEADLINE = start + 110.0          # safety margin vs 120s CPU / 230s wall kill
    H = size // 2                      # 16

    def from_quadrant(q):
        """q: H x H uint8 (top-left). Reflect to full size x size, D4-symmetric."""
        right = q[:, ::-1]
        top = np.concatenate([q, right], axis=1)          # H x size
        bottom = top[::-1, :]                             # size x size
        return bottom.astype(np.uint8)

    def make_ring(inner, outer):
        yy, xx = np.mgrid[0:H, 0:H]
        c = H - 0.5
        r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        q = ((r >= inner) & (r <= outer)).astype(np.uint8)
        return from_quadrant(q)

    def make_blobs(cx, cy, rad):
        yy, xx = np.mgrid[0:H, 0:H]
        q = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= rad).astype(np.uint8)
        return from_quadrant(q)

    def make_corner(cx, cy, rad):
        yy, xx = np.mgrid[0:H, 0:H]
        q = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= rad).astype(np.uint8)
        return from_quadrant(q)

    def make_cross(arm_w, arm_len):
        yy, xx = np.mgrid[0:H, 0:H]
        c = H - 0.5
        vert = np.abs(xx - c) <= arm_w / 2.0
        horiz = np.abs(yy - c) <= arm_w / 2.0
        q = ((vert | horiz) & (np.maximum(np.abs(xx - c), np.abs(yy - c)) <= arm_len)).astype(np.uint8)
        return from_quadrant(q)

    # ---- fast TM plane-wave gap estimator ----
    def _eps_inv_fourier(cell, N):
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        e_inv = 1.0 / eps
        F = np.fft.fft2(e_inv, s=(N, N))
        F = np.fft.fftshift(F)
        return F

    def _kpoints(nk=5):
        kpts = []
        for i in range(nk + 1):
            ky = np.pi * i / nk
            for j in range(i + 1):
                kx = np.pi * j / nk
                kpts.append((kx, ky))
        kpts.append((np.pi * 0.3, np.pi * 0.55))
        kpts.append((np.pi * 0.15, np.pi * 0.35))
        return kpts

    def _bands(kx, ky, F, N, nband=6):
        m = np.arange(-(N // 2), N // 2 + 1)
        Gx, Gy = np.meshgrid(m, m, indexing='ij')
        Gx = 2 * np.pi * Gx
        Gy = 2 * np.pi * Gy
        G = np.stack([Gx.ravel(), Gy.ravel()], axis=1)
        idx = m + N // 2
        nG = G.shape[0]
        # D[p,q] = (1/eps)_{G_p - G_q}
        D = np.empty((nG, nG), dtype=float)
        for p in range(nG):
            for q in range(nG):
                dm = idx[p, 0] - idx[q, 0]
                dn = idx[p, 1] - idx[q, 1]
                D[p, q] = F[(dm + N // 2) % N, (dn + N // 2) % N].real
        K = np.sum((G + np.array([kx, ky])) ** 2, axis=1)
        # TM: M[p,q] = (1/eps)_{p-q} * |G_p + k|^2  (row scaling by K[p])
        M = D * K[:, None]
        M = (M + M.T) * 0.5
        w2 = np.linalg.eigvalsh(M)
        w2 = w2[w2 > 1e-9]
        return np.sqrt(w2)[:nband]

    def gap_score(cell, nband=6, nk=5, N=17):
        F = _eps_inv_fourier(cell, N)
        kpts = _kpoints(nk)
        max_b4 = 0.0
        min_b5 = np.inf
        for (kx, ky) in kpts:
            b = _bands(kx, ky, F, N, nband)
            if len(b) >= 5:
                max_b4 = max(max_b4, b[3])
                min_b5 = min(min_b5, b[4])
        if min_b5 == np.inf:
            return -1.0
        return 2.0 * (min_b5 - max_b4) / (min_b5 + max_b4)

    # ---- candidate generators ----
    def candidates():
        c = []
        for inner in np.arange(0.0, 13.0, 1.0):
            for outer in np.arange(inner + 2.0, 16.0, 1.0):
                c.append(make_ring(inner, outer))
        for cx in np.arange(2.0, 14.0, 2.0):
            for cy in np.arange(2.0, 14.0, 2.0):
                for rad in np.arange(2.0, 7.0, 1.0):
                    c.append(make_blobs(cx, cy, rad))
        for cx in np.arange(0.5, 6.0, 1.0):
            for cy in np.arange(0.5, 6.0, 1.0):
                for rad in np.arange(2.0, 6.0, 1.0):
                    c.append(make_corner(cx, cy, rad))
        for w in np.arange(4.0, 12.0, 2.0):
            for ln in np.arange(6.0, 16.0, 2.0):
                c.append(make_cross(w, ln))
        return c

    best_cell = None
    best_score = -1.0

    cands = candidates()
    for cell in cands:
        if time.time() > DEADLINE:
            break
        s = gap_score(cell)
        if s > best_score:
            best_score = s
            best_cell = cell.copy()

    # hill-climb over ring and blob parameters around the best
    if best_cell is not None and time.time() < DEADLINE:
        cur = best_cell.copy()
        cur_score = best_score
        for _ in range(60):
            if time.time() > DEADLINE:
                break
            improved = False
            for inner in np.arange(0.0, 14.0, 1.0):
                for outer in np.arange(inner + 1.5, 16.0, 1.0):
                    if time.time() > DEADLINE:
                        break
                    cand = make_ring(inner, outer)
                    s = gap_score(cand)
                    if s > cur_score + 1e-4:
                        cur, cur_score, improved = cand, s, True
                        break
                if time.time() > DEADLINE or improved:
                    break
            if not improved:
                for cx in np.arange(1.0, 14.0, 1.0):
                    for cy in np.arange(1.0, 14.0, 1.0):
                        for rad in np.arange(1.5, 7.0, 0.5):
                            if time.time() > DEADLINE:
                                break
                            cand = make_blobs(cx, cy, rad)
                            s = gap_score(cand)
                            if s > cur_score + 1e-4:
                                cur, cur_score, improved = cand, s, True
                                break
                        if time.time() > DEADLINE or improved:
                            break
                    if time.time() > DEADLINE or improved:
                        break
            if not improved:
                break
        if cur_score > best_score:
            best_cell, best_score = cur, cur_score

    if best_cell is None:
        best_cell = make_ring(4.0, 9.0)

    return [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
# EVOLVE-BLOCK-END
