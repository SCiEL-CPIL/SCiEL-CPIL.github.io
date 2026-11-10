import numpy as np
import time
import math

def design(problem):
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    start_time = time.time()
    deadline = start_time + 215.0  # safety margin under 230s kill

    # ---------- Symmetry orbits (C4v) ----------
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            pts = [(i, j), (size-1-i, j), (i, size-1-j), (size-1-i, size-1-j),
                   (j, i), (size-1-j, j), (j, size-1-i), (size-1-j, size-1-i)]
            pts = list(dict.fromkeys(pts))
            for p in pts:
                seen.add(p)
            orbits.append(pts)

    # ---------- Fast TM evaluator ----------
    # TM: matrix A = (K_i + G_i)·(K_j + G_j) * (1/eps)_{G_i-G_j}
    half = 4
    G = [(h, k) for h in range(-half, half+1) for k in range(-half, half+1)]
    N = len(G)
    Gx = np.array([g[0] for g in G], dtype=float)
    Gy = np.array([g[1] for g in G], dtype=float)

    # Fourier index of each G (mod size)
    Gx_mod = Gx % size
    Gy_mod = Gy % size

    # k-points in irreducible BZ
    kpts = []
    kpts.append((0.0, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        kpts.append((0.0, t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        kpts.append((t, 0.5 - t))
    kpts.append((0.5, 0.5))

    # Precompute per-k dot matrix D[k][i,j] = (Kx_i)(Kx_j)+(Ky_i)(Ky_j)
    D_list = []
    for kx, ky in kpts:
        Kx = Gx + kx
        Ky = Gy + ky
        D = np.outer(Kx, Kx) + np.outer(Ky, Ky)
        D_list.append(D)

    def evaluate(cell):
        """cell: 0/1 array, 1=dielectric. Returns gap-midgap ratio (bands 4,5)."""
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        fft_inv = np.fft.fft2(inv_eps) / (size*size)
        # C[i,j] = (1/eps)_{G_i - G_j}
        C = fft_inv[np.ix_(Gx_mod.astype(int), Gy_mod.astype(int))]

        max4 = 0.0
        min5 = float('inf')
        for D in D_list:
            H = D * C
            H = 0.5 * (H + H.T.conj())
            ev = np.linalg.eigvalsh(H)
            if ev.shape[0] < 5:
                continue
            w4 = math.sqrt(max(0.0, float(ev[3])))
            w5 = math.sqrt(max(0.0, float(ev[4])))
            if w4 > max4:
                max4 = w4
            if w5 < min5:
                min5 = w5
        if min5 == float('inf'):
            return 0.0
        mid = (min5 + max4) / 2.0
        if mid <= 0:
            return 0.0
        return 2.0 * (min5 - max4) / mid

    # ---------- Initial design: dielectric background + air holes ----------
    def make_initial():
        cell = np.ones((size, size), dtype=int)  # dielectric background
        # Air holes (0) at quarter-points
        centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        for cy, cx in centers:
            for i in range(size):
                for j in range(size):
                    d = (i - cy)**2 + (j - cx)**2
                    if d < 25:  # radius ~5
                        cell[i, j] = 0
        # small central air hole
        for i in range(15, 17):
            for j in range(15, 17):
                cell[i, j] = 0
        return cell

    current = make_initial()
    cur = evaluate(current)
    best = current.copy()
    best_score = cur

    # ---------- Simulated annealing over orbits ----------
    T = 1.0
    T_min = 0.02
    cool = 0.9985
    n_orbits = len(orbits)

    while time.time() < deadline:
        # pick random orbit
        o = np.random.randint(n_orbits)
        pts = orbits[o]
        # flip
        for (pi, pj) in pts:
            current[pi, pj] = 1 - current[pi, pj]
        new = evaluate(current)
        d = new - cur
        if d > 0 or np.random.random() < math.exp(d / max(T, 1e-9)):
            cur = new
            if new > best_score:
                best_score = new
                best = current.copy()
        else:
            # revert
            for (pi, pj) in pts:
                current[pi, pj] = 1 - current[pi, pj]
        T = max(T_min, T * cool)

    # Final safety: ensure symmetry of best (it is, by construction)
    return best.tolist()
