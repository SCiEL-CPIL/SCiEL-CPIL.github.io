"""Derivative-free Nelder-Mead + pair-unstick local search for 200-circle packing."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START

def construct():
    """Return centers of N circles inside [-1/2, 1/2]^2 maximising common radius."""
    t0 = time.time()
    DEADLINE = 143.0  # safety margin before 150 s kill

    def _remaining():
        return DEADLINE - (time.time() - t0)

    # ---------- exact radius ----------
    def _radius(c):
        wall = 0.5 - np.max(np.abs(c), axis=1)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(min(np.min(wall), np.min(d) / 2.0))

    def _bottleneck(c):
        """Return (r, kind, ia, ib)."""
        wall = 0.5 - np.max(np.abs(c), axis=1)
        i_wall = int(np.argmin(wall))
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        i_pair = int(np.argmin(d))
        j1, j2 = divmod(i_pair, N)
        if wall[i_wall] < d[j1, j2] / 2.0:
            return wall[i_wall], 'wall', i_wall, -1
        else:
            return d[j1, j2] / 2.0, 'pair', j1, j2

    # ---------- initialisation: jittered hex lattices ----------
    def _hex_jittered(rows, cols, jitter, rng):
        """Hex lattice with per-point jitter; returns (N,2) or None."""
        pts = []
        dx = 1.0 / (cols + 0.5)
        dy = dx * math.sqrt(3) / 2.0
        y0 = -0.5 + dx / 2.0
        for r in range(rows):
            y = y0 + r * dy
            if r % 2 == 1:
                x_start = -0.5 + dx
                n_x = cols - 1
            else:
                x_start = -0.5 + dx / 2.0
                n_x = cols
            for cc in range(n_x):
                if len(pts) < N:
                    pts.append([x_start + cc * dx, y])
        if len(pts) < N:
            return None
        pts = np.array(pts[:N], dtype=float)
        pts += rng.normal(0.0, jitter, size=pts.shape)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        return pts

    def _square_grid_jittered(m, jitter, rng):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
        pts += rng.normal(0.0, jitter, size=pts.shape)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        return pts

    # ---------- Nelder-Mead with per-point coordinate updates ----------
    def _nelder_mead(c, max_time, pop=40, step=0.003):
        """
        Derivative-free NM-style search.
        - Pop simplex points around c.
        - Each iteration: evaluate all, keep best, generate new candidate
          by moving one random point along one random coordinate by ±step.
        - Accept if radius improves (greedy hill-climbing over the simplex).
        """
        c = c.copy()
        best_r = _radius(c)
        # Build initial simplex: perturbations of c
        rng = np.random.RandomState(42)
        simplex = [c.copy()]
        for _ in range(pop - 1):
            cand = c.copy()
            # Perturb a few random points
            n_pts = rng.randint(3, 10)
            idxs = rng.choice(N, n_pts, replace=False)
            cand[idxs] += rng.normal(0.0, step, size=(n_pts, 2))
            cand[:, 0] = np.clip(cand[:, 0], -0.5, 0.5)
            cand[:, 1] = np.clip(cand[:, 1], -0.5, 0.5)
            simplex.append(cand)

        # Evaluate simplex
        radii = np.array([_radius(s) for s in simplex])
        best_idx = int(np.argmax(radii))
        best_c = simplex[best_idx].copy()
        best_r = radii[best_idx]

        start = time.time()
        iter_count = 0
        while _remaining() > max(3.0, max_time * 0.05) and (time.time() - start) < max_time:
            # Generate new candidate: take worst point, move one point in one coord
            worst_idx = int(np.argmin(radii))
            base = simplex[worst_idx]
            new_c = base.copy()
            pt = rng.randint(N)
            coord = rng.randint(2)
            direction = rng.choice([-1.0, 1.0])
            new_c[pt, coord] += direction * step
            new_c[:, 0] = np.clip(new_c[:, 0], -0.5, 0.5)
            new_c[:, 1] = np.clip(new_c[:, 1], -0.5, 0.5)
            new_r = _radius(new_c)

            if new_r > radii[worst_idx]:
                simplex[worst_idx] = new_c
                radii[worst_idx] = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
            else:
                # Shrink: move worst toward centroid of best half
                half = pop // 2
                order = np.argsort(radii)[::-1]
                centroid = np.mean([simplex[i] for i in order[:half]], axis=0)
                simplex[worst_idx] = (simplex[worst_idx] + centroid) / 2.0
                radii[worst_idx] = _radius(simplex[worst_idx])
                if radii[worst_idx] > best_r:
                    best_r = radii[worst_idx]
                    best_c = simplex[worst_idx].copy()

            # Occasionally re-seed worst with fresh perturbation of best
            if iter_count % 20 == 0:
                fresh = best_c.copy()
                n_pts = rng.randint(2, 6)
                idxs = rng.choice(N, n_pts, replace=False)
                fresh[idxs] += rng.normal(0.0, step * 0.5, size=(n_pts, 2))
                fresh[:, 0] = np.clip(fresh[:, 0], -0.5, 0.5)
                fresh[:, 1] = np.clip(fresh[:, 1], -0.5, 0.5)
                fr = _radius(fresh)
                if fr > radii[worst_idx]:
                    simplex[worst_idx] = fresh
                    radii[worst_idx] = fr
                    if fr > best_r:
                        best_r = fr
                        best_c = fresh.copy()

            iter_count += 1

        return best_c, best_r

    # ---------- Pair-unstick local search ----------
    def _pair_unstick(c, max_time):
        """
        Targeted local search: repeatedly find the tightest pair, then try
        to separate them by moving one circle in a direction that increases
        the minimum radius. Uses a small set of candidate directions.
        """
        c = c.copy()
        best_r = _radius(c)
        start = time.time()
        stall_count = 0
        while _remaining() > max(2.0, max_time * 0.05) and (time.time() - start) < max_time:
            r, kind, ia, ib = _bottleneck(c)
            if r >= best_r - 1e-15:
                best_r = r
            if kind == 'wall':
                # Move ia toward center
                c_new = c.copy()
                c_new[ia] = c[ia] * 0.95  # scale toward origin
                c_new[:, 0] = np.clip(c_new[:, 0], -0.5, 0.5)
                c_new[:, 1] = np.clip(c_new[:, 1], -0.5, 0.5)
                r_new = _radius(c_new)
                if r_new > best_r:
                    c = c_new
                    best_r = r_new
                    stall_count = 0
                else:
                    # Try small random nudges
                    improved = False
                    for _ in range(10):
                        c_try = c.copy()
                        c_try[ia] += np.random.randn(2) * 0.002
                        c_try[:, 0] = np.clip(c_try[:, 0], -0.5, 0.5)
                        c_try[:, 1] = np.clip(c_try[:, 1], -0.5, 0.5)
                        rt = _radius(c_try)
                        if rt > best_r:
                            c = c_try
                            best_r = rt
                            improved = True
                            break
                    if improved:
                        stall_count = 0
                    else:
                        stall_count += 1
            else:
                # Tightest pair (ia, ib): try to separate
                dvec = c[ia] - c[ib]
                nd = np.linalg.norm(dvec)
                if nd < 1e-12:
                    # Coincident: random kick
                    c_new = c.copy()
                    c_new[ia] += np.random.randn(2) * 0.005
                    c_new[ib] -= np.random.randn(2) * 0.005
                    c_new[:, 0] = np.clip(c_new[:, 0], -0.5, 0.5)
                    c_new[:, 1] = np.clip(c_new[:, 1], -0.5, 0.5)
                    r_new = _radius(c_new)
                    if r_new > best_r:
                        c = c_new
                        best_r = r_new
                        stall_count = 0
                    else:
                        stall_count += 1
                    continue

                dir_ = dvec / nd
                # Candidate moves: push ia along dir_, ib against dir_
                # Try several magnitudes and also perpendicular kicks
                candidates = []
                for mag in [0.001, 0.002, 0.005, 0.01]:
                    # Direct separation
                    c_try = c.copy()
                    c_try[ia] += dir_ * mag
                    c_try[ib] -= dir_ * mag
                    candidates.append(c_try)
                    # Perpendicular kick on ia
                    perp = np.array([-dir_[1], dir_[0]])
                    c_try2 = c.copy()
                    c_try2[ia] += perp * mag
                    c_try2[ib] -= perp * mag
                    candidates.append(c_try2)
                    # Kick only ia
                    c_try3 = c.copy()
                    c_try3[ia] += dir_ * mag * 1.5
                    candidates.append(c_try3)
                    # Kick only ib
                    c_try4 = c.copy()
                    c_try4[ib] -= dir_ * mag * 1.5
                    candidates.append(c_try4)

                best_trial = None
                best_trial_r = best_r
                for cand in candidates:
                    cand[:, 0] = np.clip(cand[:, 0], -0.5, 0.5)
                    cand[:, 1] = np.clip(cand[:, 1], -0.5, 0.5)
                    rt = _radius(cand)
                    if rt > best_trial_r:
                        best_trial_r = rt
                        best_trial = cand

                if best_trial is not None:
                    c = best_trial
                    best_r = best_trial_r
                    stall_count = 0
                else:
                    # Random fallback
                    improved = False
                    for _ in range(8):
                        c_try = c.copy()
                        c_try[ia] += np.random.randn(2) * 0.003
                        c_try[ib] += np.random.randn(2) * 0.003
                        c_try[:, 0] = np.clip(c_try[:, 0], -0.5, 0.5)
                        c_try[:, 1] = np.clip(c_try[:, 1], -0.5, 0.5)
                        rt = _radius(c_try)
                        if rt > best_r:
                            c = c_try
                            best_r = rt
                            improved = True
                            break
                    if improved:
                        stall_count = 0
                    else:
                        stall_count += 1

            if stall_count > 50:
                break

        return c, best_r

    # ---------- Main ----------
    rng = np.random.RandomState(12345)
    best_c = None
    best_r = 0.0

    # Generate diverse initial configurations
    inits = []

    # Jittered hex lattices
    for rows in range(13, 18):
        cols = math.ceil(N / rows)
        for jitter in [0.001, 0.003, 0.006, 0.012]:
            pts = _hex_jittered(rows, cols, jitter, rng)
            if pts is not None:
                inits.append(pts)

    # Jittered square grid
    m = int(math.ceil(math.sqrt(N)))
    for jitter in [0.002, 0.005, 0.01]:
        inits.append(_square_grid_jittered(m, jitter, rng))

    # Pure random
    inits.append(rng.uniform(-0.48, 0.48, (N, 2)))
    inits.append(rng.uniform(-0.48, 0.48, (N, 2)))

    # Evaluate all inits, sort by radius
    scored = []
    for init in inits:
        r = _radius(init)
        scored.append((r, init))
    scored.sort(key=lambda x: -x[0])

    # Take top-5 inits
    top_inits = [s[1] for s in scored[:5]]
    top_radii = [s[0] for s in scored[:5]]

    # For each top init: NM + pair-unstick
    for i, (init, init_r) in enumerate(zip(top_inits, top_radii)):
        if _remaining() < 20:
            break

        # NM phase
        nm_budget = min(30, _remaining() * 0.4)
        c_nm, r_nm = _nelder_mead(init, max_time=nm_budget, pop=30, step=0.003)

        # Pair-unstick phase
        ps_budget = min(20, _remaining() * 0.3)
        c_ps, r_ps = _pair_unstick(c_nm, max_time=ps_budget)

        if r_ps > best_r:
            best_r = r_ps
            best_c = c_ps

        # Second NM round with smaller step for fine-tuning
        if _remaining() > 15:
            c_nm2, r_nm2 = _nelder_mead(c_ps, max_time=min(10, _remaining() * 0.2), pop=20, step=0.001)
            if r_nm2 > best_r:
                best_r = r_nm2
                best_c = c_nm2

            # Final pair-unstick
            if _remaining() > 10:
                c_ps2, r_ps2 = _pair_unstick(c_nm2, max_time=min(8, _remaining() * 0.15))
                if r_ps2 > best_r:
                    best_r = r_ps2
                    best_c = c_ps2

    # Final micro-polish: one more pair-unstick pass
    if _remaining() > 5 and best_c is not None:
        c_final, r_final = _pair_unstick(best_c, max_time=min(4, _remaining() * 0.3))
        if r_final > best_r:
            best_r = r_final
            best_c = c_final

    return best_c

# EVOLVE-BLOCK-END
