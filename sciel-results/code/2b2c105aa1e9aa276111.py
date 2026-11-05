import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    MARGIN = 8.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE - MARGIN - elapsed()

    n = 11
    # Unit hexagon vertices (pointy-right, angle 0)
    base = np.array([[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)])

    # Flower pattern seed: 1 center + 6 at sqrt(3) + 4 at 3 (30+60k)
    centers0 = np.array([[0.0, 0.0]])
    r1 = np.sqrt(3)
    for k in range(6):
        a = np.deg2rad(60 * k)
        centers0 = np.vstack([centers0, [r1 * np.cos(a), r1 * np.sin(a)]])
    r2 = 3.0
    for k in range(4):
        a = np.deg2rad(30 + 60 * k)
        centers0 = np.vstack([centers0, [r2 * np.cos(a), r2 * np.sin(a)]])
    centers0 = centers0[:n]
    rot0 = np.zeros(n)

    def pack_params(theta, centers, rots):
        """Return (side, maxproj). Centers (n,2), rots (n,), theta scalar."""
        na = np.deg2rad(30 + 60 * np.arange(6)) + theta
        normals = np.column_stack([np.cos(na), np.sin(na)])  # (6,2)
        c = np.cos(rots)
        s = np.sin(rots)
        rotm = np.empty((n, 2, 2))
        rotm[:, 0, 0] = c
        rotm[:, 0, 1] = -s
        rotm[:, 1, 0] = s
        rotm[:, 1, 1] = c
        verts = np.einsum('ijk,kl->ijl', rotm, base.T) + centers[:, None, :]  # (n,6,2)
        dots = np.einsum('ijl,kl->ijk', verts, normals.T)  # (n,6,6)
        mp = float(np.max(np.abs(dots)))
        return 2.0 * mp / np.sqrt(3), mp

    def smooth(params, beta):
        theta = params[0]
        centers = params[1:1 + 2 * n].reshape(n, 2)
        rots = params[1 + 2 * n:]
        na = np.deg2rad(30 + 60 * np.arange(6)) + theta
        normals = np.column_stack([np.cos(na), np.sin(na)])
        c = np.cos(rots)
        s = np.sin(rots)
        rotm = np.empty((n, 2, 2))
        rotm[:, 0, 0] = c
        rotm[:, 0, 1] = -s
        rotm[:, 1, 0] = s
        rotm[:, 1, 1] = c
        verts = np.einsum('ijk,kl->ijl', rotm, base.T) + centers[:, None, :]
        dots = np.einsum('ijl,kl->ijk', verts, normals.T)  # (n,6,6)
        sq = (dots * dots).ravel()
        mx = float(np.max(sq))
        if mx <= 0.0:
            return 0.0
        sh = np.clip((sq - mx) * beta, -600.0, 600.0)
        lse = mx + np.log(np.sum(np.exp(sh)) / sq.size) / beta
        return 2.0 * np.sqrt(lse) / np.sqrt(3)

    def flatten(theta, centers, rots):
        return np.concatenate([[theta], centers.ravel(), rots])

    def seed_params():
        return flatten(0.0, centers0.copy(), rot0.copy())

    best_side = None
    best_p = None
    p = seed_params()
    s, _ = pack_params(p[0], p[1:1 + 2 * n].reshape(n, 2), p[1 + 2 * n:])
    best_side = s
    best_p = p.copy()

    # Multi-start: perturb seed with increasing noise, smooth objective + BFGS
    rng = np.random.RandomState(12345)
    seeds = [0.0]
    for amp in [0.05, 0.1, 0.2, 0.35, 0.6, 1.0]:
        seeds.append(amp)
    for amp in seeds:
        if remaining() < 15.0:
            break
        if amp == 0.0:
            p0 = seed_params()
        else:
            p0 = seed_params().copy()
            # perturb centers and rotations (keep theta)
            p0[1:1 + 2 * n] += amp * rng.randn(2 * n)
            p0[1 + 2 * n:] += (amp * 0.5) * rng.randn(n)
        for beta in [20.0, 40.0]:
            if remaining() < 8.0:
                break
            try:
                res = minimize(smooth, p0, args=(beta,), method='BFGS',
                               options={'maxiter': 300, 'gtol': 1e-6, 'disp': False})
                s, _ = pack_params(res.x[0], res.x[1:1 + 2 * n].reshape(n, 2), res.x[1 + 2 * n:])
                if best_side is None or s < best_side - 1e-12:
                    best_side = s
                    best_p = res.x.copy()
            except Exception:
                pass

    # Final local polish on exact objective (Powell, robust to non-smoothness)
    if remaining() > 6.0:
        try:
            res = minimize(lambda pp: pack_params(pp[0], pp[1:1 + 2 * n].reshape(n, 2), pp[1 + 2 * n:])[0],
                           best_p, method='Powell',
                           options={'maxiter': 600, 'ftol': 1e-14, 'xtol': 1e-14, 'disp': False})
            s, _ = pack_params(res.x[0], res.x[1:1 + 2 * n].reshape(n, 2), res.x[1 + 2 * n:])
            if best_side is None or s < best_side - 1e-12:
                best_side = s
                best_p = res.x.copy()
        except Exception:
            pass

    # Extract
    theta = best_p[0]
    centers = best_p[1:1 + 2 * n].reshape(n, 2)
    rots = best_p[1 + 2 * n:]
    side, _ = pack_params(theta, centers, rots)

    inner = []
    for i in range(n):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(np.rad2deg(rots[i]))])

    cx = float(np.mean(centers[:, 0]))
    cy = float(np.mean(centers[:, 1]))

    return inner, (cx, cy), float(side) + 1e-9, float(np.rad2deg(theta))

if __name__ == "__main__":
    construct()
