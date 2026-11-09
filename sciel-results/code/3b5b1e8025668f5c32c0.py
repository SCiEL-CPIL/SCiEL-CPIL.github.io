"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

R = np.sqrt(3.0) / 2.0            # inradius of a unit hexagon
PHI = np.deg2rad(np.arange(0, 360, 60))   # support directions of a unit hexagon


def _support_inner(theta):
    """Support function of a unit hexagon at orientation theta (radians)."""
    return R * np.max(np.cos(theta - PHI))


def _support_outer(theta, S):
    """Support function of the outer hexagon at orientation theta, side length S."""
    return S * np.max(np.cos(theta - PHI))


def _pair_min_dist(theta_a, theta_b):
    """Minimum center distance for two unit hexagons to be disjoint."""
    return R * (np.max(np.cos(theta_a - PHI)) + np.max(np.cos(theta_b - PHI)))


def construct():
    n = N
    # initial guess: hexagonal lattice of unit hexagons
    d0 = 2.0 * R
    xs, ys = [], []
    for i in range(-n // 2, n - n // 2):
        for j in range(-n // 2, n - n // 2):
            xs.append(i * d0 + (0.5 * d0 if (j % 2) else 0.0))
            ys.append(j * d0 * np.sqrt(3.0) / 2.0)
    m = len(xs)
    x0 = np.array(xs + ys, dtype=float)
    x0 = np.concatenate([x0, np.zeros(m), [2.0 * R * np.sqrt(3.0) * np.sqrt(m)]])

    def _clearances(x):
        S = x[2 * m]
        theta = x[m:2 * m]
        pos = x[:2 * m].reshape(m, 2)
        P = np.array([_pair_min_dist(theta[i], theta[j])
                      for i in range(m) for j in range(i + 1, m)])
        dpair = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        C = dpair[np.triu_indices(m, 1)] - P
        douter = _support_outer(theta, S) - _support_inner(theta)
        C = np.concatenate([C, douter])
        return C

    def _objective(x):
        S = x[2 * m]
        C = _clearances(x)
        Cc = np.clip(C, -5.0, 5.0)
        p = 8.0
        neg = np.sum(np.minimum(Cc, 0.0) ** p)
        return -np.sum(Cc ** p) / len(Cc) + 10.0 * neg + 0.001 * S

    def _jacobian(x):
        S = x[2 * m]
        theta = x[m:2 * m]
        pos = x[:2 * m].reshape(m, 2)
        p = 8.0
        C = _clearances(x)
        Cc = np.clip(C, -5.0, 5.0)
        neg = np.minimum(Cc, 0.0)
        grad = np.zeros(2 * m + 1)
        # inner position gradient
        gpos = np.zeros((m, 2))
        for i in range(m):
            for j in range(i + 1, m):
                idx = i * (m - 1) - i * (i + 1) // 2 + (j - i - 1)
                v = pos[i] - pos[j]
                d = max(np.linalg.norm(v), 1e-8)
                w = p * (Cc[idx] ** (p - 1))
                gpos[i] += w * v / d
                gpos[j] -= w * v / d
        grad[:2 * m] = gpos.reshape(-1)
        # angle gradient (pair + outer)
        for i in range(m):
            gi = 0.0
            for j in range(i + 1, m):
                idx = i * (m - 1) - i * (i + 1) // 2 + (j - i - 1)
                w = p * (Cc[idx] ** (p - 1))
                ca, sa = np.cos(theta[i]), np.sin(theta[i])
                cb, sb = np.cos(theta[j]), np.sin(theta[j])
                da = R * np.min(-sa * np.cos(PHI) + ca * np.sin(PHI))
                db = R * np.min(-sb * np.cos(PHI) + cb * np.sin(PHI))
                gi += -w * (da + db)
            wo = p * (Cc[idx + 1] ** (p - 1)) if False else 0.0
            # outer term index
            idxo = m * (m - 1) // 2 + i
            wo = p * (Cc[idxo] ** (p - 1))
            co = np.cos(theta[i])
            so = np.sin(theta[i])
            da_in = R * np.min(-so * np.cos(PHI) + co * np.sin(PHI))
            da_out = S * np.min(-so * np.cos(PHI) + co * np.sin(PHI))
            gi += -wo * (da_out - da_in)
            grad[m + i] = gi
        # S gradient (outer terms)
        gs = 0.0
        for i in range(m):
            idxo = m * (m - 1) // 2 + i
            wo = p * (Cc[idxo] ** (p - 1))
            co = np.cos(theta[i])
            so = np.sin(theta[i])
            da_out = S * np.min(-so * np.cos(PHI) + co * np.sin(PHI))
            gs += -wo * da_out
        gs += 0.001
        grad[2 * m] = gs
        # penalty gradient
        for i in range(m):
            for j in range(i + 1, m):
                idx = i * (m - 1) - i * (i + 1) // 2 + (j - i - 1)
                if C[idx] < 0.0:
                    v = pos[i] - pos[j]
                    d = max(np.linalg.norm(v), 1e-8)
                    w = 10.0 * p * (neg[idx] ** (p - 1))
                    grad[2 *i] += w * v[0] / d
                    grad[2*i+1] += w * v[1] / d
                    grad[2*j] -= w * v[0] / d
                    grad[2*j+1] -= w * v[1] / d
        return grad

    res = minimize(_objective, x0, method='trust-constr', jac=_jacobian,
                   options={'maxiter': 250, 'gtol': 1e-6, 'verbose': 0})
    x = res.x
    S = max(float(x[2 * m]), 2.0 * R)
    theta = x[m:2 * m]
    pos = x[:2 * m].reshape(m, 2)

    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(np.rad2deg(theta[i]))]
             for i in range(m)]
    return inner, (0.0, 0.0), float(S), 0.0
# EVOLVE-BLOCK-END
