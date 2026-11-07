"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_hermite(n, max_deg):
    """Compute physicists' Hermite polynomial H_n coefficients (ascending)."""
    if max_deg < n:
        return [0.0] * (max_deg + 1)
    if n == 0:
        c = [0.0] * (max_deg + 1)
        c[0] = 1.0
        return c
    if n == 1:
        c = [0.0] * (max_deg + 1)
        c[1] = 2.0
        return c
    h0 = [0.0] * (max_deg + 1)
    h0[0] = 1.0
    h1 = [0.0] * (max_deg + 1)
    h1[1] = 2.0
    for k in range(2, n + 1):
        h2 = [0.0] * (max_deg + 1)
        for j in range(max_deg + 1):
            if j + 1 <= max_deg:
                h2[j + 1] += 2.0 * h1[j]
            h2[j] += -2.0 * (k - 1) * h0[j]
        h0, h1 = h1, h2
    return h1


# Precompute Hermite coefficients
_H4 = np.array(_compute_hermite(4, 12))
_H8 = np.array(_compute_hermite(8, 12))
_H12 = np.array(_compute_hermite(12, 12))
_H12_0 = float(_H12[0])  # Should be 665280


def _q_from_coeffs(c0, c1, c2):
    """Compute q(z) coefficients (ascending, degree 5 in z)."""
    c3 = -(c0 + 12.0 * c1 + 1680.0 * c2) / _H12_0
    q = np.zeros(6)
    for k in range(6):
        p = 2 * k + 2
        val = 0.0
        if p <= 4:
            val += c1 * _H4[p]
        if p <= 8:
            val += c2 * _H8[p]
        if p <= 12:
            val += c3 * _H12[p]
        q[k] = val
    return q


def _eval(c0, c1, c2):
    """Compute objective value. Lower is better. Returns inf if invalid."""
    try:
        q = _q_from_coeffs(c0, c1, c2)
        # If leading coeff negative, flip (g -> -g)
        if q[5] < 0:
            q = -q
        if abs(q[5]) < 1e-15:
            return float('inf')
        # Find roots of q in z
        roots = np.roots(q[::-1])
        best_z = float('inf')
        for r in roots:
            if abs(r.imag) > 1e-9 * max(1.0, abs(r.real)):
                continue
            z = float(r.real)
            if z < 1e-10:
                continue
            # Check sign change: evaluate q at z±eps
            eps = 1e-6 * max(1.0, z)
            qm = float(np.polyval(q[::-1], z - eps))
            qp = float(np.polyval(q[::-1], z + eps))
            if qm * qp < 0:
                if z < best_z:
                    best_z = z
        if best_z == float('inf'):
            return float('inf')
        return best_z / (2.0 * np.pi)
    except Exception:
        return float('inf')


def construct():
    deadline = time.time() + 195.0
    seed = np.array([-1.13, 0.04, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _eval(seed[0], seed[1], seed[2])

    rng = np.random.default_rng(12345)

    # Multi-start Nelder-Mead
    n_starts = 15
    for si in range(n_starts):
        if time.time() > deadline - 15.0:
            break
        if si == 0:
            x = best.copy()
        elif si < 5:
            x = best + 0.05 * rng.standard_normal(3)
        elif si < 10:
            x = best + 0.2 * rng.standard_normal(3)
        else:
            x = best + 0.5 * rng.standard_normal(3)

        # Nelder-Mead
        simplex = [x.copy()]
        for d in range(3):
            p = x.copy()
            step = 0.03 * max(1.0, abs(x[d]))
            p[d] += step
            simplex.append(p)

        for it in range(800):
            if time.time() > deadline - 5.0:
                break
            vals = [_eval(s[0], s[1], s[2]) for s in simplex]
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]

            if vals[0] < best_val:
                best_val = vals[0]
                best = simplex[0].copy()

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            ref = centroid + (centroid - worst)
            rv = _eval(ref[0], ref[1], ref[2])

            if rv < vals[0]:
                exp = centroid + 2.0 * (ref - centroid)
                ev = _eval(exp[0], exp[1], exp[2])
                simplex[-1] = exp if ev < rv else ref
            elif rv < vals[-2]:
                simplex[-1] = ref
            else:
                con = centroid + 0.5 * (worst - centroid)
                cv = _eval(con[0], con[1], con[2])
                if cv < vals[-1]:
                    simplex[-1] = con
                else:
                    for i in range(1, 4):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])

    # Polish
    if time.time() < deadline - 3.0:
        for scale in [0.005, 0.001, 0.0003]:
            for _ in range(80):
                if time.time() > deadline - 3.0:
                    break
                p = best + scale * rng.standard_normal(3)
                v = _eval(p[0], p[1], p[2])
                if v < best_val:
                    best_val = v
                    best = p.copy()

    return best
# EVOLVE-BLOCK-END
