"""Optimized broadband AR coating design via characteristic-matrix reflectance + search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _reflectance_stack(thicknesses, materials, labels, n0, ns, wl):
    """Mean normal-incidence reflectance (percent) over wavelengths wl, exact CM method."""
    n = len(thicknesses)
    if n == 0:
        r = (n0 - ns) / (n0 + ns)
        return 100.0 * r * r
    # phase per layer per wavelength: beta = 2*pi*n*d/lambda
    beta = 2.0 * math.pi * np.outer(np.asarray(thicknesses, float),
                                    np.asarray(materials, float)) / np.asarray(wl, float)
    # transfer matrix per layer: [[cos, i sin/n], [i n sin, cos]]
    c = np.cos(beta)          # (n, m)
    s = np.sin(beta)          # (n, m)
    ns_arr = np.asarray(materials, float)[:, None]
    m11 = c
    m12 = 1j * s / ns_arr
    m21 = 1j * ns_arr * s
    m22 = c
    m01 = 1.0
    m02 = 0.0
    m10 = 0.0
    m11b = 1.0
    for k in range(n):
        a11 = m11[k]
        a12 = m12[k]
        a21 = m21[k]
        a22 = m22[k]
        nm11 = m01 * a11 + m02 * a21
        nm12 = m01 * a12 + m02 * a22
        nm21 = m10 * a11 + m11b * a21
        nm22 = m10 * a12 + m11b * a22
        m01, m02, m10, m11b = nm11, nm12, nm21, nm22
    # admittance at incident side: Y = (m10 + m11*ns) / (m20 + m21*ns)
    Y = (m10 + m11b * ns) / (m21 + m22 * ns)
    r = (n0 - Y) / (n0 + Y)
    R = (r * np.conj(r)).real
    return float(np.mean(R)) * 100.0


def _make_wl(problem):
    b0, b1 = problem["band_nm"]
    return np.linspace(b0, b1, 401)


def _valid(thicknesses, labels, problem):
    max_layer = problem.get("max_layer_nm", 400)
    max_total = problem.get("max_total_nm", 1000)
    if len(thicknesses) > problem.get("max_layers", 20):
        return False
    if any(t < -1e-9 or t > max_layer + 1e-9 for t in thicknesses):
        return False
    if sum(thicknesses) > max_total + 1e-6:
        return False
    return True


def _objective(x, problem, wl, labels, materials):
    thicknesses = [max(0.0, min(400.0, float(v))) for v in x]
    if sum(thicknesses) > problem.get("max_total_nm", 1000) + 1e-6:
        return 1e6
    return _reflectance_stack(thicknesses, materials, labels, problem["ambient_index"],
                              problem["substrate_index"], wl)


def design(problem):
    deadline = time.time() + 140.0  # safety margin under 150s wall clock
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    wl = _make_wl(problem)
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)

    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    # quarter-wave optical thickness targets
    qL = centre / (4.0 * materials["L"])
    qH = centre / (4.0 * materials["H"])

    def local_optim(labels, x0):
        """Nelder-Mead style local search on thicknesses (bounded via projection)."""
        labels = list(labels)
        n = len(labels)
        x = np.array([max(0.0, min(max_layer, v)) for v in x0], float)
        # Nelder-Mead
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        f = _objective(x, problem, wl, labels, materials)
        best_x, best_f = x.copy(), f
        simplex = [x.copy()]
        for i in range(n):
            xi = x.copy()
            step = 0.1 * max_layer if xi[i] == 0 else 0.1 * max(xi[i], 0.05 * max_layer)
            xi[i] += step
            simplex.append(xi)
        fvals = [_objective(v, problem, wl, labels, materials) for v in simplex]
        it = 0
        max_it = 6000
        while it < max_it and (time.time() < deadline - 8.0):
            it += 1
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if abs(fvals[-1] - fvals[0]) < 1e-9:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            # reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = _objective(xr, problem, wl, labels, materials)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[0]:
                # expansion
                xe = centroid + gamma * (xr - centroid)
                fe = _objective(xe, problem, wl, labels, materials)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            else:
                # contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = _objective(xc, problem, wl, labels, materials)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = _objective(simplex[i], problem, wl, labels, materials)
        # take best
        order = np.argsort(fvals)
        bx = simplex[order[0]]
        bf = fvals[order[0]]
        if bf < best_f:
            best_f, best_x = bf, bx
        return list(best_x), best_f

    best = (None, 1e9)

    def consider(labels, x, f):
        nonlocal best
        if f < best[1]:
            best = (list(labels), list(x), f) if False else (labels, x, f)
            best = (list(labels), list(x))
            best = (list(labels), list(x))
        # store best
        if f < best_f_global[0]:
            best_f_global[0] = f
            best_f_global[1] = (list(labels), list(x))

    best_f_global = [1e9, (None, None)]

    def upd(labels, x, f):
        if f < best_f_global[0]:
            best_f_global[0] = f
            best_f_global[1] = (list(labels), list(x))

    # candidate layer sequences (air -> substrate)
    sequences = []
    # quarter-wave stacks
    sequences.append(["L"])
    sequences.append(["H"])
    sequences.append(["L", "H"])
    sequences.append(["H", "L"])
    sequences.append(["L", "H", "L"])
    sequences.append(["H", "L", "H"])
    sequences.append(["L", "H", "L", "H"])
    sequences.append(["H", "L", "H", "L"])
    sequences.append(["L", "H", "L", "H", "L"])
    sequences.append(["H", "L", "H", "L", "H"])
    sequences.append(["L", "H", "L", "H", "L", "H"])
    sequences.append(["H", "L", "H", "L", "H", "L"])
    # symmetric-ish 8-10 layer broadband
    for pat in ["LHLHLH", "HLHLHL", "LHLHLHLH", "HLHLHLHL", "LHLHLHLHLH", "HLHLHLHLHL"]:
        if len(pat) <= max_layers:
            sequences.append(list(pat))

    # dedupe
    seen = set()
    uniq = []
    for s in sequences:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    sequences = uniq

    for labels in sequences:
        if time.time() > deadline - 15.0:
            break
        n = len(labels)
        # initial guess: quarter-wave for each layer
        x0 = []
        for lab in labels:
            if lab == "L":
                x0.append(qL)
            else:
                x0.append(qH)
        # scale to respect total budget if needed
        s = sum(x0)
        if s > max_total:
            x0 = [v * (max_total / s) for v in x0]
        f0 = _objective(x0, problem, wl, labels, materials)
        upd(labels, x0, f0)
        # local refine
        xl, fl = local_optim(labels, x0)
        upd(labels, xl, fl)

    # If we still have time, do a couple of random-restart local searches
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(12345)
        for trial in range(60):
            if time.time() > deadline - 15.0:
                break
            n = int(rng.integers(2, min(max_layers, 10) + 1))
            labels = [str(rng.choice(["L", "H"])) for _ in range(n)]
            x0 = [float(rng.uniform(0.0, max_layer)) for _ in range(n)]
            s = sum(x0)
            if s > max_total:
                x0 = [v * (max_total / s) for v in x0]
            f0 = _objective(x0, problem, wl, labels, materials)
            upd(labels, x0, f0)
            xl, fl = local_optim(labels, x0)
            upd(labels, xl, fl)

    labels, x = best_f_global[1]
    if labels is None:
        return ["L"], [qL]
    # clamp and validate
    x = [max(0.0, min(max_layer, float(v))) for v in x]
    if sum(x) > max_total:
        scale = max_total / sum(x)
        x = [v * scale for v in x]
    return labels, x
# EVOLVE-BLOCK-END
