"""Antireflection coating design: simulated annealing over sequence + thicknesses."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize


def _build_reflectance(problem):
    """Return a fast callable: (labels, thicknesses) -> mean reflectance in percent."""
    n0 = float(problem["ambient_index"])
    ns = float(problem["substrate_index"])
    nL = float(problem["materials"]["L"])
    nH = float(problem["materials"]["H"])
    band = problem["band_nm"]
    wl = np.linspace(band[0], band[1], 401)
    N = wl.size
    two_pi = 2.0 * math.pi

    def reflectance(labels, thicknesses):
        m = len(labels)
        # M total = I, shape (N,2,2)
        M = np.zeros((N, 2, 2), dtype=complex)
        M[:, 0, 0] = 1.0
        M[:, 1, 1] = 1.0
        for j in range(m):
            nj = nL if labels[j] == "L" else nH
            tj = thicknesses[j]
            if tj <= 0.0:
                # zero-thickness layer acts as identity; skip
                continue
            delta = two_pi * nj * tj / wl
            c = np.cos(delta)
            s = np.sin(delta)
            Mj = np.zeros((N, 2, 2), dtype=complex)
            Mj[:, 0, 0] = c
            Mj[:, 0, 1] = 1j * s / nj
            Mj[:, 1, 0] = 1j * nj * s
            Mj[:, 1, 1] = c
            M = M @ Mj
        B = M[:, 0, 0] + M[:, 0, 1] * ns
        C = M[:, 1, 0] + M[:, 1, 1] * ns
        r = (n0 * B - C) / (n0 * B + C)
        R = (r.real ** 2 + r.imag ** 2)
        return float(R.mean()) * 100.0

    return reflectance


def _valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
    if len(labels) != len(thicknesses):
        return False
    if len(labels) == 0 or len(labels) > max_layers:
        return False
    tot = 0.0
    for t in thicknesses:
        if t < 0.0 or t > max_layer_nm + 1e-9:
            return False
        tot += t
    if tot > max_total_nm + 1e-9:
        return False
    return True


def _quarter(n, lam0):
    return lam0 / (4.0 * n)


def design(problem):
    t_start = time.time()
    DEADLINE = 130.0  # safety margin before the 150 s wall-clock kill

    max_layers = int(problem["max_layers"])
    max_total_nm = float(problem["max_total_nm"])
    max_layer_nm = float(problem["max_layer_nm"])
    band = problem["band_nm"]
    lam0 = 0.5 * (band[0] + band[1])
    nL = float(problem["materials"]["L"])
    nH = float(problem["materials"]["H"])

    reflectance = _build_reflectance(problem)

    rng = np.random.default_rng(12345)

    def clamp_t(t):
        if t < 0.0:
            return 0.0
        if t > max_layer_nm:
            return max_layer_nm
        return float(t)

    def total_t(th):
        return float(sum(th))

    # ---- initial patterns (diverse) ----
    def make_initial(seed_idx):
        patterns = [
            ["L", "H", "L", "H"],
            ["H", "L", "H", "L"],
            ["L", "H", "L", "H", "L", "H"],
            ["H", "L", "H", "L", "H", "L"],
            ["L", "H", "L", "H", "L", "H", "L", "H"],
            ["L", "H", "L"],
            ["H", "L", "H", "L", "H"],
        ]
        labels = list(patterns[seed_idx % len(patterns)])
        th = []
        for lab in labels:
            n = nL if lab == "L" else nH
            t = _quarter(n, lam0)
            t *= float(rng.uniform(0.85, 1.15))
            th.append(clamp_t(t))
        # enforce total budget by scaling if needed
        tot = total_t(th)
        if tot > max_total_nm:
            scale = max_total_nm / tot
            th = [clamp_t(t * scale) for t in th]
        return labels, th

    # ---- simulated annealing on one stack ----
    def sa_run(labels, thicknesses, iters):
        labels = list(labels)
        th = [float(t) for t in thicknesses]
        cur = reflectance(labels, th)
        best_labels = list(labels)
        best_th = list(th)
        best_val = cur
        T0 = 2.0
        Tend = 0.02
        cool = (Tend / T0) ** (1.0 / max(1, iters))
        T = T0
        for it in range(iters):
            # pick a move
            move = rng.random()
            trial_labels = list(labels)
            trial_th = list(th)
            if move < 0.55:
                # perturb a thickness
                k = int(rng.integers(0, len(labels)))
                trial_th[k] = clamp_t(th[k] + float(rng.normal(0.0, 8.0)))
            elif move < 0.70:
                # flip a label
                k = int(rng.integers(0, len(labels)))
                trial_labels[k] = "H" if labels[k] == "L" else "L"
            elif move < 0.82 and len(labels) < max_layers:
                # add a layer
                pos = int(rng.integers(0, len(labels) + 1))
                lab = "L" if rng.random() < 0.5 else "H"
                n = nL if lab == "L" else nH
                t = clamp_t(float(rng.uniform(0.3, 1.0)) * _quarter(n, lam0))
                if total_t(th) + t <= max_total_nm + 1e-9:
                    trial_labels.insert(pos, lab)
                    trial_th.insert(pos, t)
                else:
                    t = 0.0
                    trial_labels.insert(pos, lab)
                    trial_th.insert(pos, t)
            elif move < 0.92 and len(labels) > 1:
                # remove a layer
                k = int(rng.integers(0, len(labels)))
                trial_labels.pop(k)
                trial_th.pop(k)
            else:
                # swap adjacent
                if len(labels) > 1:
                    k = int(rng.integers(0, len(labels) - 1))
                    trial_labels[k], trial_labels[k + 1] = trial_labels[k + 1], trial_labels[k]
                    trial_th[k], trial_th[k + 1] = trial_th[k + 1], trial_th[k]

            if not _valid(trial_labels, trial_th, max_layers, max_total_nm, max_layer_nm):
                T *= cool
                continue
            trial_val = reflectance(trial_labels, trial_th)
            d = trial_val - cur
            if d < 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                labels = trial_labels
                th = trial_th
                cur = trial_val
                if cur < best_val:
                    best_val = cur
                    best_labels = list(labels)
                    best_th = list(th)
            T *= cool
        return best_labels, best_th, best_val

    # ---- local polish on thicknesses for a fixed sequence ----
    def polish(labels, thicknesses):
        x0 = np.array([float(t) for t in thicknesses])
        if len(x0) == 0:
            return list(labels), list(thicknesses), reflectance(labels, thicknesses)

        def obj(x):
            x = np.clip(x, 0.0, max_layer_nm)
            if float(x.sum()) > max_total_nm + 1e-9:
                # penalize over-budget
                return reflectance(labels, list(x)) + 5.0 * (float(x.sum()) - max_total_nm)
            return reflectance(labels, list(x))

        try:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 400, "xatol": 0.5, "fatol": 1e-4})
            x = np.clip(res.x, 0.0, max_layer_nm)
            if float(x.sum()) > max_total_nm:
                scale = max_total_nm / float(x.sum())
                x = x * scale
            val = reflectance(labels, list(x))
            return list(labels), [float(t) for t in x], val
        except Exception:
            return list(labels), list(thicknesses), reflectance(labels, thicknesses)

    # ---- main loop: restarts, each with SA then polish ----
    global_labels, global_th, global_val = None, None, float("inf")

    def consider(labels, thicknesses, val):
        nonlocal global_labels, global_th, global_val
        if _valid(labels, thicknesses, max_layers, max_total_nm, max_layer_nm):
            if val < global_val:
                global_labels = list(labels)
                global_th = [float(t) for t in thicknesses]
                global_val = val

    restart = 0
    while time.time() - t_start < DEADLINE:
        init_labels, init_th = make_initial(restart)
        # SA budget: scale iterations to ~ fill remaining time with a cap
        remaining = DEADLINE - (time.time() - t_start)
        # estimate: each SA iter ~ a few ms; cap iters per restart
        iters = int(min(4000, max(500, remaining * 250)))
        if iters < 200:
            break
        bl, bt, bv = sa_run(init_labels, init_th, iters)
        consider(bl, bt, bv)
        # polish the best from this restart
        if time.time() - t_start < DEADLINE - 5.0:
            pl, pt, pv = polish(bl, bt)
            consider(pl, pt, pv)
        restart += 1

    # final polish on the global best with a bit more effort if time allows
    if global_labels is not None and time.time() - t_start < DEADLINE - 3.0:
        pl, pt, pv = polish(global_labels, global_th)
        consider(pl, pt, pv)

    if global_labels is None:
        # fallback: single quarter-wave L layer (always valid)
        t = _quarter(nL, lam0)
        return ["L"], [min(t, max_layer_nm)]

    return global_labels, global_th
# EVOLVE-BLOCK-END
