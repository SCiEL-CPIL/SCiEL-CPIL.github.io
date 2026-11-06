"""Antireflection coating design: constructive symmetric stack + fast local refinement."""
import time
import math
import numpy as np

_START = None


def _now():
    return time.time()


def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index,
                          band_nm, n_wl=401):
    """Mean normal-incidence reflectance (%) via characteristic-matrix method."""
    n0 = float(ambient_index)
    ns = float(substrate_index)
    nl = float(materials["L"])
    nh = float(materials["H"])
    wl = np.linspace(band_nm[0], band_nm[1], n_wl)
    # 2x2 transfer matrix for a layer of thickness d (nm) at each wavelength.
    # phase = 2*pi*n*d/lam  (lam in nm)
    # M = [[cos, i sin/n], [i n sin, cos]]
    d = np.asarray(thicknesses, dtype=float)
    M = np.array([[1.0 + 0.0j, 0.0 + 0.0j]], dtype=complex)  # identity, shape (2,2)
    # We process layer by layer; each layer contributes a (2,2,401) matrix.
    # To keep it fast, accumulate a (2,2,401) running matrix.
    acc = np.identity(2, dtype=complex)[..., np.newaxis]  # (2,2,1) -> broadcast
    acc = np.broadcast_to(acc, (2, 2, len(wl))).copy()
    for i in range(len(d)):
        n = nl if i % 2 == 0 else nh  # alternating L,H,L,H,...
        phase = (2.0 * math.pi * n * d[i]) / wl  # (401,)
        c = np.cos(phase)
        s = np.sin(phase)
        # layer matrix (2,2,401)
        Lm = np.empty((2, 2, len(wl)), dtype=complex)
        Lm[0, 0] = c
        Lm[0, 1] = 1j * s / n
        Lm[1, 0] = 1j * n * s
        Lm[1, 1] = c
        acc = Lm @ acc
    # Final: Y = (m11 + m12*n0) / (m21 + m22*n0)  with n0 ambient, and reflectance
    # r = (Y - n0) / (Y + n0)  ... but careful: Y is admittance. Actually with
    # characteristic matrix M = [m11 m12; m21 m22], the input admittance is
    # Y = (m11 + m12 * n_sub) / (m21 + m22 * n_sub). Then r = (Y - n0)/(Y + n0).
    m11 = acc[0, 0]
    m12 = acc[0, 1]
    m21 = acc[1, 0]
    m22 = acc[1, 1]
    Y = (m11 + m12 * ns) / (m21 + m22 * ns)
    r = (Y - n0) / (Y + n0)
    R = (r * np.conj(r)).real
    R = np.clip(R, 0.0, 1.0)
    return 100.0 * np.mean(R)


def _mean_reflectance(x, materials, ambient_index, substrate_index, band_nm):
    return _reflectance_spectrum(x, materials, ambient_index, substrate_index, band_nm)


def _n_array(n, materials):
    return np.array([materials["L"] if i % 2 == 0 else materials["H"] for i in range(n)],
                    dtype=float)


def design(problem):
    global _START
    if _START is None:
        _START = _now()
    DEADLINE = _START + 145.0  # safety margin under 150s wall-clock

    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = int(problem.get("max_layers", 20))
    max_total_nm = float(problem.get("max_total_nm", 1000.0))
    max_layer_nm = float(problem.get("max_layer_nm", 400.0))

    lam_c = 0.5 * (band_nm[0] + band_nm[1])  # 600 nm
    nl = materials["L"]
    nh = materials["H"]

    # Initial symmetric quarter-wave-ish design with an even number of layers.
    # Start small, grow.
    def make_initial(n_layers):
        # alternating L,H,... starting with L on the air side.
        # symmetric physical thickness: pair (i, n-1-i) equal.
        base = lam_c / (4.0)  # quarter wave in vacuum ~150 nm
        d = []
        for i in range(n_layers):
            n = nl if i % 2 == 0 else nh
            d.append(base / n)
        # symmetrize
        half = n_layers // 2
        for i in range(half):
            avg = 0.5 * (d[i] + d[n_layers - 1 - i])
            d[i] = avg
            d[n_layers - 1 - i] = avg
        if n_layers % 2 == 1:
            pass
        return np.array(d, dtype=float)

    def is_valid(d):
        if len(d) == 0 or len(d) > max_layers:
            return False
        if np.any(d < -1e-6) or np.any(d > max_layer_nm + 1e-6):
            return False
        if np.sum(d) > max_total_nm + 1e-6:
            return False
        return True

    # Local optimizer: Nelder-Mead with clipping to bounds, time-guarded.
    def nelder_mead(x0, n_layers, iters):
        from scipy.optimize import minimize
        n = n_layers
        d0 = np.clip(x0, 0.0, max_layer_nm)

        def obj(v):
            v = np.clip(v, 0.0, max_layer_nm)
            return _mean_reflectance(v, materials, ambient_index, substrate_index, band_nm)

        best_x = d0.copy()
        best_f = obj(d0)
        t0 = _now()
        try:
            res = minimize(
                obj, d0, method="Nelder-Mead",
                options={"maxiter": iters, "xatol": 1e-3, "fatol": 1e-5,
                         "adaptive": True},
            )
            xf = np.clip(res.x, 0.0, max_layer_nm)
            ff = obj(xf)
            if ff < best_f:
                best_x, best_f = xf, ff
        except Exception:
            pass
        return best_x, best_f, (t0)

    # Simulated-annealing restart for escaping local minima.
    def sa_restart(x0, n_layers, iters):
        d = np.clip(x0.copy(), 0.0, max_layer_nm)
        f = _mean_reflectance(d, materials, ambient_index, substrate_index, band_nm)
        best_d = d.copy()
        best_f = f
        t0 = _now()
        T = 2.0
        cool = 0.9995
        for it in range(iters):
            if _now() > DEADLINE:
                break
            i = int(np.random.randint(n_layers))
            step = np.random.uniform(20.0, 60.0)
            nd = d.copy()
            nd[i] = np.clip(nd[i] + np.random.uniform(-step, step), 0.0, max_layer_nm)
            nf = _mean_reflectance(nd, materials, ambient_index, substrate_index, band_nm)
            df = nf - f
            if df < 0 or np.random.rand() < math.exp(-df / max(T, 1e-9)):
                d, f = nd, nf
                if f < best_f:
                    best_d, best_f = d.copy(), f
            T *= cool
        return best_d, best_f

    # Grow-layer strategy: start with 4 layers, add pairs, refine.
    best_x = make_initial(4)
    best_f = _mean_reflectance(best_x, materials, ambient_index, substrate_index, band_nm)

    # Coarse pass with a few layers
    for n_layers in [4, 6, 8, 10]:
        if n_layers > max_layers:
            break
        if _now() > DEADLINE - 20.0:
            break
        x0 = make_initial(n_layers)
        # try a couple of starting scalings
        for scale in [0.9, 1.0, 1.1]:
            if _now() > DEADLINE - 15.0:
                break
            x0s = np.clip(x0 * scale, 0.0, max_layer_nm)
            if np.sum(x0s) > max_total_nm:
                continue
            xm, fm, _ = nelder_mead(x0s, n_layers, iters=400)
            if fm < best_f:
                best_f, best_x = fm, xm
        # SA restart from the best of this size
        if _now() > DEADLINE - 15.0:
            break
        xs, fs = sa_restart(best_x, n_layers, iters=2000)
        if fs < best_f:
            best_f, best_x = fs, xs

    # Final refinement on the best size found
    if _now() < DEADLINE - 8.0:
        n_best = len(best_x)
        xm, fm, _ = nelder_mead(best_x, n_best, iters=2000)
        if fm < best_f:
            best_f, best_x = fm, xm
        if _now() < DEADLINE - 5.0:
            xs, fs = sa_restart(best_x, n_best, iters=3000)
            if fs < best_f:
                best_f, best_x = fs, xs

    # Clamp to valid range
    best_x = np.clip(best_x, 0.0, max_layer_nm)
    # If total exceeds, scale down proportionally
    if np.sum(best_x) > max_total_nm:
        best_x = best_x * (max_total_nm / np.sum(best_x))
        best_x = np.clip(best_x, 0.0, max_layer_nm)

    labels = ["L" if i % 2 == 0 else "H" for i in range(len(best_x))]
    thicknesses = [float(v) for v in best_x]
    return labels, thicknesses
