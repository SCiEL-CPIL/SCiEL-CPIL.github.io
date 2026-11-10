# EVOLVE-BLOCK-START
import numpy as np

# ---------------------------------------------------------------------------
# Continuous-state HJB approximation with semi-Lagrangian scheme.
#
# Model (per unit time, continuous-inventory approximation of the discrete
# task):
#   V_t + 0.5*sigma^2*V_xx = max_{b,a} [ lam_b(b)*(b - V_x)
#                                        + lam_a(a)*(-a - V_x)
#                                        - 0.5*gamma*sigma^2*q^2 ]
#   lam_b = A*exp(-k b),  lam_a = A*exp(-k a),   V_T = 0
#
# First-order condition (concave in each depth) gives the optimal depth in
# closed form, eliminating the depth grid search:
#   interior (|V_x| < A):  b* = a* = (1/k)*ln(A/lambda*),  lambda* = A - |V_x|
#   boundary (|V_x| >= A): no quote (depth = inf)
#   value term = A*ln(A/lambda*) - |V_x|*ln(A/lambda*)
# ---------------------------------------------------------------------------

def _solve_regime(sigma, A, k, gamma, T, steps, Q, nq, nstep, dt):
    dq = 2.0 * Q / (nq - 1)
    q = np.linspace(-Q, Q, nq)
    V = np.zeros(nq)
    # store dV/dx at each time index (0..nstep), index nstep == terminal == 0
    Vx_store = np.zeros((nstep + 1, nq))
    Vx_store[nstep] = 0.0

    for j in range(nstep - 1, -1, -1):
        Vx = np.gradient(V, dq)
        Vx_store[j + 1] = Vx
        Vt = -0.5 * gamma * sigma * sigma * q * q
        # semi-Lagrangian back-shift for the diffusion term
        shift = 0.5 * sigma * sigma * dt / (dq * dq)
        s = float(shift)
        x0 = np.arange(nq) - s
        i0 = np.floor(x0)
        f = x0 - i0
        i0l = np.clip(i0.astype(np.int64), 0, nq - 1)
        i0r = np.clip(i0l + 1, 0, nq - 1)
        Vxx = (1.0 - f) * V[i0l] + f * V[i0r]
        Vxx = np.where(np.abs(x0) > nq, 0.0, Vxx)
        # optimal control value (closed form)
        avx = np.abs(Vx)
        lam = A - avx
        interior = lam > 1e-12
        val = np.zeros(nq)
        val[interior] = (A - avx[interior]) * np.log(A / lam[interior])
        # Euler step
        V = V + dt * (Vt + 0.5 * sigma * sigma * Vxx + val)
    return Vx_store, q, dq


def _build_tables():
    T = 1.0
    steps = 1000
    Q = 30
    nq = 2401          # fine inventory grid (dq = 0.025)
    nstep = 250        # time steps (dt = 0.004, ~4x finer than the task step)
    dt = T / nstep
    regimes = [(2.0, 140.0, 1.5, 0.05),
               (2.0, 140.0, 1.5, 0.5),
               (1.5, 50.0, 0.75, 0.2)]
    tables = []
    for (sigma, A, k, gamma) in regimes:
        Vx_store, q, dq = _solve_regime(sigma, A, k, gamma, T, steps, Q, nq, nstep, dt)
        tables.append((Vx_store, q, dq, sigma, A, k, gamma))
    return tables


_TABLES = _build_tables()


def _regime_key(params):
    return (round(float(params["sigma"]), 6),
            round(float(params["A"]), 6),
            round(float(params["k"]), 6),
            round(float(params["gamma"]), 6))


def quote(t, inventory, mid, params):
    key = _regime_key(params)
    tab = None
    for tb in _TABLES:
        if (round(float(tb[3]), 6), round(float(tb[4]), 6),
                round(float(tb[5]), 6), round(float(tb[6]), 6)) == key:
            tab = tb
            break
    if tab is None:
        # fallback: risk-neutral depth
        depth = np.full(len(inventory), 1.0 / float(params["k"]))
        return depth, depth.copy()

    Vx_store, q, dq, sigma, A, k, gamma = tab
    T = float(params.get("T", 1.0))
    steps = int(params.get("steps", 1000))
    Q = float(params.get("Q", 30.0))
    dt_task = T / steps

    n = int(round(t / dt_task))
    if n < 0:
        n = 0
    elif n > steps - 1:
        n = steps - 1

    # map task time to table time index (table has nstep+1 rows, row nstep = T)
    nstep = Vx_store.shape[0] - 1
    frac = (n + 1) * nstep / steps          # in [0, nstep]
    j0 = int(np.floor(frac))
    if j0 > nstep - 1:
        j0 = nstep - 1
    w = frac - j0

    inv = np.asarray(inventory, dtype=np.float64)
    idx = (inv + Q) / (2.0 * Q) * (len(q) - 1)
    i0 = np.floor(idx)
    f = idx - i0
    i0l = np.clip(i0.astype(np.int64), 0, len(q) - 1)
    i0r = np.clip(i0l + 1, 0, len(q) - 1)
    Vx = (1.0 - w) * ((1.0 - f) * Vx_store[j0, i0l] + f * Vx_store[j0, i0r]) \
        + w * ((1.0 - f) * Vx_store[j0 + 1, i0l] + f * Vx_store[j0 + 1, i0r])

    avx = np.abs(Vx)
    lam = A - avx
    interior = lam > 1e-9
    depth = np.full(len(inv), np.inf)
    depth[interior] = (1.0 / k) * np.log(A / lam[interior])
    # keep depths finite & sane (task validity: not below -3/k, not NaN/inf-as-number)
    depth = np.where(np.isfinite(depth), depth, np.inf)
    return depth, depth.copy()
# EVOLVE-BLOCK-END
