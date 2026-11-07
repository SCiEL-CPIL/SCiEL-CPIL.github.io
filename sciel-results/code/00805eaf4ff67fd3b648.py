import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize

# ---- Constants from the task contract ----
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT
N_CASES = 9

_DEADLINE = 315.0
_START_TIME = time.time()

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # coherent: -i(H rho - rho H)  =>  L = -i( kron(I,H) - kron(H.T, I) )
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # incoherent (constant)
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # test states (6) and X-pi target states (6)
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
    # X-pi target: |0><->|1> (global phase irrelevant for Tr(rho_t rho))
    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)
    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# ---- Fast per-pulse simulation (all 9 cases) ----
def _score(I, Q):
    infidelities = np.empty(9)
    k = 0
    det_vals = (0.0, -2 * np.pi * 0.002, 2 * np.pi * 0.002)
    sc_vals = (0.97, 1.0, 1.03)
    for di_val in det_vals:
        for sc_val in sc_vals:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _det_coh
                L += sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

# ---- Pulse family (Gaussian + optional 2nd harmonic + DRAG) ----
def _make_pulse(params):
    sigma, amp, drag, h2 = params
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    x = (t - center) / sigma
    g = np.exp(-0.5 * x * x)
    g = g + h2 * np.exp(-0.5 * (2.0 * x) ** 2)
    area = np.sum(g) * DT
    if area <= 0:
        g = np.exp(-0.5 * x * x)
        area = np.sum(g) * DT
    I = g * (np.pi / area) * amp
    Q = np.zeros(N_STEPS)
    if drag != 0.0:
        dI = np.empty(N_STEPS)
        dI[1:-1] = (I[2:] - I[:-2]) / (2.0 * DT)
        dI[0] = (I[1] - I[0]) / DT
        dI[-1] = (I[-1] - I[-2]) / DT
        Q = drag * dI
    # enforce constraints
    ampv = np.sqrt(I * I + Q * Q)
    ma = np.max(ampv)
    if ma > AMPLITUDE_BOUND:
        s = AMPLITUDE_BOUND / ma
        I = I * s
        Q = Q * s
    for i in range(N_STEPS - 1):
        dI = I[i + 1] - I[i]
        dQ = Q[i + 1] - Q[i]
        sl = np.hypot(dI, dQ) / DT
        if sl > SLEW_BOUND:
            f = SLEW_BOUND * DT / np.hypot(dI, dQ)
            I[i + 1] = I[i] + dI * f
            Q[i + 1] = Q[i] + dQ * f
    s0 = np.hypot(I[0], Q[0]) / DT
    if s0 > SLEW_BOUND:
        f = SLEW_BOUND * DT / np.hypot(I[0], Q[0])
        I[0] *= f
        Q[0] *= f
    sN = np.hypot(I[-1], Q[-1]) / DT
    if sN > SLEW_BOUND:
        f = SLEW_BOUND * DT / np.hypot(I[-1], Q[-1])
        I[-1] *= f
        Q[-1] *= f
    return I, Q

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > AMPLITUDE_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    return True

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    sigma0 = TOTAL_TIME / 6.0
    seed = np.array([sigma0, 1.0, 0.0, 0.0])
    sI, sQ = _make_pulse(seed)
    best_score = _score(sI, sQ)
    best_I, best_Q = sI.copy(), sQ.copy()
    best_p = seed.copy()

    # ---- Differential evolution ----
    lb = np.array([sigma0 * 0.6, 0.85, -0.6, -0.4])
    ub = np.array([sigma0 * 1.4, 1.05, 0.6, 0.4])
    pop = 12
    rng = np.random.RandomState(0)
    pop_p = lb + (ub - lb) * rng.rand(pop, 4)
    pop_p[0] = best_p
    pop_s = np.empty(pop)
    for i in range(pop):
        pop_s[i] = _score(*_make_pulse(pop_p[i]))
    order = np.argsort(pop_s)
    best_i = order[0]
    if pop_s[best_i] < best_score:
        best_score, best_p = pop_s[best_i], pop_p[best_i].copy()
        best_I, best_Q = _make_pulse(best_p)

    F, CR = 0.8, 0.9
    gen = 0
    while elapsed() < _DEADLINE - 25:
        gen += 1
        improved = False
        for i in range(pop):
            if elapsed() > _DEADLINE - 20:
                break
            r1, r2 = rng.randint(0, pop, 2)
            if r1 == r2 or r1 == i or r2 == i:
                continue
            mut = pop_p[i] + F * (pop_p[r1] - pop_p[r2])
            cand = np.empty(4)
            for d in range(4):
                cand[d] = mut[d] if rng.random() < CR else pop_p[i][d]
            cand = np.clip(cand, lb, ub)
            cs = _score(*_make_pulse(cand))
            if cs < pop_s[i]:
                pop_p[i] = cand
                pop_s[i] = cs
                improved = True
                if cs < best_score:
                    best_score, best_p = cs, cand.copy()
                    best_I, best_Q = _make_pulse(best_p)
        if not improved:
            # shrink population to best few and re-seed
            keep = np.argsort(pop_s)[:max(4, pop // 2)]
            pop_p = pop_p[keep].copy()
            pop_s = pop_s[keep].copy()
            pop = pop_p.shape[0]
            extra = lb + (ub - lb) * rng.random((pop, 4))
            pop_p = np.vstack([pop_p, extra])
            pop_s = np.concatenate([pop_s,
                                    np.array([_score(*_make_pulse(p)) for p in extra])])
            pop = pop_p.shape[0]
        if gen > 60:
            break

    # ---- Nelder-Mead local refinement ----
    if elapsed() < _DEADLINE - 10:
        def obj(p):
            try:
                return _score(*_make_pulse(np.clip(p, lb, ub)))
            except Exception:
                return 1.0
        try:
            res = scipy.optimize.minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-9,
                                    'adaptive': True})
            p = np.clip(res.x, lb, ub)
            s = obj(p)
            if s < best_score:
                best_score, best_p = s, p.copy()
                best_I, best_Q = _make_pulse(best_p)
        except Exception:
            pass

    # ---- final safety: keep valid best, else seed ----
    if not _valid(best_I, best_Q):
        best_I, best_Q = sI, sQ
    return np.column_stack([best_I, best_Q])
