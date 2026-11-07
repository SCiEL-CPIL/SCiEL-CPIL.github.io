import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# ---- Constants from the task contract ----
ANH = -2 * np.pi * 0.25          # anharmonicity (rad/ns)
AMP_B = 0.4                      # amplitude bound
SLEW_B = 0.12                    # slew bound
DT = 0.5                         # step (ns)
N = 80                           # steps
T = 40.0                         # duration (ns)
T1 = 25000.0
Tphi = 40000.0

I3 = np.eye(3)
_sq2 = np.sqrt(2.0)
a = np.array([[0, 1, 0],
              [0, 0, _sq2],
              [0, 0, 0]], complex)
adag = a.conj().T
n = adag @ a
nn1 = n @ n - n                  # n(n-1)
L1 = a / np.sqrt(T1)
L2 = n * np.sqrt(2.0 / Tphi)
_LOPS = [L1, L2]

# ---- Target states for the X gate: X|0>=|1>, X|1>=|0> ----
_e0 = np.array([1, 0, 0], complex)
_e1 = np.array([0, 1, 0], complex)
_TS = [
    _e1, _e0,
    (_e1 + _e0) / np.sqrt(2.0), (_e1 - _e0) / np.sqrt(2.0),
    (_e1 + 1j * _e0) / np.sqrt(2.0), (_e1 - 1j * _e0) / np.sqrt(2.0),
]

# ---- 9 (detuning, scale) cases ----
_DETS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
_SCALES = [0.97, 1.0, 1.03]
_CASES = [(d, s) for d in _DETS for s in _SCALES]


def _superop(I, Q, det, scale):
    H = det * n + 0.5 * ANH * nn1 + 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    # vec convention vec(AXB) = (B^T (kron) A) vec(X)
    S = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    for L in _LOPS:
        Ldag = L.conj().T
        LL = Ldag @ L
        S = S + np.kron(L.conj(), L) - 0.5 * np.kron(I3, LL) - 0.5 * np.kron(LL, I3)
    return S


def _simulate(I, Q, det, scale):
    v = np.zeros(9, complex)
    v[0] = 1.0
    for k in range(N):
        S = _superop(I[k], Q[k], det, scale)
        v = expm(S * DT) @ v
    return v


def _score_pulse(I, Q):
    infs = np.empty(len(_CASES))
    for idx, (det, scale) in enumerate(_CASES):
        v = _simulate(I, Q, det, scale)
        rho = v.reshape(3, 3)
        fsum = 0.0
        for t in _TS:
            fsum += np.vdot(t, rho @ t).real
        fid = fsum / len(_TS)
        infs[idx] = 1.0 - fid
    return float(np.mean(infs) + np.max(infs))


def _make_pulse(params):
    A, D, sig, tshift = params
    t = (np.arange(N) + 0.5) * DT
    tc = T / 2.0
    sig = max(sig, 1.0)
    g = np.exp(-0.5 * ((t - tc) / sig) ** 2)
    # DRAG term: derivative of a (possibly shifted) Gaussian envelope
    td = t - tc - tshift
    gd = np.exp(-0.5 * (td / sig) ** 2)
    dgd = -(td / sig ** 2) * gd
    I = A * g
    Q = D * dgd
    return I, Q


def _penalty(I, Q):
    amp = np.sqrt(I ** 2 + Q ** 2)
    p = 1000.0 * max(0.0, np.max(amp) - AMP_B) ** 2
    dI = np.diff(I); dQ = np.diff(Q)
    slew = np.sqrt(dI ** 2 + dQ ** 2) / DT
    p += 1000.0 * max(0.0, np.max(slew) - SLEW_B) ** 2
    p += 1000.0 * max(0.0, np.sqrt(I[0] ** 2 + Q[0] ** 2) / DT - SLEW_B) ** 2
    p += 1000.0 * max(0.0, np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / DT - SLEW_B) ** 2
    return p


def _valid(I, Q):
    if np.max(np.sqrt(I ** 2 + Q ** 2)) > AMP_B + 1e-9:
        return False
    dI = np.diff(I); dQ = np.diff(Q)
    if np.max(np.sqrt(dI ** 2 + dQ ** 2)) / DT > SLEW_B + 1e-9:
        return False
    if np.sqrt(I[0] ** 2 + Q[0] ** 2) / DT > SLEW_B + 1e-9:
        return False
    if np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / DT > SLEW_B + 1e-9:
        return False
    return True


def _objective(params, deadline):
    if time.time() > deadline:
        raise TimeoutError
    I, Q = _make_pulse(params)
    return _score_pulse(I, Q) + _penalty(I, Q)


def _fallback_pulse():
    # seed-like DRAG, guaranteed valid
    I, Q = _make_pulse([0.25, -0.03, 6.67, 0.0])
    return np.column_stack([I, Q])


def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 270.0  # safety margin before 300s kill

    best_params = None
    best_obj = np.inf

    starts = [
        [0.30, -0.04, 6.0, 0.0],
        [0.35, -0.06, 5.0, 0.5],
        [0.25, -0.02, 7.0, -0.5],
        [0.40, -0.08, 4.5, 0.0],
        [0.30,  0.04, 6.0, 0.0],
    ]

    class _Abort(Exception):
        pass

    def _cb(*_):
        if time.time() > deadline:
            raise _Abort

    for sp in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(_objective, sp, args=(deadline,), method='Nelder-Mead',
                           options={'maxiter': 400, 'maxfev': 400,
                                    'xatol': 1e-4, 'fatol': 1e-7,
                                    'adaptive': True, 'callback': _cb})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except _Abort:
            break
        except Exception:
            continue

    if best_params is not None:
        I, Q = _make_pulse(best_params)
        if _valid(I, Q):
            return np.column_stack([I, Q])
    return _fallback_pulse()
