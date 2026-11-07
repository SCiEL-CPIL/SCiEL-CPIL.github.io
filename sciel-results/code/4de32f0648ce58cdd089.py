"""Robust X-pi pulse for a 3-level transmon. DRAG base + Legendre refinement,
time-budgeted optimizer with explicit deadline guard."""
import time
import numpy as np
from scipy.linalg import expm

# ---- fixed constants from the task contract ----
N = 80
DT = 0.5
T1 = 25000.0
T2P = 40000.0
ANH = -2.0 * np.pi * 0.25
AMP = 0.4
SLEW = 0.12
DET_B = 2.0 * np.pi * 0.002

# ---- precompute operator matrices (3x3) ----
def _a_mat():
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    return a

A = _a_mat()
ADAG = A.conj().T
N_OP = ADAG @ A
# detuning operator n + 0.5*anh*n*(n-1)
N1 = N_OP
N2 = N_OP @ (N_OP - np.eye(3))
DET_OP = N1 + 0.5 * ANH * N2
XOP = A + ADAG
YOP = 1j * (A - ADAG)
L1 = A / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / T2P)

# basis states
E0 = np.array([1, 0, 0], dtype=complex)
E1 = np.array([0, 1, 0], dtype=complex)
PLUS = (E0 + E1) / np.sqrt(2.0)
MINUS = (E0 - E1) / np.sqrt(2.0)
RPLUS = (E0 + 1j * E1) / np.sqrt(2.0)
RMINUS = (E0 - 1j * E1) / np.sqrt(2.0)
STATES = [E0, E1, PLUS, MINUS, RPLUS, RMINUS]

# Liouvillian in vectorized form: dvec = -i*(H^L - H^R)vec + sum gamma_k L_k^L vec L_k^R
def _build_liouv(detuning, scale):
    H = detuning * N_OP + 0.5 * scale * 0  # placeholder
    # H = detuning*n + 0.5*anh*n(n-1) + 0.5*scale*(I*XOP + Q*YOP)
    # but I,Q vary per step; build static part now, add drive per step
    H0 = detuning * N_OP + 0.5 * ANH * N2
    HL = np.kron(H0, np.eye(3))
    HR = np.kron(np.eye(3), H0.conj().T)
    out = -1j * (HL - HR)
    for Lk in (L1, L2):
        LHL = np.kron(Lk, np.eye(3))
        LHR = np.kron(np.eye(3), Lk.conj().T)
        out = out + LHL @ LHR - 0.5 * (np.kron(Lk @ Lk.conj().T, np.eye(3))
                                       + np.kron(np.eye(3), Lk @ Lk.conj().T))
    return out

def _drive_liouv(I, Q, scale):
    H = 0.5 * scale * (I * XOP + Q * YOP)
    HL = np.kron(H, np.eye(3))
    HR = np.kron(np.eye(3), H.conj().T)
    return -1j * (HL - HR)

# Precompute static liouvillian per (detuning) — scale only affects drive,
# but detuning part is independent of scale. Good: cache by detuning.
_STATIC_CACHE = {}
def _static_liouv(detuning):
    key = round(detuning, 12)
    if key not in _STATIC_CACHE:
        H0 = detuning * N_OP + 0.5 * ANH * N2
        HL = np.kron(H0, np.eye(3))
        HR = np.kron(np.eye(3), H0.conj().T)
        out = -1j * (HL - HR)
        for Lk in (L1, L2):
            LHL = np.kron(Lk, np.eye(3))
            LHR = np.kron(np.eye(3), Lk.conj().T)
            out = out + LHL @ LHR - 0.5 * (np.kron(Lk @ Lk.conj().T, np.eye(3))
                                           + np.kron(np.eye(3), Lk @ Lk.conj().T))
        _STATIC_CACHE[key] = out
    return _STATIC_CACHE[key]

def _simulate(IQ, detuning, scale, states):
    """IQ: (N,2) array of I,Q. Returns mean fidelity over states."""
    L0 = _static_liouv(detuning)
    fids = []
    for s0 in states:
        rho = np.outer(s0, s0.conj())
        vec = rho.reshape(9)
        for k in range(N):
            I, Q = IQ[k]
            L = L0 + _drive_liouv(I, Q, scale)
            # exponentiate over DT
            vec = expm(L * DT) @ vec
        rho = vec.reshape(3, 3)
        tr = np.trace(rho)
        rho = rho / tr
        fid = np.real(np.trace(rho @ np.outer(s0, s0.conj())))
        fid = max(0.0, min(1.0, fid))
        fids.append(fid)
    return float(np.mean(fids))

def _fidelity_case(IQ, detuning, scale):
    # infidelity for the case
    return 1.0 - _simulate(IQ, detuning, scale, STATES)

def _score(IQ):
    dets = (-DET_B, 0.0, DET_B)
    scales = (0.97, 1.0, 1.03)
    vals = []
    for d in dets:
        for sc in scales:
            vals.append(_fidelity_case(IQ, d, sc))
    arr = np.array(vals)
    return float(arr.mean() + arr.max())

def _valid(IQ):
    # amplitude bound
    if np.any(np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2) > AMP + 1e-9):
        return False
    # slew bound including 0->step1 and last->0
    if np.any(np.abs(IQ[0]) > SLEW * DT + 1e-9):
        return False
    if np.any(np.abs(IQ[-1]) > SLEW * DT + 1e-9):
        return False
    d = np.diff(IQ, axis=0)
    if np.any(np.sqrt(np.sum(d ** 2, axis=1)) / DT > SLEW + 1e-9):
        return False
    return True

def _clamp(IQ):
    """Project onto amplitude and slew constraints (approx)."""
    IQ = IQ.copy()
    # amplitude
    r = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    mask = r > AMP
    if np.any(mask):
        IQ[mask] = IQ[mask] * (AMP / r[mask])[:, None]
    # slew: forward pass
    prev = np.zeros(2)
    for k in range(N):
        d = IQ[k] - prev
        nd = np.linalg.norm(d)
        maxd = SLEW * DT
        if nd > maxd:
            IQ[k] = prev + d * (maxd / nd)
        prev = IQ[k]
    # backward pass
    prev = np.zeros(2)
    for k in range(N - 1, -1, -1):
        d = IQ[k] - prev
        nd = np.linalg.norm(d)
        maxd = SLEW * DT
        if nd > maxd:
            IQ[k] = prev + d * (maxd / nd)
        prev = IQ[k]
    # re-clamp amplitude
    r = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    mask = r > AMP
    if np.any(mask):
        IQ[mask] = IQ[mask] * (AMP / r[mask])[:, None]
    return IQ

# ---- seed: DRAG-style Gaussian ----
def _seed_pulse():
    t = (np.arange(N) + 0.5) * DT
    T = 40.0
    sigma = 40.0 / 6.0
    g = np.exp(-((t - T / 2) ** 2) / (2 * sigma ** 2))
    # offset to 0 at edges: subtract edge value
    edge = g[0]
    g = g - edge
    # area pi in I
    area = np.sum(g) * DT
    if area != 0:
        g = g * (np.pi / area)
    IQ = np.zeros((N, 2))
    IQ[:, 0] = g
    # DRAG Q component proportional to derivative of g
    dg = np.gradient(g, DT)
    # standard DRAG: Q = -alpha * dg, alpha ~ 1/(2*pi*|anh|)
    alpha = 1.0 / (2.0 * np.pi * 0.25)
    IQ[:, 1] = -alpha * dg
    return _clamp(IQ)

def _legendre_basis():
    # 5 smooth basis functions on [0,1], zero at endpoints, orthonormal-ish
    x = (np.arange(N) + 0.5) / N
    basis = []
    # shifted Legendre polynomials P_k(2x-1), multiply by x(1-x) to vanish at ends
    from numpy.polynomial.legendre import legvander
    V = legvander(2 * x - 1, 6)
    w = x * (1 - x)
    for k in range(6):
        b = w * V[:, k]
        # orthonormalize
        for j in range(len(basis)):
            b = b - np.dot(b, basis[j]) / np.dot(basis[j], basis[j]) * basis[j]
        nrm = np.sqrt(np.dot(b, b))
        if nrm > 1e-12:
            basis.append(b / nrm)
    return np.array(basis).T  # (N, 6)

def pulse(target_gate, hardware_model):
    t0 = time.time()
    deadline = t0 + 300.0  # wall-clock budget with margin

    best = _seed_pulse()
    best_score = _score(best)

    # quick local search: perturb seed with Legendre basis
    B = _legendre_basis()
    # start coefficients
    coef = np.zeros(B.shape[1])
    cur = best.copy()
    cur_score = best_score

    # coordinate / Nelder-Mead style on coefficients
    def eval_coefs(c):
        IQ = best + B @ c
        IQ = _clamp(IQ)
        if not _valid(IQ):
            return 1e6
        return _score(IQ)

    # Nelder-Mead
    def nelder_mead(f, x0, iters=200):
        n = len(x0)
        simplex = [x0.copy()]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.05
            simplex.append(xi)
        vals = [f(s) for s in simplex]
        for it in range(iters):
            order = np.argsort(vals)
            simplex = [simplex[i] for i in order]
            vals = [vals[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + 1.0 * (centroid - simplex[-1])
            fr = f(xr)
            if fr < vals[0]:
                xe = centroid + 2.0 * (xr - centroid)
                fe = f(xe)
                if fe < fr:
                    simplex[-1], vals[-1] = xe, fe
                else:
                    simplex[-1], vals[-1] = xr, fr
            elif fr < vals[-2]:
                simplex[-1], vals[-1] = xr, fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < vals[-1]:
                    simplex[-1], vals[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = 0.5 * (simplex[i] + simplex[0])
                        vals[i] = f(simplex[i])
        order = np.argsort(vals)
        return simplex[order[0]], vals[order[0]]

    # run optimization with deadline guard
    iters = 150
    while time.time() < deadline - 20.0:
        # one Nelder-Mead round
        x0 = coef.copy()
        new_x, new_s = nelder_mead(eval_coefs, x0, iters=max(20, iters))
        if new_s < cur_score:
            coef = new_x
            cur = _clamp(best + B @ coef)
            cur_score = new_s
            if cur_score < best_score:
                best_score = cur_score
                best = cur.copy()
        # sometimes restart from random small coefficients
        if np.random.random() < 0.3 and time.time() < deadline - 30.0:
            x0 = 0.1 * np.random.randn(B.shape[1])
            new_x, new_s = nelder_mead(eval_coefs, x0, iters=30)
            if new_s < cur_score:
                coef = new_x
                cur = _clamp(best + B @ coef)
                cur_score = new_s
                if cur_score < best_score:
                    best_score = cur_score
                    best = cur.copy()
        # reduce iters to finish within budget
        iters = max(10, iters - 5)

    # final safety: ensure validity
    if not _valid(best):
        best = _clamp(best)
        if not _valid(best):
            best = _seed_pulse()
    return best

# keep the entry-point name/signature as in the current program: pulse(target_gate, hardware_model)
