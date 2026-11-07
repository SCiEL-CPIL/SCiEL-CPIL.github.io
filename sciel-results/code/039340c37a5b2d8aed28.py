import numpy as np
from scipy.optimize import minimize

# ---- constants from the task contract ----
N = 80
DT = 0.5
T1 = 25000.0
T2 = 40000.0
A_MAX = 0.4
SLEW_MAX = 0.12

# 3x3 matrices
A = np.diag(np.sqrt([1.0, 2.0, 0.0]))
AD = A.conj().T
N_OP = AD @ A
N2 = N_OP @ N_OP

# Liouvillian super-operator blocks (vectorization: |psi> -> psi[:,None].reshape(9,1))
# sigma term: 0.5*I*(a+a^dagger) + 0.5*i*Q*(a-a^dagger)  = 0.5*(I+iQ)*a + 0.5*(I-iQ)*a^dagger
S_A = 0.5 * A
S_AD = 0.5 * AD
# Lindblad: a/sqrt(T1), n*sqrt(2/Tphi)
G0 = A / np.sqrt(T1)
G1 = N_OP * np.sqrt(2.0 / T2)
G0DG0 = G0.conj().T @ G0
G1DG1 = G1.conj().T @ G1

DETUNINGS = (-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002)
SCALES = (0.97, 1.0, 1.03)

# basis states
E0 = np.array([1, 0, 0], dtype=complex)
E1 = np.array([0, 1, 0], dtype=complex)
P0 = np.outer(E0, E0.conj())
P1 = np.outer(E1, E1.conj())
P10 = np.outer(E1, E0.conj())
P01 = np.outer(E0, E1.conj())

# fidelity projectors
def _fid_vec(psi):
    psi = psi / np.linalg.norm(psi)
    return (abs(psi[0]) ** 2 + abs(psi[1]) ** 2) / 2.0

def _fid_rho(rho):
    return 0.5 * (np.real(np.trace(rho @ P0)) + np.real(np.trace(rho @ P1)))

# Legendre polynomials on [0,1]
def _legendre(k, t):
    P0 = np.ones_like(t)
    if k == 0:
        return P0
    P1 = t
    if k == 1:
        return P1
    for n in range(2, k + 1):
        Pn = ((2 * n - 1) * t * P1 - (n - 1) * P0) / n
        P0, P1 = P1, Pn
    return P1

_T = np.linspace(0.0, 1.0, N)
B = np.stack([_legendre(k, _T) for k in range(8)], axis=1)  # (N,8)

# precompute Liouvillian blocks for the 9 cases
_CASES = []
for det in DETUNINGS:
    for s in SCALES:
        # static part: det*n + 0.5*an*n(n-1)  (an = -2*pi*0.25)
        static = det * N_OP + 0.5 * (-2 * np.pi * 0.25) * (N2 - N_OP)
        # per-case drive scaling
        sA = s * S_A
        sAD = s * S_AD
        # super-operator: rho' = -i[static+0.5(I+iQ)a+0.5(I-iQ)a^dagger, rho] + dissipators
        # Build as: L = A0 + A1*(I+iQ) + A2*(I-iQ)
        # A0 = -i[static, .] + D
        A0 = -1j * (static @ None)  # placeholder, built explicitly below
        # explicit super-operator matrices acting on rho (9x9)
        def super_op(H):
            return -1j * (np.kron(H, np.eye(9)) - np.kron(np.eye(9), H.conj().T))
        def deph(Dmat):
            return np.kron(np.eye(9), Dmat) - 0.5 * np.kron(Dmat.conj().T @ Dmat, np.eye(9)) - 0.5 * np.kron(np.eye(9), Dmat.conj().T @ Dmat)
        L0 = super_op(static) + deph(G0) + deph(G1)
        L1 = super_op(0.5 * A) + deph(0)  # coefficient (I+iQ)
        L2 = super_op(0.5 * AD) + deph(0) # coefficient (I-iQ)
        # include scale in L1,L2
        L1 = s * super_op(0.5 * A)
        L2 = s * super_op(0.5 * AD)
        _CASES.append((L0, L1, L2))


def _simulate(I, Q, case):
    L0, L1, L2 = case
    states = [E0, E1,
              (E0 + E1) / np.sqrt(2), (E0 - E1) / np.sqrt(2),
              (E0 + 1j * E1) / np.sqrt(2), (E0 - 1j * E1) / np.sqrt(2)]
    fids = np.zeros(6)
    for si, psi in enumerate(states):
        rho = np.outer(psi, psi.conj()).reshape(9)
        for i in range(N):
            L = L0 + (I[i] + 1j * Q[i]) * L1 + (I[i] - 1j * Q[i]) * L2
            rho = rho + DT * (L @ rho)
        fids[si] = _fid_rho(rho.reshape(3, 3))
    return fids


def _score(I, Q):
    fids = np.array([_simulate(I, Q, c) for c in _CASES])  # (9,6)
    infid = 1.0 - fids.mean(axis=1)
    return float(infid.mean() + infid.max())


def _valid(I, Q):
    amp = np.sqrt(I * I + Q * Q)
    if np.any(amp > A_MAX + 1e-12):
        return False
    d = np.zeros(N + 1)
    d[1:N + 1] = np.sqrt(np.diff(I) ** 2 + np.diff(Q) ** 2)
    d[0] = np.hypot(I[0], Q[0])
    if np.any(d / DT > SLEW_MAX + 1e-12):
        return False
    return True


def _objective(x):
    I = B @ x[0:8]
    Q = B @ x[8:16]
    if not _valid(I, Q):
        return 1e6 + _penalty(I, Q)
    return _score(I, Q)


def _penalty(I, Q):
    amp = np.sqrt(I * I + Q * Q)
    p = 0.0
    over = np.maximum(0.0, amp - A_MAX)
    p += 100.0 * np.sum(over ** 2)
    d = np.zeros(N + 1)
    d[1:N + 1] = np.sqrt(np.diff(I) ** 2 + np.diff(Q) ** 2)
    d[0] = np.hypot(I[0], Q[0])
    over = np.maximum(0.0, d / DT - SLEW_MAX)
    p += 100.0 * np.sum(over ** 2)
    return p


def _seed_pulse():
    # DRAG-style gaussian: sigma = 40/6 ns, area pi over 40 ns, offset to 0 at edges
    t = np.arange(N) * DT
    sigma = 40.0 / 6.0
    g = np.exp(-((t - 20.0) ** 2) / (2 * sigma ** 2))
    # offset so edges are ~0
    g = g - (g[0] + g[-1]) / 2.0
    g = np.maximum(g, 0.0)
    # normalize area to pi
    area = np.sum(g) * DT
    g = g * (np.pi / area)
    I = g
    Q = np.zeros(N)
    return I, Q


def pulse(target_gate, hardware_model):
    import time
    t0 = time.time()
    DEADLINE = 300.0

    # seed
    I0, Q0 = _seed_pulse()
    best_I, best_Q = I0.copy(), Q0.copy()
    best_score = _score(best_I, best_Q)

    # build initial x from seed via least squares
    xI = np.linalg.lstsq(B, I0, rcond=None)[0]
    xQ = np.linalg.lstsq(B, Q0, rcond=None)[0]
    x0 = np.concatenate([xI, xQ])

    # amplitude bound: ensure |I|,|Q| within A_MAX/2 each to keep sqrt<=0.4 margin
    bound = A_MAX / 2.0
    bounds = [(-bound, bound)] * 16

    def time_ok():
        return (time.time() - t0) < DEADLINE

    # local optimization with L-BFGS-B
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 400, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        if time_ok():
            x = res.x
            I = B @ x[0:8]
            Q = B @ x[8:16]
            if _valid(I, Q):
                s = _score(I, Q)
                if s < best_score:
                    best_score, best_I, best_Q = s, I.copy(), Q.copy()
    except Exception:
        pass

    # a couple of random restarts if time remains
    rng = np.random.default_rng(0)
    for r in range(6):
        if not time_ok():
            break
        x = x0 + rng.normal(0, 0.05, 16)
        try:
            res = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'maxfun': 800, 'ftol': 1e-12, 'gtol': 1e-8})
            if time_ok():
                x = res.x
                I = B @ x[0:8]
                Q = B @ x[8:16]
                if _valid(I, Q):
                    s = _score(I, Q)
                    if s < best_score:
                        best_score, best_I, best_Q = s, I.copy(), Q.copy()
        except Exception:
            pass

    return np.column_stack([best_I, best_Q])
