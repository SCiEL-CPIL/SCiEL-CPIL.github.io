"""Direct constraint-satisfying optimization of 80 I/Q samples for a 3-level transmon X-pi gate."""
import numpy as np
from scipy.optimize import minimize
import time

# ----------------------------- constants -----------------------------
ANHARMONICITY = -2.0 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
Tphi = 40000.0
DT = 0.5
N = 80
DURATION = N * DT

# nine hardware cases
_DET = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCA = np.array([0.97, 1.0, 1.03])
CASES = [(d, s) for d in _DET for s in _SCA]

# ----------------------------- operators (3x3) -----------------------------
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2.0)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
# superoperators
def _super_op(op):
    return np.kron(op, np.conj(op.T))
_S_A = _super_op(_a)
_S_ADAG = _super_op(_adag)
_S_N = _super_op(_n)
_S_I = _super_op(np.eye(3, dtype=complex))
# commutator superoperators [A, rho] = A rho - rho A
_C_A = _S_A @ _S_I - _S_I @ _S_A
_C_ADAG = _S_ADAG @ _S_I - _S_I @ _S_ADAG
_C_N = _S_N @ _S_I - _S_I @ _S_N
# dissipator superoperators D[L]
def _diss_super(L):
    Ldag = L.conj().T
    S_L = _super_op(L)
    S_Ldag = _super_op(Ldag)
    return 2.0*(S_L @ S_Ldag @ _S_I - 0.5*(S_Ldag @ S_L + S_L @ S_Ldag))
_D_A = _diss_super(_a / np.sqrt(T1))
_D_N = _diss_super(_n * np.sqrt(2.0/Tphi))

# basis states (3,)
_B0 = np.array([1, 0, 0], dtype=complex)
_B1 = np.array([0, 1, 0], dtype=complex)
_B2 = np.array([0, 0, 1], dtype=complex)
def _ket(v):
    return np.kron(v, v.conj())  # |v><v| vectorized
_BASIS = [_ket(_B0), _ket(_B1),
          _ket((_B0 + _B1)/np.sqrt(2)), _ket((_B0 - _B1)/np.sqrt(2)),
          _ket((_B0 + 1j*_B1)/np.sqrt(2)), _ket((_B0 - 1j*_B1)/np.sqrt(2))]

# ----------------------------- simulation -----------------------------
def _fidelity(rho, b):
    return float(np.real(np.vdot(b, rho)))

def _simulate_pulse(pulse, detuning, scale):
    """Return (6,) fidelity vector to the basis states."""
    # static part of Liouvillian
    H0 = detuning * _n + 0.5 * ANHARMONICITY * _n @ (_n - _S_I)
    L0 = -1j * _C_N * detuning - 1j * _C_N * (0.5*ANHARMONICITY) * (_S_N - _S_I) \
         + _D_A + _D_N
    # note: 0.5*anharm*n(n-1) = 0.5*anharm*(n^2 - n); commutator with (n^2 - n)
    # _C_N*(n-1) = _C_N*_S_N - _C_N  (since [_n, n-1] = [n,n]-[n,1]=0? no)
    # Actually [n, n-1] = 0, so commutator of H with rho uses [n(n-1),rho]=n(n-1)rho-rho n(n-1).
    # Build directly to be safe:
    H0_op = detuning * _n + 0.5 * ANHARMONICITY * (_n @ _n - _n)
    L0 = -1j * (_super_op(H0_op) @ _S_I - _S_I @ _super_op(H0_op)) + _D_A + _D_N

    fid = np.zeros(6)
    for i in range(6):
        rho = _BASIS[i].copy()
        for k in range(N):
            I, Q = pulse[k, 0], pulse[k, 1]
            H1 = 0.5 * scale * (_a + _adag * 1j * 1j)  # placeholder
            # drive: I (a+a^dag) + i Q (a - a^dag)
            drive = I * (_a + _adag) + 1j * Q * (_a - _adag)
            H1 = 0.5 * scale * drive
            S1 = _super_op(H1)
            L1 = -1j * (S1 @ _S_I - _S_I @ S1)
            L = L0 + L1
            # first-order step (matches the stepwise exponentiation to O(dt))
            rho = rho + DT * (L @ rho)
            # renormalize trace to suppress numerical drift
            tr = np.real(np.vdot(np.ones(9), rho))
            if tr != 0:
                rho = rho / tr
        # fidelity to |1>
        fid[i] = _fidelity(rho, _B1)
    return fid

def _infid_for_case(pulse, detuning, scale):
    f = _simulate_pulse(pulse, detuning, scale)
    return 1.0 - np.mean(f)

def _score(pulse):
    vals = [_infid_for_case(pulse, d, s) for (d, s) in CASES]
    return float(np.mean(vals) + np.max(vals))

# ----------------------------- seed (DRAG) -----------------------------
def _drag_seed():
    x = np.linspace(-1.0, 1.0, N)
    sig = (40.0/6.0) / DURATION  # sigma in normalized x units
    g = np.exp(-0.5 * (x/sig)**2)
    # offset so edges ~0
    g = g - g[0]
    g = np.maximum(g, 0.0)
    # area pi: scale g so sum(g)*DT = pi  (integral of amplitude)
    area = np.sum(g) * DT
    if area > 0:
        g = g * (np.pi / area)
    # DRAG: Q proportional to derivative of I (Gaussian derivative)
    I = g
    Q = np.gradient(I) * 0.0  # small baseline; optimizer will shape it
    # small initial DRAG beta to break symmetry
    Q = 0.05 * np.gradient(I)
    pulse = np.column_stack([I, Q])
    # enforce constraints approximately via projection
    pulse = _project(pulse)
    return pulse

# ----------------------------- projection (convex-ish) -----------------------------
# Enforce: ||p_k|| <= A  and  ||p_{k+1}-p_k||/dt <= S, and boundary to zero.
# We solve a least-norm problem: min ||p - target||^2 s.t. box on diffs and amplitude.
# Use a simple iterative clamping + a scipy least-squares for the diff constraints.
def _project(pulse):
    A = AMPLITUDE_BOUND
    S = SLEW_BOUND
    p = pulse.astype(float).copy()
    for _ in range(60):
        # amplitude clamp
        r = np.linalg.norm(p, axis=1)
        m = r > A
        if np.any(m):
            p[m] = p[m] * (A / r[m])[:, None]
        # slew clamp: limit each consecutive difference to S*dt
        step = S * DT
        # from zero to p0
        r0 = np.linalg.norm(p[0])
        if r0 > step:
            p[0] = p[0] * (step / r0)
        for k in range(N - 1):
            d = p[k+1] - p[k]
            rd = np.linalg.norm(d)
            if rd > step:
                p[k+1] = p[k] + d * (step / rd)
        # from last to zero
        rl = np.linalg.norm(p[-1])
        if rl > step:
            p[-1] = p[-1] * (step / rl)
    return p

# ----------------------------- objective with penalty -----------------------------
def _objective(x):
    p = _project(x)
    return _score(p)

# ----------------------------- main -----------------------------
def pulse(target_gate, hardware_model):
    deadline = time.time() + 290.0
    best = _drag_seed()
    best_score = _score(best)

    # L-BFGS-B refinement on the projected samples
    try:
        res = minimize(_objective, best.ravel(), method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 200,
                                'ftol': 1e-12, 'gtol': 1e-9, 'disp': False})
        cand = _project(res.x.reshape(N, 2))
        cs = _score(cand)
        if cs < best_score:
            best, best_score = cand, cs
    except Exception:
        pass

    if time.time() > deadline - 20:
        return best

    # local Nelder-Mead polish on a reduced parameterization around best
    try:
        res2 = minimize(_objective, best.ravel(), method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-12,
                                 'adaptive': True, 'disp': False})
        cand2 = _project(res2.x.reshape(N, 2))
        cs2 = _score(cand2)
        if cs2 < best_score:
            best, best_score = cand2, cs2
    except Exception:
        pass

    return best

# keep the public entry point name used by the seed
quantum_seed = pulse
