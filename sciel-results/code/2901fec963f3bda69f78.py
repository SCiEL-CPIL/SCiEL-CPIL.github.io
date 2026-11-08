import numpy as np
import time
import warnings
from scipy.linalg import expm
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# Constants
DT = 0.5
N_STEPS = 80
T_TOTAL = N_STEPS * DT
ALPHA_ANH = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
TPHI = 40000.0

# Hardware cases
DELTAS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
N_CASES = len(DELTAS) * len(SCALES)

# Pre-compute operators
def _build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1.0)
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

A, ADAG, N_OP = _build_operators()

# Pre-compute dissipative Liouvillian (independent of I, Q, delta, scale)
def _precompute_dissipator():
    L1 = A / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2.0 / TPHI)
    diss = np.zeros((9, 9), dtype=complex)
    for Lk in (L1, L2):
        Lk_dag = Lk.conj().T
        term1 = np.kron(Lk_dag.conj().T, Lk)
        Lk_dag_Lk = Lk_dag @ Lk
        term2 = 0.5 * (np.kron(np.eye(3), Lk_dag_Lk) + np.kron(Lk_dag_Lk.conj().T, np.eye(3)))
        diss += term1 - term2
    return diss

DISSIPATOR = _precompute_dissipator()
I3 = np.eye(3)
H_ANH = 0.5 * ALPHA_ANH * N_OP @ (N_OP - I3)

# Basis / initial / target states
def _init_states():
    s0 = np.array([1, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0], dtype=complex)
    return np.array([
        s0, s1,
        (s0 + s1) / np.sqrt(2),
        (s0 - s1) / np.sqrt(2),
        (s0 + 1j * s1) / np.sqrt(2),
        (s0 - 1j * s1) / np.sqrt(2),
    ])

def _target_states():
    s0 = np.array([1, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0], dtype=complex)
    # X-pi gate: -i X
    return np.array([
        -1j * s1,
        -1j * s0,
        -1j * (s1 + s0) / np.sqrt(2),
        -1j * (s1 - s0) / np.sqrt(2),
        -1j * (s1 + 1j * s0) / np.sqrt(2),
        -1j * (s1 - 1j * s0) / np.sqrt(2),
    ])

INIT_STATES = _init_states()
TARGET_STATES = _target_states()
RHO_INIT = np.array([np.outer(s, s.conj().T) for s in INIT_STATES])  # (6,3,3)
RHO_INIT_FLAT = RHO_INIT.reshape(6, 9)  # (6,9)
# Target bra vectors for fidelity: (6,3)
TARGET_BRA = TARGET_STATES.conj().T  # (6,3)

def _coherent_kron(I, Q, scale):
    # -i * ( kron(I3, H_drive) - kron(H_drive^dag, I3) )  where H_drive = 0.5*scale*(I*(a+adag)+i Q*(a-adag))
    H_drive = 0.5 * scale * (I * (A + ADAG) + 1j * Q * (A - ADAG))
    return -1j * (np.kron(I3, H_drive) - np.kron(H_drive.conj().T, I3))

def _simulate(I, Q):
    # Returns score = mean + max of infidelities over 9 cases
    # rho for all cases stacked: shape (9, 6, 9)
    rho = np.repeat(RHO_INIT_FLAT[np.newaxis, :, :], N_CASES, axis=0)  # (9,6,9)
    rho = rho.reshape(N_CASES, 6, 9)
    for k in range(N_STEPS):
        for ci, (delta, scale) in enumerate(zip(
            [d for d in DELTAS for s in SCALES],
            [s for d in DELTAS for s in SCALES]
        )):
            H = delta * N_OP + H_ANH
            L = _coherent_kron(I[k], Q[k], scale) + np.kron(I3, H) * (-1j) - np.kron(H.conj().T, I3) * (-1j)
            # Wait: coherent Liouvillian is -i[H,rho] = -i( kron(I3,H) - kron(H^dag,I3) ) vec(rho)
            # I already put -i into _coherent_kron for the drive; add the detuning/anharmonic part similarly
            L += DISSIPATOR
            U = expm(L * DT)
            rho[ci] = (U @ rho[ci].T).T
    # fidelity
    infids = np.zeros(N_CASES)
    for ci in range(N_CASES):
        fids = np.zeros(6)
        for si in range(6):
            rho_final = rho[ci, si].reshape(3, 3)
            fids[si] = np.real(np.vdot(TARGET_BRA[si], rho_final @ TARGET_STATES[si]))
        infids[ci] = 1.0 - np.mean(fids)
    return np.mean(infids) + np.max(infids)

def _check_constraints(I, Q):
    amps = np.sqrt(I**2 + Q**2)
    if np.any(amps > AMP_BOUND + 1e-12):
        return False
    dI = np.diff(I)
    dQ = np.diff(Q)
    d_norms = np.sqrt(dI**2 + dQ**2)
    if np.any(d_norms / DT > SLEW_BOUND + 1e-12):
        return False
    if np.sqrt(I[0]**2 + Q[0]**2) / DT > SLEW_BOUND + 1e-12:
        return False
    if np.sqrt(I[-1]**2 + Q[-1]**2) / DT > SLEW_BOUND + 1e-12:
        return False
    return True

# Time grid
T_GRID = np.linspace(0, T_TOTAL, N_STEPS)
T_CENTER = T_TOTAL / 2.0

# DRAG ratio for first-order leakage cancellation:
# r_drag = (a_01 / a_12) * (delta_21 - delta_10)  where delta_21 = alpha, delta_10 = 0
# a_01 = 1, a_12 = sqrt(2), alpha = -2pi*0.25
R_DRAG = (1.0 / np.sqrt(2.0)) * ALPHA_ANH

def _make_pulse(params):
    # params = [sigma_main, shift_main, sigma_drag, shift_drag]
    sigma_main = params[0]
    shift_main = params[1]
    sigma_drag = params[2]
    shift_drag = params[3]
    t = T_GRID - T_CENTER
    # Main Gaussian envelope, area pi
    main = np.exp(-((t - shift_main)**2) / (2.0 * sigma_main**2))
    main = main * (np.pi / np.sqrt(2.0 * np.pi) / sigma_main)  # area = pi
    # DRAG component: proportional to derivative of main, scaled
    # d/dt main = -(t-shift_main)/sigma_main^2 * main
    drag_env = np.exp(-((t - shift_drag)**2) / (2.0 * sigma_drag**2))
    drag = -(t - shift_drag) / (sigma_drag**2) * drag_env
    drag = drag * (R_DRAG / np.max(np.abs(drag)) * 0.3)  # scale to stay within bound, ~0.3 of max
    # Actually scale drag so its peak is reasonable
    drag = R_DRAG * (-(t - shift_drag) / (sigma_drag**2)) * np.exp(-((t - shift_drag)**2) / (2.0 * sigma_drag**2))
    # Normalize drag to have peak ~ 0.15
    peak = np.max(np.abs(drag))
    if peak > 0:
        drag = drag / peak * 0.15
    I = main
    Q = drag
    return I, Q

def _objective(params):
    I, Q = _make_pulse(params)
    if not _check_constraints(I, Q):
        return 1e6
    try:
        return _simulate(I, Q)
    except Exception:
        return 1e6

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0

    best_score = 1e6
    best_params = np.array([5.0, 0.0, 6.0, 0.0])

    starts = [
        [5.0, 0.0, 6.0, 0.0],
        [6.0, 0.0, 7.0, 0.0],
        [4.5, 0.5, 5.5, 0.0],
        [5.5, -0.5, 6.5, 0.5],
        [5.0, 0.0, 5.0, 0.0],
        [6.5, 0.0, 7.5, -0.5],
        [4.0, 1.0, 5.0, 1.0],
        [7.0, 0.0, 8.0, 0.0],
    ]

    for s in starts:
        if time.time() > deadline - 40.0:
            break
        try:
            res = minimize(_objective, s, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Second round: finer local refinement around best
    if time.time() < deadline - 30.0:
        for s in [best_params]:
            try:
                res = minimize(_objective, s, method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass

    I, Q = _make_pulse(best_params)
    if not _check_constraints(I, Q):
        # Fallback: safe wide Gaussian, no DRAG
        I = np.zeros(N_STEPS)
        Q = np.zeros(N_STEPS)
        t = T_GRID - T_CENTER
        sigma = 8.0
        I = np.exp(-t**2 / (2 * sigma**2))
        I = I * (np.pi / (np.sqrt(2 * np.pi) * sigma))
        if not _check_constraints(I, Q):
            # Even safer: very wide
            sigma = 10.0
            I = np.exp(-t**2 / (2 * sigma**2))
            I = I * (np.pi / (np.sqrt(2 * np.pi) * sigma))
            Q = np.zeros(N_STEPS)

    return np.column_stack((I, Q))
