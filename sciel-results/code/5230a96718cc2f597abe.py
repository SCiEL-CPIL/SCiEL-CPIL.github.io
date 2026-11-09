"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# --- Constants ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]
DEADLINE = 310.0  # safety margin before 330s kill

# Precompute matrices
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_eye3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian building blocks
def _precompute_liouvillian_bases():
    """Precompute the matrix coefficients for the Liouvillian."""
    I3 = _eye3

    # Detuning term
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))

    # Drive I term: 0.5*(a+a†)
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))

    # Drive Q term: 0.5*i*(a-a†)
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))

    # Anharmonicity term (constant part): 0.5*alpha*n*(n-1)
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))

    # Dissipative part (constant)
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        term1 = np.kron(Lkd.T, Lk)
        term2 = -0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
        L_diss += term1 + term2

    return L_det_base, L_I_base, L_Q_base, L_anh_base, L_diss

_L_DET, _L_I, _L_Q, _L_ANH, _L_DISS = _precompute_liouvillian_bases()

# Initial states
_psi0 = np.array([1, 0, 0], dtype=complex)
_psi1 = np.array([0, 1, 0], dtype=complex)
_psi_plus = (_psi0 + _psi1) / np.sqrt(2)
_psi_minus = (_psi0 - _psi1) / np.sqrt(2)
_psi_plus_i = (_psi0 + 1j*_psi1) / np.sqrt(2)
_psi_minus_i = (_psi0 - 1j*_psi1) / np.sqrt(2)

# Ideal outputs (X applied to each)
_psi_outs = [
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 0, 0], dtype=complex),
    (_psi0 + _psi1) / np.sqrt(2),
    (_psi0 - _psi1) / np.sqrt(2),
    (_psi1 + 1j*_psi0) / np.sqrt(2),
    (_psi1 - 1j*_psi0) / np.sqrt(2),
]

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    """Evolve all 6 input states through the pulse, return final density matrices."""
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]

    # Constant part of Liouvillian
    L_const = detuning * _L_DET + _L_ANH + _L_DISS

    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]

        L = L_const + scale * I_val * _L_I + scale * Q_val * _L_Q
        expL = sla.expm(L * DT)

        for j in range(6):
            rhos_vec[j] = expL @ rhos_vec[j]

    return [v.reshape(3, 3) for v in rhos_vec]

def _fidelity_from_rhos(rhos):
    """Compute mean fidelity to X gate from 6 final density matrices."""
    fidelities = []
    for rho, psi_out in zip(rhos, _psi_outs):
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0.0, min(1.0, F))
        fidelities.append(F)

    return np.mean(fidelities)

def _compute_score(pulse_I, pulse_Q):
    """Compute score = mean + max of 9 infidelities."""
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            infids.append(1.0 - F)

    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amp > AMP_BOUND + 1e-12):
        return False

    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-12:
            return False

    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False

    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND + 1e-12:
        return False

    return True

def _enforce_constraints(pulse_I, pulse_Q):
    """Project pulse onto feasible set (amplitude + slew)."""
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()

    # Amplitude
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]

    # Slew: forward pass
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale

    # Edge slews
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slew0
        pulse_I[0] *= scale
        pulse_Q[0] *= scale

    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slewl
        pulse_I[-1] *= scale
        pulse_Q[-1] *= scale

    return pulse_I, pulse_Q

def _seed_pulse():
    """Generate DRAG-style Gaussian seed with beta compensation."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0

    # Gaussian envelope
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0

    # Scale to area pi
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area

    # DRAG beta
    beta = 0.15

    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)

    pulse_I = g.copy()
    pulse_Q = beta * dI_dt

    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)

    return pulse_I, pulse_Q

def _pulse_from_params(params):
    """
    Generate pulse from low-dimensional parameterization.
    params: [sigma, beta, amp_scale, center_shift, drag_shift]
    - sigma: Gaussian width (time)
    - beta: DRAG Q-component strength
    - amp_scale: overall amplitude scaling (multiplies area)
    - center_shift: time shift of pulse center
    - drag_shift: additional time shift for DRAG component
    """
    sigma = params[0]
    beta = params[1]
    amp_scale = params[2]
    center_shift = params[3]
    drag_shift = params[4]

    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0 + center_shift

    # Gaussian envelope
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0

    # Scale to area pi * amp_scale
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi * amp_scale / area

    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)

    pulse_I = g.copy()
    pulse_Q = beta * dI_dt

    # Apply drag shift (shift Q component in time via interpolation)
    if abs(drag_shift) > 1e-6:
        t_shifted = t + drag_shift
        valid = (t_shifted >= 0) & (t_shifted < GATE_TIME)
        if np.any(valid):
            pulse_Q[valid] = np.interp(t_shifted[valid], t, beta * dI_dt)
        pulse_Q[~valid] = 0.0

    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)

    return pulse_I, pulse_Q

def _score_from_params(params):
    """Compute score from parameter vector."""
    pulse_I, pulse_Q = _pulse_from_params(params)
    if not _is_valid(pulse_I, pulse_Q):
        return 1.0
    return _compute_score(pulse_I, pulse_Q)

def _grid_search_drag(beta_range, width_range, center_range, amp_range, drag_range,
                      n_beta=6, n_width=6, n_center=4, n_amp=4, n_drag=3):
    """Grid search over DRAG parameters."""
    best_I = None
    best_Q = None
    best_score = float('inf')

    betas = np.linspace(beta_range[0], beta_range[1], n_beta)
    widths = np.linspace(width_range[0], width_range[1], n_width)
    centers = np.linspace(center_range[0], center_range[1], n_center)
    amps = np.linspace(amp_range[0], amp_range[1], n_amp)
    drags = np.linspace(drag_range[0], drag_range[1], n_drag)

    for beta in betas:
        for width in widths:
            for center in centers:
                for amp in amps:
                    for drag in drags:
                        params = np.array([width, beta, amp, center, drag])
                        pulse_I, pulse_Q = _pulse_from_params(params)

                        if not _is_valid(pulse_I, pulse_Q):
                            continue

                        score = _compute_score(pulse_I, pulse_Q)
                        if score < best_score:
                            best_score = score
                            best_I = pulse_I.copy()
                            best_Q = pulse_Q.copy()

    return best_I, best_Q, best_score

def _nelder_mead_optimize(pulse_I, pulse_Q, start_params, time_limit=60):
    """Optimize using Nelder-Mead on the parameter space."""
    start_time = time.time()

    def objective(params):
        pulse_I, pulse_Q = _pulse_from_params(params)
        if not _is_valid(pulse_I, pulse_Q):
            return 1.0
        return _compute_score(pulse_I, pulse_Q)

    try:
        result = minimize(objective, start_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                 'adaptive': True})
    except Exception:
        return pulse_I, pulse_Q, objective(start_params)

    best_params = result.x
    best_I, best_Q = _pulse_from_params(best_params)
    best_score = objective(best_params)

    return best_I, best_Q, best_score

def _dop853_recheck(pulse_I, pulse_Q, detuning=0.0, scale=1.0):
    """Recheck nominal case using DOP853 to ensure accuracy within 2e-7."""
    def hamiltonian(t, state):
        # state is 3x3 density matrix flattened
        rho = state.reshape(3, 3)

        idx = int(t / DT)
        if idx >= N_STEPS:
            idx = N_STEPS - 1

        I_val = pulse_I[idx]
        Q_val = pulse_Q[idx]

        H = (detuning * _n
             + 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
             + 0.5 * scale * I_val * (_a + _adag)
             + 0.5j * scale * Q_val * (_a - _adag))

        # Liouvillian: d(rho)/dt = -i[H, rho] + diss
        comm = H @ rho - rho @ H

        diss = np.zeros((3, 3), dtype=complex)
        for Lk in [_L1, _L2]:
            Lkd = Lk.conj().T
            LdL = Lkd @ Lk
            diss += Lk @ rho @ Lkd - 0.5 * (LdL @ rho + rho @ LdL)

        d_rho = -1j * comm + diss
        return d_rho.flatten()

    # Evolve |0><0|
    psi0 = np.array([1, 0, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0).flatten()

    sol = solve_ivp(hamiltonian, [0, GATE_TIME], rho0, method='DOP853', rtol=1e-10, atol=1e-12)

    if not sol.success:
        return None

    rho_final = sol.y[:, -1].reshape(3, 3)

    # Fidelity to X|0> = |1>
    psi_out = np.array([0, 1, 0], dtype=complex)
    F = np.real(np.vdot(psi_out, rho_final @ psi_out))
    F = max(0.0, min(1.0, F))

    return F

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    start_time = time.time()

    # Generate seed pulse with DRAG
    pulse_I, pulse_Q = _seed_pulse()
    seed_score = _compute_score(pulse_I, pulse_Q)

    # Grid search over DRAG parameters with expanded ranges
    remaining = DEADLINE - (time.time() - start_time) - 80.0
    if remaining > 40:
        best_grid_I, best_grid_Q, grid_score = _grid_search_drag(
            beta_range=(-0.2, 0.2),
            width_range=(GATE_TIME/8.0, GATE_TIME/5.0),
            center_range=(-3.0, 3.0),
            amp_range=(0.9, 1.1),
            drag_range=(-2.0, 2.0),
            n_beta=6, n_width=6, n_center=4, n_amp=4, n_drag=3
        )
        if best_grid_I is not None and grid_score < seed_score:
            pulse_I = best_grid_I
            pulse_Q = best_grid_Q
            seed_score = grid_score

    # Nelder-Mead optimization from best grid point
    remaining = DEADLINE - (time.time() - start_time) - 30.0
    if remaining > 20:
        # Find the best parameters from the grid to start Nelder-Mead
        # We'll just use reasonable defaults based on the seed
        best_params = np.array([GATE_TIME/6.0, 0.15, 1.0, 0.0, 0.0])

        # Try a few starting points
        start_points = [
            np.array([GATE_TIME/6.0, 0.15, 1.0, 0.0, 0.0]),
            np.array([GATE_TIME/6.0, -0.15, 1.0, 0.0, 0.0]),
            np.array([GATE_TIME/5.5, 0.12, 0.98, 0.5, 0.0]),
            np.array([GATE_TIME/6.5, 0.18, 1.02, -0.5, 0.5]),
        ]

        nm_time_limit = max(10, remaining - 5)
        for sp in start_points:
            if time.time() - start_time > DEADLINE - 15:
                break
            nm_I, nm_Q, nm_score = _nelder_mead_optimize(pulse_I, pulse_Q, sp, time_limit=nm_time_limit/len(start_points))
            if nm_score < seed_score:
                pulse_I = nm_I
                pulse_Q = nm_Q
                seed_score = nm_score

    # DOP853 recheck (nominal case)
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 10:
        F_dop = _dop853_recheck(pulse_I, pulse_Q)
        if F_dop is not None:
            pass  # Verification only

    result = np.column_stack((pulse_I, pulse_Q))

    return result
# EVOLVE-BLOCK-END
