"""Rewritten program: spectral-nulling DRAG pulse for three-level transmon X-pi gate."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25  # in rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Time grid
t = np.arange(N_STEPS) * DT  # 0 to 39.5 ns

# Precompute basis operators
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
# a_dag = a.T
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T
n_op = a_dag @ a_op  # diag(0, 1, 2)

# Lindblad operators
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)

# Precompute constant parts of Liouvillian
I3 = np.eye(3, dtype=complex)
L1dagL1 = L1.conj().T @ L1
L2dagL2 = L2.conj().T @ L2
dissip_L1 = np.kron(L1.T, L1) - 0.5 * (np.kron(I3, L1dagL1) + np.kron(L1dagL1.T, I3))
dissip_L2 = np.kron(L2.T, L2) - 0.5 * (np.kron(I3, L2dagL2) + np.kron(L2dagL2.T, I3))
dissip_total = dissip_L1 + dissip_L2

# Precompute anharmonicity part
n_nm1 = n_op @ (n_op - np.eye(3))
anharm_super = -1j * (np.kron(I3, 0.5 * ANHARMONICITY * n_nm1) - np.kron((0.5 * ANHARMONICITY * n_nm1).T, I3))

def build_liouvillian_fast(detuning, scale, I_val, Q_val):
    """Build the Liouvillian superoperator for one step using precomputed parts."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    H_drive = 0.5 * scale * (I_val * (a_op + a_dag) + 1j * Q_val * (a_op - a_dag))
    H_det = detuning * n_op
    
    # Commutator superoperator: -i[H, rho]
    H_total = H_det + H_drive
    comm = -1j * (np.kron(I3, H_total) - np.kron(H_total.T, I3))
    
    return comm + anharm_super + dissip_total

def simulate_pulse_fast(I, Q, detuning, scale):
    """Simulate the pulse and return mean fidelity across all initial states."""
    # Precompute Liouvillians using the correct function name
    liouvillians = [build_liouvillian_fast(detuning, scale, I[i], Q[i]) for i in range(N_STEPS)]
    # Precompute step propagators
    U_steps = [expm(Super * DT) for Super in liouvillians]
    
    # Initial states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_plus_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_minus_i = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # Precompute target density matrices
    rho_targets = [np.outer(psi_t, psi_t.conj()).flatten() for psi_t in target_states]
    
    fidelities = []
    for psi_init, rho_target in zip(initial_states, rho_targets):
        rho = np.outer(psi_init, psi_init.conj()).flatten()
        for U_step in U_steps:
            rho = U_step @ rho
        fid = np.real(np.dot(rho_target.conj(), rho))
        fid = np.clip(fid, 0, 1)
        fidelities.append(fid)
    
    return np.mean(fidelities)

def simulate_pulse(I, Q, detuning, scale):
    """Simulate the pulse and return mean fidelity across all initial states."""
    return simulate_pulse_fast(I, Q, detuning, scale)

def compute_score(I, Q):
    """Compute the score: mean + max infidelity over 9 cases."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            fid = simulate_pulse_fast(I, Q, det, scale)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def validate_pulse(I, Q):
    """Check amplitude and slew constraints."""
    # Amplitude bound
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.max(amplitudes) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero into step 1 and from last step to zero
    p = np.column_stack([I, Q])  # (80, 2)
    
    # From zero to step 0
    if np.linalg.norm(p[0]) / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(p[i+1] - p[i])
        if diff / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    if np.linalg.norm(p[-1]) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def gaussian_pulse(t, sigma, area, shift=0.0):
    """Gaussian envelope with given sigma, area, and time shift."""
    center = DURATION / 2.0 + shift
    g = area / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((t - center) / sigma)**2)
    # Offset to zero at edges
    g[0] = 0.0
    g[-1] = 0.0
    return g

def generate_pulse_from_params(params):
    """Generate I and Q pulses from 4 parameters: [amp_x, shift_x, amp_drag, shift_drag]."""
    amp_x, shift_x, amp_drag, shift_drag = params
    
    # Main X-pulse: Gaussian with area pi
    sigma = 40.0 / 6.0
    I = gaussian_pulse(t, sigma, np.pi, shift_x)
    I *= amp_x  # Scale
    
    # DRAG component: derivative of the Gaussian, with independent shift
    center_drag = DURATION / 2.0 + shift_drag
    Q_base = -(t - center_drag) / sigma**2 * np.exp(-0.5 * ((t - center_drag) / sigma)**2)
    # Normalize Q_base to have unit area-like scale
    max_abs = np.max(np.abs(Q_base))
    if max_abs > 0:
        Q_base = Q_base / max_abs
    Q_base[0] = 0.0
    Q_base[-1] = 0.0
    Q = amp_drag * Q_base
    
    return I, Q

def objective(params):
    """Objective function: score + penalty for constraint violations."""
    I, Q = generate_pulse_from_params(params)
    
    # Check constraints
    if not validate_pulse(I, Q):
        # Return a large penalty
        return 100.0
    
    return compute_score(I, Q)

def compute_score_fast(I, Q):
    """Compute the score: mean + max infidelity over 9 cases."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            fid = simulate_pulse_fast(I, Q, det, scale)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes (rad/ns)."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Initial guess: DRAG-style with physically-motivated Q component
    # sigma = 40/6, area = pi
    sigma = 40.0 / 6.0
    I0 = gaussian_pulse(t, sigma, np.pi, 0.0)
    
    # Derivative for DRAG
    center = DURATION / 2.0
    Q0_base = -(t - center) / sigma**2 * np.exp(-0.5 * ((t - center) / sigma)**2)
    max_abs = np.max(np.abs(Q0_base))
    if max_abs > 0:
        Q0_base = Q0_base / max_abs
    Q0_base[0] = 0.0
    Q0_base[-1] = 0.0
    Q0 = 0.1 * Q0_base  # Larger initial DRAG amplitude
    
    # Check if initial pulse is valid
    if not validate_pulse(I0, Q0):
        # Scale down to satisfy constraints
        max_amp = np.max(np.sqrt(I0**2 + Q0**2))
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp
            I0 *= scale
            Q0 *= scale
        # Check slew
        p = np.column_stack([I0, Q0])
        max_slew = 0
        if np.linalg.norm(p[0]) / DT > SLEW_BOUND:
            max_slew = max(max_slew, np.linalg.norm(p[0]) / DT / SLEW_BOUND)
        for i in range(N_STEPS - 1):
            s = np.linalg.norm(p[i+1] - p[i]) / DT
            if s > SLEW_BOUND:
                max_slew = max(max_slew, s / SLEW_BOUND)
        if np.linalg.norm(p[-1]) / DT > SLEW_BOUND:
            max_slew = max(max_slew, np.linalg.norm(p[-1]) / DT / SLEW_BOUND)
        if max_slew > 1:
            I0 /= max_slew
            Q0 /= max_slew
    
    # Optimize the 4 parameters
    # params = [amp_x, shift_x, amp_drag, shift_drag]
    best_score = 100.0
    best_params = np.array([1.0, 0.0, 0.1, 0.0])
    
    # Try multiple optimization attempts with different initial guesses
    initial_guesses = [
        np.array([1.0, 0.0, 0.1, 0.0]),
        np.array([1.0, 0.0, -0.1, 0.0]),
        np.array([1.0, 0.3, 0.1, 0.0]),
        np.array([1.0, -0.3, 0.1, 0.0]),
        np.array([1.0, 0.0, 0.1, 0.3]),
        np.array([1.0, 0.0, 0.1, -0.3]),
        np.array([0.98, 0.0, 0.08, 0.0]),
        np.array([1.02, 0.0, 0.12, 0.0]),
        np.array([1.0, 0.2, 0.1, 0.2]),
        np.array([1.0, -0.2, 0.1, -0.2]),
    ]
    
    for guess in initial_guesses:
        try:
            result = minimize(objective, guess, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_score < 100.0:  # Valid result found
        I, Q = generate_pulse_from_params(best_params)
        if validate_pulse(I, Q):
            return np.column_stack([I, Q])
    
    # Fallback: return the initial DRAG pulse
    return np.column_stack([I0, Q0])
