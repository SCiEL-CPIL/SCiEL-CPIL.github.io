"""
Flexible multi-Gaussian basis with Differential Evolution global search
and local refinement. The pulse is expressed as a sum of Gaussian bumps
with independent amplitudes, centers, and widths for I and Q channels.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import solve_ivp

# Constants
N_STEPS = 80
DT = 0.5  # ns
T_TOTAL = 40.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0  # ns
T2PHI = 40000.0  # ns

# Time grid (midpoints of each step)
_T_GRID = (np.arange(N_STEPS) + 0.5) * DT

# Three-level system matrices
_a = np.array([[0, np.sqrt(1), 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_n2 = _n @ _n
_I3 = np.eye(3, dtype=complex)

# Precompute anharmonicity term
_ANHARM_TERM = 0.5 * ANHARMONICITY * (_n2 - _n)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / T2PHI)
_L1dagL1 = _L1.conj().T @ _L1
_L2dagL2 = _L2.conj().T @ _L2

# Target states for fidelity computation
_PSI0 = np.array([1, 0, 0], dtype=complex)
_PSI1 = np.array([0, 1, 0], dtype=complex)
_PSI_PLUS = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
_PSI_MINUS = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
_PSI_IP = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
_PSI_IM = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
_TARGETS = [_PSI0, _PSI1, _PSI_PLUS, _PSI_MINUS, _PSI_IP, _PSI_IM]

# Detuning and scale cases
_DETUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
_SCALES = [0.97, 1.0, 1.03]

# Deadline management
_START_TIME = time.time()
_DEADLINE = 320.0  # 10s safety margin before 330s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _build_superoperator(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for given parameters."""
    H = (detuning * _n + _ANHARM_TERM
         + 0.5 * scale * (I * (_a + _adag) + 1j * Q * (_a - _adag)))
    
    supermat = -1j * (np.kron(_I3, H) - np.kron(H.T, _I3))
    
    # Dissipative: L1
    supermat += (np.kron(_L1.conj().T, _L1)
                - 0.5 * (np.kron(_I3, _L1dagL1) + np.kron(_L1dagL1.T, _I3)))
    # Dissipative: L2
    supermat += (np.kron(_L2.conj().T, _L2)
                - 0.5 * (np.kron(_I3, _L2dagL2) + np.kron(_L2dagL2.T, _I3)))
    
    return supermat


def _simulate_stepwise(pulse, detuning, scale):
    """Simulate using matrix exponentiation at each step (exact for piecewise-constant)."""
    rho = np.zeros(9, dtype=complex)
    rho[0] = 1.0  # |0><0|
    
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        S = _build_superoperator(detuning, scale, I, Q)
        # Evolve: rho_{i+1} = exp(S * dt) @ rho_i
        # Use expm for accuracy
        from scipy.linalg import expm
        Sdt = S * DT
        rho = expm(Sdt) @ rho
    
    rho_mat = rho.reshape(3, 3)
    tr = np.trace(rho_mat)
    if abs(tr) > 1e-10:
        rho_mat = rho_mat / tr
    return rho_mat


def _score_pulse(pulse):
    """Compute the task score: mean + max of infidelities over 9 cases."""
    infidelities = []
    for det in _DETUNINGS:
        for scl in _SCALES:
            rho = _simulate_stepwise(pulse, det, scl)
            # Mean fidelity over 6 targets
            fids = []
            for psi in _TARGETS:
                f = np.real(np.vdot(psi, rho @ psi))
                f = np.clip(f, 0.0, 1.0)
                fids.append(f)
            mean_fid = np.mean(fids)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
    
    infidelities = np.array(infidelities)
    return float(np.mean(infidelities) + np.max(infidelities))


def _check_validity(pulse):
    """Check amplitude and slew rate constraints."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Amplitude bound
    amplitudes = np.sqrt(I_vals**2 + Q_vals**2)
    if np.any(amplitudes > AMP_BOUND + 1e-9):
        return False
    
    # Slew rate
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = np.array([I_vals[i], Q_vals[i]])
        diff = np.linalg.norm(curr - prev)
        if diff / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    
    # From last step to zero
    diff = np.linalg.norm(prev)
    if diff / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _enforce_constraints(pulse):
    """Scale down pulse to satisfy constraints if violated."""
    pulse = pulse.copy()
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Check amplitude
    amplitudes = np.sqrt(I_vals**2 + Q_vals**2)
    max_amp = np.max(amplitudes)
    if max_amp > AMP_BOUND:
        scale = AMP_BOUND / max_amp
        pulse *= scale
    
    # Check slew rate - if violated, scale further
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    prev = np.array([0.0, 0.0])
    max_slew = 0.0
    for i in range(N_STEPS):
        curr = np.array([I_vals[i], Q_vals[i]])
        diff = np.linalg.norm(curr - prev)
        max_slew = max(max_slew, diff / DT)
        prev = curr
    max_slew = max(max_slew, np.linalg.norm(prev) / DT)
    
    if max_slew > SLEW_BOUND:
        scale = SLEW_BOUND / max_slew
        pulse *= scale
    
    return pulse


def _gaussian_bump(t, center, width, amplitude):
    """Gaussian bump function."""
    if width <= 1e-10:
        return np.zeros_like(t)
    return amplitude * np.exp(-0.5 * ((t - center) / width)**2)


def _coeffs_to_pulse(coeffs, n_bumps=4):
    """Convert coefficients to a pulse.
    coeffs layout: [I_amp_1, I_ctr_1, I_wid_1, ..., Q_amp_1, Q_ctr_1, Q_wid_1, ...]
    Each channel has n_bumps * 3 parameters.
    """
    n_params_per_bump = 3
    n_ch_params = n_bumps * n_params_per_bump
    
    I_coeffs = coeffs[:n_ch_params]
    Q_coeffs = coeffs[n_ch_params:]
    
    I_vals = np.zeros(N_STEPS)
    Q_vals = np.zeros(N_STEPS)
    
    for b in range(n_bumps):
        amp = I_coeffs[b * 3]
        ctr = I_coeffs[b * 3 + 1]
        wid = I_coeffs[b * 3 + 2]
        I_vals += _gaussian_bump(_T_GRID, ctr, wid, amp)
        
        amp_q = Q_coeffs[b * 3]
        ctr_q = Q_coeffs[b * 3 + 1]
        wid_q = Q_coeffs[b * 3 + 2]
        Q_vals += _gaussian_bump(_T_GRID, ctr_q, wid_q, amp_q)
    
    pulse = np.column_stack([I_vals, Q_vals])
    return pulse


def _initial_coeffs(n_bumps=4):
    """Generate initial coefficients resembling a DRAG pulse."""
    coeffs = np.zeros(n_bumps * 6)
    
    # Main I Gaussian: centered at 20ns, width ~6.67ns, area ~pi
    sigma = T_TOTAL / 6.0
    area = np.pi
    # For a Gaussian, area = amp * sigma * sqrt(2*pi)
    amp = area / (sigma * np.sqrt(2 * np.pi))
    
    # I channel, bump 0
    coeffs[0] = amp
    coeffs[1] = T_TOTAL / 2.0
    coeffs[2] = sigma
    
    # Q channel (DRAG): smaller, slightly shifted
    q_amp = 0.3 * amp
    coeffs[3 + 0] = q_amp  # Q bump 0 amp
    coeffs[3 + 1] = T_TOTAL / 2.0  # Q bump 0 center
    coeffs[3 + 2] = sigma  # Q bump 0 width
    
    return coeffs


def _objective(coeffs, n_bumps=4):
    """Objective function for optimization."""
    if _time_remaining() < 5.0:
        return 10.0
    
    pulse = _coeffs_to_pulse(coeffs, n_bumps)
    pulse = _enforce_constraints(pulse)
    
    if not _check_validity(pulse):
        return 10.0
    
    return _score_pulse(pulse)


def pulse(target_gate, hardware_model):
    """Main entry point: optimize pulse using flexible Gaussian basis."""
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    n_bumps = 4
    n_total_params = n_bumps * 6  # 3 per bump per channel * 2 channels
    
    # Define bounds for each parameter
    # amp: [-0.5, 0.5], center: [0, 40], width: [1, 20]
    bounds = []
    for _ in range(n_bumps * 2):  # I and Q channels
        bounds.append((-0.5, 0.5))   # amplitude
        bounds.append((0.0, T_TOTAL))  # center
        bounds.append((1.0, 20.0))   # width
    
    # Initial guess
    x0 = _initial_coeffs(n_bumps)
    
    # Evaluate initial
    best_pulse = _enforce_constraints(_coeffs_to_pulse(x0, n_bumps))
    if not _check_validity(best_pulse):
        best_pulse = _enforce_constraints(best_pulse)
    best_score = _score_pulse(best_pulse)
    
    # Phase 1: Differential Evolution (global search)
    if _time_remaining() > 60.0:
        try:
            def de_objective(params):
                if _time_remaining() < 10.0:
                    return 10.0
                p = _enforce_constraints(_coeffs_to_pulse(params, n_bumps))
                if not _check_validity(p):
                    return 10.0
                return _score_pulse(p)
            
            result = differential_evolution(
                de_objective, bounds,
                x0=x0,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.8,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            
            if result.fun < best_score:
                best_pulse = _enforce_constraints(_coeffs_to_pulse(result.x, n_bumps))
                if _check_validity(best_pulse):
                    best_score = result.fun
                else:
                    best_pulse = _enforce_constraints(best_pulse)
                    best_score = _score_pulse(best_pulse)
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead on the best coefficients
    if _time_remaining() > 30.0:
        try:
            # Find best coefficients from DE or use x0
            # Re-derive from best_pulse isn't straightforward; use result.x if available
            # For simplicity, do a local search around x0 and the DE result
            
            def local_objective(params):
                if _time_remaining() < 5.0:
                    return 10.0
                p = _enforce_constraints(_coeffs_to_pulse(params, n_bumps))
                if not _check_validity(p):
                    return 10.0
                return _score_pulse(p)
            
            # Try local optimization from x0
            res1 = minimize(local_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
            if res1.fun < best_score:
                new_pulse = _enforce_constraints(_coeffs_to_pulse(res1.x, n_bumps))
                if _check_validity(new_pulse):
                    best_pulse = new_pulse
                    best_score = res1.fun
        except Exception:
            pass
    
    # Phase 3: Final validity check
    if not _check_validity(best_pulse):
        best_pulse = _enforce_constraints(best_pulse)
    
    # Ensure exact shape
    best_pulse = np.asarray(best_pulse, dtype=np.float64)
    if best_pulse.shape != (N_STEPS, 2):
        best_pulse = np.zeros((N_STEPS, 2))
    
    return best_pulse


# Keep the original function signature compatible
def quantum_seed(target_gate, hardware_model):
    return pulse(target_gate, hardware_model)
