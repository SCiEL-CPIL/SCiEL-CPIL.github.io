"""
Robust DRAG pulse for 3-level transmon X-pi gate.
Uses low-dimensional basis expansion (7 params) with Differential Evolution
and Nelder-Mead refinement to find optimal pulse shape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import expm

# Constants from task contract
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2.0 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2.0 * np.pi * 0.002
DT = 0.5  # ns
N_STEPS = 80
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Precompute ladder operator
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_a_dag = _a.conj().T
_n = _a_dag @ _a
_n2 = _n @ _n
_nn1 = _n2 - _n  # n(n-1)
_a_plus_adag = _a + _a_dag
_a_minus_adag = _a - _a_dag

# Time samples
t_samples = (np.arange(N_STEPS) + 0.5) * DT
t_center = TOTAL_TIME / 2.0

# Detuning and scale values for the 9 cases
detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
scales = [0.97, 1.0, 1.03]

# Initial states for fidelity calculation
_states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [1/np.sqrt(2), -1j/np.sqrt(2), 0]
], dtype=complex)

# X gate (target)
X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)

# Basis functions: Hermite-Gaussian with sigma = TOTAL_TIME/6
sigma_base = TOTAL_TIME / 6.0
t_norm = (t_samples - t_center) / sigma_base
g = np.exp(-t_norm**2 / 2.0)
g_norm = np.sqrt(np.sum(g**2))
g = g / g_norm

h1 = t_norm * g
h1_norm = np.sqrt(np.sum(h1**2))
h1 = h1 / h1_norm

h2 = (t_norm**2 - 0.5) * g
h2_norm = np.sqrt(np.sum(h2**2))
h2 = h2 / h2_norm

h3 = (t_norm**3 - 1.5 * t_norm) * g
h3_norm = np.sqrt(np.sum(h3**2))
h3 = h3 / h3_norm


def _make_pulse_from_basis(params):
    """Convert basis coefficients to a valid pulse."""
    A, c1, c2, c3, d1, d2, d3 = params
    
    I_pulse = A * (g + c1 * h1 + c2 * h2 + c3 * h3)
    Q_pulse = A * (d1 * h1 + d2 * h2 + d3 * h3)
    
    pulse_IQ = np.column_stack([I_pulse, Q_pulse])
    
    # Clamp amplitude
    amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    max_amp = np.max(amplitudes)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse_IQ = pulse_IQ * scale_factor
    
    # Check and clamp slew rate
    prev = np.array([0.0, 0.0])
    max_slew = 0.0
    for i in range(N_STEPS):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        slew = diff / DT
        if slew > max_slew:
            max_slew = slew
        prev = pulse_IQ[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    slew = diff / DT
    if slew > max_slew:
        max_slew = slew
    
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        pulse_IQ = pulse_IQ * scale_factor
    
    return pulse_IQ


def _simulate_unitary(pulse_IQ, detuning, scale):
    """Simulate the quantum dynamics and return the final unitary matrix."""
    U_total = np.eye(3, dtype=complex)
    H_static = detuning * _n + 0.5 * ANHARMONICITY * _nn1
    
    for i in range(N_STEPS):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * _a_plus_adag + 1j * Q_val * _a_minus_adag)
        U_step = expm(-1j * H * DT)
        U_total = U_step @ U_total
    
    return U_total


def _compute_score_from_unitary(U):
    """Compute mean + max infidelity from the final unitary."""
    infidelities = []
    for psi in _states:
        phi = X_gate @ psi
        psi_final = U @ psi
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
    
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid


def _evaluate_pulse(pulse_IQ):
    """Evaluate a pulse across all 9 cases."""
    scores = []
    for det in detunings:
        for sc in scales:
            U = _simulate_unitary(pulse_IQ, det, sc)
            score = _compute_score_from_unitary(U)
            scores.append(score)
    
    return np.mean(scores) + np.max(scores)


def _score_params(params):
    """Score a set of basis parameters across all 9 cases."""
    A, c1, c2, c3, d1, d2, d3 = params
    
    # Quick validity check
    if A <= 0 or A > 1.0:
        return 1.0
    if abs(c1) > 2.0 or abs(c2) > 2.0 or abs(c3) > 2.0:
        return 1.0
    if abs(d1) > 2.0 or abs(d2) > 2.0 or abs(d3) > 2.0:
        return 1.0
    
    pulse_IQ = _make_pulse_from_basis(params)
    return _evaluate_pulse(pulse_IQ)


def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I,Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Initial guess: standard DRAG-like pulse
    # Area should be ~pi for X-pi gate
    # A * g has integral ~ A * sigma * sqrt(2*pi) = A * (40/6) * sqrt(2*pi)
    # Want area ~ pi, so A ~ pi / ((40/6) * sqrt(2*pi)) ~ 6.28 / (6.67 * 2.51) ~ 0.377
    # But with basis normalization, A is scaled differently
    # Let's start with A=0.35, small DRAG
    x0 = np.array([0.35, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0])
    
    # Compute initial score
    best_params = x0.copy()
    best_score = _score_params(x0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 100.0:
        # Run Differential Evolution
        bounds = [
            (0.1, 0.6),       # A: amplitude
            (-0.5, 0.5),      # c1: linear shape
            (-0.5, 0.5),      # c2: quadratic shape
            (-0.5, 0.5),      # c3: cubic shape
            (-0.5, 0.5),      # d1: DRAG linear
            (-0.5, 0.5),      # d2: DRAG quadratic
            (-0.5, 0.5),      # d3: DRAG cubic
        ]
        
        try:
            result = differential_evolution(
                _score_params,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                update_interval=1,
            )
            if result.fun < best_score:
                best_params = result.x
                best_score = result.fun
        except Exception:
            pass
    
    # Refine with Nelder-Mead if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 40.0:
        try:
            res = minimize(
                _score_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8},
            )
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Generate final pulse
    pulse_array = _make_pulse_from_basis(best_params)
    
    # Final validity check and clamping
    amp = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.any(amp > AMPLITUDE_BOUND + 1e-9):
        mask = amp > AMPLITUDE_BOUND
        factor = AMPLITUDE_BOUND / amp[mask]
        pulse_array[mask, 0] *= factor
        pulse_array[mask, 1] *= factor
    
    # Slew rate check
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        dI = pulse_array[i, 0] - prev[0]
        dQ = pulse_array[i, 1] - prev[1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            pulse_array[i, 0] = prev[0] + dI * factor
            pulse_array[i, 1] = prev[1] + dQ * factor
        prev = pulse_array[i].copy()
    
    # From last to zero
    dI = 0.0 - pulse_array[-1, 0]
    dQ = 0.0 - pulse_array[-1, 1]
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-9:
        factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
        pulse_array[-1, 0] *= (1 - factor)
        pulse_array[-1, 1] *= (1 - factor)
    
    return pulse_array
# EVOLVE-BLOCK-END
