"""Parameterized Gaussian DRAG pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import minimize

# --- Constants from task ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
SCALE_VALUES = np.array([0.97, 1.0, 1.03])
AMP_BOUND = 0.4
SLEW_BOUND = 0.12

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

# Precompute Liouvillian terms that don't depend on H
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

def build_liouvillian_full(H):
    """Build full Liouvillian for given Hamiltonian."""
    coh_L = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh_L + DISS_L

def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3-level Hamiltonian matrix."""
    NN1 = N_MAT @ (N_MAT - I3)
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT)))
    return H

def evolve_all_states(pulse, detuning, scale):
    """Evolve all 6 test states and return mean fidelity."""
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    X3 = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=complex)
    
    exp_mats = []
    for i in range(N_STEPS):
        I_val, Q_val = pulse[i]
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        L = build_liouvillian_full(H)
        exp_mats.append(sla.expm(L * DT))
    
    fidelities = []
    for psi0 in states:
        rho0 = np.outer(psi0, psi0.conj())
        rho_vec = rho0.reshape(-1).copy()
        
        for i in range(N_STEPS):
            rho_vec = exp_mats[i] @ rho_vec
        
        rho = rho_vec.reshape(3, 3)
        psi_target = X3 @ psi0
        f = np.vdot(psi_target, rho @ psi_target).real
        f = np.clip(f, 0, 1)
        fidelities.append(f)
    
    return np.mean(fidelities)

def score_only(pulse):
    """Compute score only (no gradient)."""
    case_fidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            f = evolve_all_states(pulse, detuning, scale)
            case_fidelities.append(f)
    
    case_fidelities = np.array(case_fidelities)
    case_infidelities = 1.0 - case_fidelities
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    score = mean_inf + max_inf
    
    return score

def check_validity(pulse):
    """Check amplitude and slew constraints."""
    for i in range(len(pulse)):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse)):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    diff = np.linalg.norm(prev - np.array([0.0, 0.0])) / DT
    if diff > SLEW_BOUND + 1e-9:
        return False
    return True

def enforce_constraints(pulse):
    """Project pulse onto feasible set."""
    pulse = pulse.copy()
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    # Enforce slew bound forward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Enforce slew bound backward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS - 1, -1, -1):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Re-enforce amplitude
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    return pulse

def generate_pulse_from_params(params):
    """
    Generate a pulse from 5 parameters: [amp, sigma, center, drag_beta, tilt]
    amp: peak amplitude of Gaussian (rad/ns)
    sigma: Gaussian width (ns)
    center: time center (ns)
    drag_beta: DRAG correction coefficient (scales derivative of I)
    tilt: linear tilt for detuning compensation (rad/ns^2)
    """
    amp, sigma, center, drag_beta, tilt = params
    
    # Ensure sigma is not too small to avoid slew violations
    sigma = max(sigma, 3.0)
    
    t = np.arange(N_STEPS) * DT
    
    # Base Gaussian
    g = amp * np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset to ensure zero at edges (simple subtraction of edge value)
    # This is a crude way to ensure it starts/ends near 0, but we'll rely on 
    # the constraint enforcement to strictly enforce it if needed.
    # Actually, for DRAG, we usually want the integral to be pi.
    # Let's compute area and scale if necessary? 
    # The task says "seed is ... area pi". 
    # If we fix amp, we might not hit pi exactly. 
    # However, the optimizer can adjust amp to get the right rotation angle.
    # Let's stick to the raw shape and let amp handle the angle.
    
    # Derivative of Gaussian for DRAG
    # d/dt [ exp(-0.5 ((t-c)/s)^2) ] = exp(...) * (-(t-c)/s^2)
    dg = g * (-(t - center) / (sigma**2))
    
    # Q component is proportional to derivative
    q = drag_beta * dg
    
    # Add tilt to I
    i = g + tilt * (t - center)
    
    pulse = np.column_stack((i, q))
    
    # Enforce constraints strictly
    pulse = enforce_constraints(pulse)
    
    return pulse

def seed_params():
    """Return initial parameter guess."""
    # amp ~ 0.2, sigma ~ 5, center ~ 20, beta ~ 0.1, tilt ~ 0
    return np.array([0.2, 5.0, 20.0, 0.1, 0.0])

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Initial guess
    x0 = seed_params()
    p0 = generate_pulse_from_params(x0)
    best_pulse = p0.copy()
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Objective function for optimizer
    def objective(x):
        p = generate_pulse_from_params(x)
        return score_only(p)
    
    # Bounds for parameters
    # amp: [0, 0.4]
    # sigma: [3, 15]
    # center: [10, 30]
    # drag_beta: [-0.5, 0.5]
    # tilt: [-0.05, 0.05]
    bounds = [
        (0.0, 0.4),
        (3.0, 15.0),
        (10.0, 30.0),
        (-0.5, 0.5),
        (-0.05, 0.05)
    ]
    
    # Use Nelder-Mead for 5D optimization
    # It's robust and doesn't need gradients
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'maxiter': 200,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True,
                'disp': False
            }
        )
        
        if result.fun < best_score:
            opt_pulse = generate_pulse_from_params(result.x)
            opt_score = score_only(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
                
    except Exception as e:
        pass
    
    # If we have time, try a second pass with a slightly different initial guess
    if time.time() < deadline - 60:
        x0_2 = np.array([0.25, 4.0, 20.0, -0.1, 0.0])
        try:
            result2 = minimize(
                objective,
                x0_2,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': 100,
                    'xatol': 1e-4,
                    'fatol': 1e-6,
                    'adaptive': True,
                    'disp': False
                }
            )
            if result2.fun < best_score:
                opt_pulse = generate_pulse_from_params(result2.x)
                opt_score = score_only(opt_pulse)
                if opt_score < best_score:
                    best_pulse = opt_pulse
                    best_score = opt_score
        except Exception:
            pass
    
    return best_pulse
# EVOLVE-BLOCK-END
