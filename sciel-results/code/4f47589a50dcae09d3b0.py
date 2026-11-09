"""Direct 160-D pulse optimization using Dual-Annealing for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import dual_annealing

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
MAX_DIFF = SLEW_BOUND * DT  # 0.06

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)
NN1 = N_MAT @ (N_MAT - I3)

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
    
    # Precompute exponentials for this specific case
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

def generate_seed_pulse():
    """Generate the initial DRAG-style Gaussian pulse."""
    amp = 0.2
    sigma = 5.0
    center = 20.0
    drag_beta = 0.1
    tilt = 0.0
    
    t = np.arange(N_STEPS) * DT
    g = amp * np.exp(-0.5 * ((t - center) / sigma)**2)
    dg = g * (-(t - center) / (sigma**2))
    q = drag_beta * dg
    i = g + tilt * (t - center)
    
    pulse = np.column_stack((i, q))
    pulse = enforce_constraints(pulse)
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Initial guess: Analytic DRAG pulse
    x0 = generate_seed_pulse().flatten()
    best_pulse = x0.reshape(80, 2)
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Objective function for Dual-Annealing
    def objective(x):
        p = x.reshape(80, 2)
        p = enforce_constraints(p)
        return score_only(p)
    
    # Bounds for each of the 160 parameters
    # Amplitude bound is 0.4, so I and Q individually can be up to 0.4
    # Slew bound is handled by projection, but we set loose bounds for the optimizer
    bounds = [(-0.4, 0.4) for _ in range(160)]
    
    # Use Dual-Annealing for global optimization in high dimensions
    try:
        result = dual_annealing(
            objective,
            x0,
            bounds=bounds,
            maxiter=500,  # Limit iterations to fit in time
            maxfun=2000,
            seed=42,
            initial_temp=512.0,
            restart_temp_ratio=2e-5,
            visit=2.62,
            accept=1.2,
            progress=True  # Can be set to False if too slow, but helps debugging
        )
        
        opt_pulse = result.x.reshape(80, 2)
        opt_pulse = enforce_constraints(opt_pulse)
        opt_score = score_only(opt_pulse)
        
        if opt_score < best_score:
            best_pulse = opt_pulse
            best_score = opt_score
            
    except Exception as e:
        pass
    
    # If we have time, try a second run with different seed or settings
    if time.time() < deadline - 60:
        try:
            result2 = dual_annealing(
                objective,
                x0,
                bounds=bounds,
                maxiter=300,
                maxfun=1500,
                seed=123,
                initial_temp=1024.0,
                restart_temp_ratio=2e-5,
                visit=2.62,
                accept=1.2
            )
            opt_pulse2 = result2.x.reshape(80, 2)
            opt_pulse2 = enforce_constraints(opt_pulse2)
            opt_score2 = score_only(opt_pulse2)
            
            if opt_score2 < best_score:
                best_pulse = opt_pulse2
                best_score = opt_score2
        except Exception:
            pass
    
    return best_pulse
# EVOLVE-BLOCK-END
