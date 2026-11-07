"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns


def _build_matrices():
    """Build 3x3 matrices for a, adag, n."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n


def _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, adag, n):
    """Build the Liouvillian superoperator for one step."""
    I3 = np.eye(3, dtype=complex)
    
    # Hamiltonian
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - I3))
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    
    # Hamiltonian superoperator: vec(-i[H, rho]) = -i*(H ⊗ I - I ⊗ H^T) vec(rho)
    H_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    Ldag1 = L1.conj().T
    Ldag2 = L2.conj().T
    Ldag1_L1 = Ldag1 @ L1
    Ldag2_L2 = Ldag2 @ L2
    
    # Dissipation superoperators
    L1_super = np.kron(L1, L1.conj())
    L1_diss_super = -0.5 * (np.kron(Ldag1_L1, I3) + np.kron(I3, Ldag1_L1.T))
    
    L2_super = np.kron(L2, L2.conj())
    L2_diss_super = -0.5 * (np.kron(Ldag2_L2, I3) + np.kron(I3, Ldag2_L2.T))
    
    return H_super + L1_super + L1_diss_super + L2_super + L2_diss_super


def _simulate_pulse(IQ, detuning, scale, a, adag, n):
    """
    Simulate the evolution of a pulse for given detuning and scale.
    Returns the mean fidelity to X gate.
    """
    I3 = np.eye(3, dtype=complex)
    
    # Target gate: X_pi (X gate up to global phase)
    X_gate = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    # Precompute superoperators for each step
    supers = []
    for step in range(N_STEPS):
        I_val = IQ[step, 0]
        Q_val = IQ[step, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, adag, n)
        supers.append(expm(L_super * DT))
    
    fidelities = []
    for psi0 in states:
        rho = np.outer(psi0, psi0.conj())
        rho_vec = rho.flatten().copy()
        
        for step in range(N_STEPS):
            rho_vec = supers[step] @ rho_vec
        
        rho_final = rho_vec.reshape(3, 3)
        psi_target = X_gate @ psi0
        rho_target = np.outer(psi_target, psi_target.conj())
        fidelity = np.real(np.trace(rho_target @ rho_final))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)


def _compute_score(IQ):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    a, adag, n = _build_matrices()
    
    all_infidelities = []
    for det in detunings:
        for sc in scales:
            fid = _simulate_pulse(IQ, det, sc, a, adag, n)
            all_infidelities.append(1.0 - fid)
    
    infidelities = np.array(all_infidelities)
    return np.mean(infidelities) + np.max(infidelities)


def _check_validity(IQ):
    """Check if the pulse satisfies the constraints."""
    amps = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    # Slew rate bound
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        diff = np.linalg.norm(IQ[i] - prev) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
        prev = IQ[i]
    
    # From last step to zero
    diff = np.linalg.norm(np.array([0.0, 0.0]) - IQ[-1]) / DT
    if diff > SLEW_BOUND + 1e-12:
        return False
    
    return True


def _make_pulse(sigma_factor, drag_coeff, amplitude_scale=1.0):
    """Generate a DRAG-style Gaussian pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = (40.0 / 6.0) * sigma_factor
    
    env = np.exp(-(t - TOTAL_TIME / 2)**2 / (2 * sigma**2))
    # Offset to 0 at edges
    env = env - env[0] - env[-1]
    env = np.maximum(env, 0)
    
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    
    env *= amplitude_scale
    I = env
    Q = drag_coeff * np.gradient(I, DT)
    
    IQ = np.column_stack([I, Q])
    
    # Enforce validity
    if not _check_validity(IQ):
        # Scale down amplitude
        max_amp = np.max(np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            IQ *= (AMPLITUDE_BOUND / max_amp) * 0.95
        
        # Enforce slew rate by clipping
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            diff = np.linalg.norm(IQ[i] - prev)
            max_diff = SLEW_BOUND * DT * 0.95
            if diff > max_diff:
                scale = max_diff / diff
                IQ[i] = prev + scale * (IQ[i] - prev)
            prev = IQ[i]
        
        # Check final slew
        diff = np.linalg.norm(IQ[-1])
        max_diff = SLEW_BOUND * DT * 0.95
        if diff > max_diff:
            IQ[-1] *= max_diff / diff
    
    return IQ


def _optimize_pulse():
    """Optimize the pulse parameters."""
    start_time = time.time()
    deadline = 320.0  # Safety margin
    
    best_IQ = None
    best_score = np.inf
    
    # Grid search over parameters
    sigma_factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    drag_coeffs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    amp_scales = [0.8, 0.9, 1.0, 1.1]
    
    for sf in sigma_factors:
        for dc in drag_coeffs:
            for ascale in amp_scales:
                if time.time() > deadline:
                    return best_IQ, best_score
                
                IQ = _make_pulse(sf, dc, ascale)
                if not _check_validity(IQ):
                    continue
                
                score = _compute_score(IQ)
                if score < best_score:
                    best_score = score
                    best_IQ = IQ.copy()
    
    # Local refinement around best
    if best_IQ is not None:
        # Find which parameters gave the best
        # Re-search with finer grid around best region
        for sf in np.linspace(0.7, 1.5, 9):
            for dc in np.linspace(-0.6, 0.6, 13):
                for ascale in np.linspace(0.7, 1.2, 6):
                    if time.time() > deadline:
                        break
                    IQ = _make_pulse(sf, dc, ascale)
                    if not _check_validity(IQ):
                        continue
                    score = _compute_score(IQ)
                    if score < best_score:
                        best_score = score
                        best_IQ = IQ.copy()
    
    return best_IQ, best_score


def pulse(target_gate, hardware_model):
    """
    Generate a pulse for the given target gate and hardware model.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    
    # First, get a good seed
    IQ_seed = _make_pulse(1.0, -0.2, 1.0)
    
    # Optimize
    best_IQ, best_score = _optimize_pulse()
    
    if best_IQ is None:
        best_IQ = IQ_seed
    
    # Final validity check
    if not _check_validity(best_IQ):
        best_IQ = IQ_seed
        if not _check_validity(best_IQ):
            # Last resort: zero pulse
            best_IQ = np.zeros((N_STEPS, 2))
    
    return best_IQ

# EVOLVE-BLOCK-END
