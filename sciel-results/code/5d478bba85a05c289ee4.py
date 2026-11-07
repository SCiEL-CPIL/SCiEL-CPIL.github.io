"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T_PHI = 40000.0

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns


def build_operators():
    """Build a, a_dagger, n for 3-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n


a_op, a_dag_op, n_op = build_operators()


def hamiltonian(detuning, scale, I, Q):
    """Build Hamiltonian for given parameters."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    H += 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    return H


def lindblad_operators():
    """Build Lindblad operators."""
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T_PHI)
    return [L1, L2]


L_ops = lindblad_operators()


def liouvillian_step(detuning, scale, I, Q):
    """
    Compute the superoperator for one step of evolution.
    Returns the matrix that maps density matrix (vectorized) to new density matrix.
    """
    H = hamiltonian(detuning, scale, I, Q)
    U = expm(-1j * H * DT)
    
    # Build the superoperator: rho -> U rho U^dag + sum L rho L^dag - 0.5{L^dag L, rho}
    # Vectorize: vec(rho) -> (U^T ⊗ U) vec(rho) + ...
    # For Lindblad: L rho L^dag -> (L^T ⊗ L) vec(rho)
    # Anti-commutator: 0.5 (L^dag L rho + rho L^dag L) -> 0.5 (I ⊗ L^dag L + (L^dag L)^T ⊗ I) vec(rho)
    
    dim = 3
    dim2 = dim * dim
    
    # Unitary part
    S_u = np.kron(U.T, U)
    
    # Lindblad part
    S_l = np.zeros((dim2, dim2), dtype=complex)
    for L in L_ops:
        Ldag = L.conj().T
        # L rho L^dag
        S_l += np.kron(L.T, L)
        # -0.5 {L^dag L, rho}
        LdagL = Ldag @ L
        S_l -= 0.5 * (np.kron(np.eye(dim), LdagL) + np.kron(LdagL.T, np.eye(dim)))
    
    # Scale Lindblad terms by DT (since they are rates)
    S_total = S_u + DT * S_l
    
    return S_total


def evolve_pulse(pulse, detuning, scale):
    """
    Evolve the state under the given pulse.
    pulse: (80, 2) array of (I, Q)
    Returns final density matrix.
    """
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        S = liouvillian_step(detuning, scale, I, Q)
        rho_vec = rho.flatten()
        rho_vec = S @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho


def fidelity_with_X(rho):
    """
    Compute fidelity with X gate applied to |0>.
    X|0> = |1>, so we want rho to be close to |1><1|.
    """
    target = np.zeros((3, 3), dtype=complex)
    target[1, 1] = 1.0
    return np.real(np.trace(rho @ target))


def compute_score(pulse):
    """
    Compute the score for a given pulse.
    Returns (mean_infidelity, max_infidelity).
    """
    fidelities = []
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    for det in detunings:
        for sc in scales:
            rho = evolve_pulse(pulse, det, sc)
            fid = fidelity_with_X(rho)
            infidelity = 1.0 - fid
            fidelities.append(infidelity)
    
    fidelities = np.array(fidelities)
    return np.mean(fidelities), np.max(fidelities)


def check_validity(pulse):
    """Check if pulse satisfies amplitude and slew constraints."""
    # Amplitude bound
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        if np.sqrt(I**2 + Q**2) > AMPLITUDE_BOUND + 1e-9:
            return False
    
    # Slew bound: from zero into step 1
    I0 = 0.0
    Q0 = 0.0
    I1 = pulse[0, 0]
    Q1 = pulse[0, 1]
    if np.sqrt((I1 - I0)**2 + (Q1 - Q0)**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        dI = pulse[i + 1, 0] - pulse[i, 0]
        dQ = pulse[i + 1, 1] - pulse[i, 1]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    I_last = pulse[-1, 0]
    Q_last = pulse[-1, 1]
    if np.sqrt(I_last**2 + Q_last**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def drag_gaussian_pulse():
    """
    Generate a DRAG-style Gaussian pulse.
    sigma = 40/6 ns, offset to 0 at edges, area = pi.
    """
    t = np.linspace(0, TOTAL_TIME, N_STEPS)
    sigma = TOTAL_TIME / 6.0
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - TOTAL_TIME / 2) / sigma) ** 2)
    
    # Normalize area to pi
    area = np.sum(gauss) * DT
    gauss = gauss * (np.pi / area)
    
    # DRAG: add a derivative term to suppress leakage
    # d/dt of Gaussian
    gauss_deriv = -((t - TOTAL_TIME / 2) / sigma**2) * gauss
    
    # DRAG parameter: typically beta = -2 * delta * sigma / (pi) or similar
    # For anharmonicity alpha, the DRAG term is proportional to alpha * sigma
    # Let's use a standard DRAG form: I = gauss, Q = beta * gauss_deriv
    # beta is chosen to cancel leakage
    
    # The optimal beta for DRAG is approximately: beta = -2 * delta * sigma / (pi * alpha)
    # But since we're at resonance (delta=0 for nominal), the DRAG term helps with
    # off-resonant cases. A common choice is beta ~ alpha * sigma
    
    # Let's try beta = ANHARMONICITY * sigma / (2 * np.pi)
    # Actually, the standard DRAG correction is:
    # I(t) = A(t)
    # Q(t) = -beta * A'(t)
    # where beta = 2 * delta * sigma / (pi * alpha) for detuning delta
    # At resonance, we want to correct for the anharmonicity-induced leakage.
    # A simpler approach: use a small Q component proportional to the derivative
    
    beta = 0.1  # Start with a small DRAG parameter
    
    I = gauss
    Q = beta * gauss_deriv
    
    pulse = np.column_stack([I, Q])
    
    # Ensure validity
    if not check_validity(pulse):
        # Scale down to satisfy constraints
        max_amp = np.max(np.sqrt(I**2 + Q**2))
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp
            pulse *= scale
        
        # Check slew
        for i in range(N_STEPS - 1):
            dI = pulse[i + 1, 0] - pulse[i, 0]
            dQ = pulse[i + 1, 1] - pulse[i, 1]
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                scale = SLEW_BOUND / slew
                pulse *= scale
    
    return pulse


def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes for the X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    # Start with DRAG Gaussian pulse
    p = drag_gaussian_pulse()
    
    # Simple local optimization to improve robustness
    # Try a few variations of the DRAG parameter
    
    best_pulse = p
    best_score = np.inf
    
    for beta in [0.0, 0.05, 0.1, 0.15, 0.2]:
        t = np.linspace(0, TOTAL_TIME, N_STEPS)
        sigma = TOTAL_TIME / 6.0
        gauss = np.exp(-0.5 * ((t - TOTAL_TIME / 2) / sigma) ** 2)
        area = np.sum(gauss) * DT
        gauss = gauss * (np.pi / area)
        gauss_deriv = -((t - TOTAL_TIME / 2) / sigma**2) * gauss
        
        I = gauss
        Q = beta * gauss_deriv
        
        p_trial = np.column_stack([I, Q])
        
        # Check and fix validity
        if not check_validity(p_trial):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > AMPLITUDE_BOUND:
                scale = AMPLITUDE_BOUND / max_amp
                p_trial *= scale
            for i in range(N_STEPS - 1):
                dI = p_trial[i + 1, 0] - p_trial[i, 0]
                dQ = p_trial[i + 1, 1] - p_trial[i, 1]
                slew = np.sqrt(dI**2 + dQ**2) / DT
                if slew > SLEW_BOUND:
                    scale = SLEW_BOUND / slew
                    p_trial *= scale
        
        if not check_validity(p_trial):
            continue
        
        try:
            mean_inf, max_inf = compute_score(p_trial)
            score = mean_inf + max_inf
            if score < best_score:
                best_score = score
                best_pulse = p_trial
        except:
            continue
    
    return best_pulse
# EVOLVE-BLOCK-END
