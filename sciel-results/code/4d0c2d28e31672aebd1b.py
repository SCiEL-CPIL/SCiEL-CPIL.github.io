"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

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

def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    # a: lowering operator for 3-level transmon
    # |0> = [1,0,0], |1> = [0,1,0], |2> = [0,0,1]
    # a|1> = |0>, a|2> = sqrt(2)|1>
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a  # diag(0, 1, 2)
    return a, a_dag, n

def _liouvillian(I, Q, detuning, scale, a, a_dag, n):
    """Build the Liouvillian superoperator for one step."""
    # Hamiltonian: H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)  # T1 relaxation
    L2 = n * np.sqrt(2.0 / TPHI)  # Tphi dephasing
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # We need to exponentiate this superoperator.
    # Represent rho as a 9-dim vector (column-stacking of 3x3 matrix).
    # The Liouvillian superoperator in this basis:
    # -i(H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 0.5(L_k^dag L_k ⊗ I + I ⊗ (L_k^dag L_k)^T))
    
    I3 = np.eye(3, dtype=complex)
    
    # -i[H, rho] term: -i(H ⊗ I - I ⊗ H^T)
    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Lindblad terms
    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))
    
    return L_total

def _simulate_pulse(pulse, detuning, scale, a, a_dag, n):
    """Simulate the pulse and return the process matrix (or unitary for coherent part)."""
    # Start with identity as the initial process
    # We'll track the evolution of the density matrix for each input state
    # Actually, for fidelity calculation, we can track the unitary evolution of basis states
    # But with decoherence, we need the full process matrix.
    
    # Let's use the superoperator approach: evolve the identity density matrix
    # Actually, we need to compute the process matrix: for each input |j>, evolve rho_j = |j><j|
    # and get the output states. Then fidelity = <psi| U |psi> for pure states.
    
    # With decoherence, we track the density matrix evolution.
    # Fidelity for pure state |psi> input: F = <psi| rho_out |psi>
    
    L_super = np.zeros((9, 9), dtype=complex)
    
    # We'll evolve each basis state's density matrix
    fidelities = []
    
    # Input states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # Wait, the task says: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # That's 1 + 1 + 2 + 2 = 6 states? Let me re-read.
    # "fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # So 6 states total.
    
    input_states = []
    # |0>
    input_states.append(np.array([1, 0, 0], dtype=complex))
    # |1>
    input_states.append(np.array([0, 1, 0], dtype=complex))
    # (|0> + |1>)/sqrt(2)
    input_states.append(np.array([1, 1, 0], dtype=complex) / np.sqrt(2))
    # (|0> - |1>)/sqrt(2)
    input_states.append(np.array([1, -1, 0], dtype=complex) / np.sqrt(2))
    # (|0> + i|1>)/sqrt(2)
    input_states.append(np.array([1, 1j, 0], dtype=complex) / np.sqrt(2))
    # (|0> - i|1>)/sqrt(2)
    input_states.append(np.array([1, -1j, 0], dtype=complex) / np.sqrt(2))
    
    # Target: X gate maps |0> -> |1>, |1> -> |0>
    # So for input |0>, target is |1>; for input |1>, target is |0>
    # For superpositions, X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = same state
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2), fidelity is |<target|output>|^2
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2), global phase doesn't matter for fidelity
    # Actually fidelity is |<psi_target|psi_output>|^2
    
    # Let's just compute the evolved state and compare to X|psi_in>
    
    for psi_in in input_states:
        # Target state after X gate
        psi_target = np.zeros(3, dtype=complex)
        psi_target[0] = psi_in[1]  # X|0> = |1>, so component 0 of target = component 1 of input
        psi_target[1] = psi_in[0]  # X|1> = |0>, so component 1 of target = component 0 of input
        # Component 2 stays 0 since X doesn't affect |2> and input has no |2> component
        
        # Evolve the density matrix rho = |psi_in><psi_in|
        rho = np.outer(psi_in, psi_in.conj())
        
        for i in range(N_STEPS):
            I_val = pulse[i, 0]
            Q_val = pulse[i, 1]
            
            L_super = _liouvillian(I_val, Q_val, detuning, scale, a, a_dag, n)
            # Exponentiate the superoperator for time DT
            U_super = expm(L_super * DT)
            
            # Vectorize rho (column-stacking)
            rho_vec = rho.flatten(order='F')
            rho_vec = U_super @ rho_vec
            rho = rho_vec.reshape((3, 3), order='F')
        
        # Fidelity: |<psi_target|rho|psi_target>|
        # Actually for pure state input with decoherence, fidelity is Tr(|psi_target><psi_target| rho)
        # = <psi_target| rho |psi_target>
        fidelity = np.real(np.conj(psi_target) @ rho @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero into step 1 and from last step to zero
    # p[0] - 0
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
    
    # Last step to zero
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False
    
    return True

def _compute_score(pulse, a, a_dag, n):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _generate_drag_pulse(beta=0.0):
    """Generate a DRAG-style pulse."""
    # Gaussian pulse with sigma = 40/6 ns, area = pi
    # 80 steps of dt = 0.5 ns, total 40 ns
    t = np.arange(N_STEPS) * DT  # 0 to 39.5 ns
    t_center = 20.0  # center of the 40 ns window
    
    sigma = 40.0 / 6.0
    # Gaussian amplitude
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to 0 at edges: subtract the value at the edges
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Normalize area to pi
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    # DRAG: add a derivative term
    # d(gauss)/dt
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = gauss
    Q = beta * dgauss
    
    # Check and clip to bounds
    pulse = np.column_stack([I, Q])
    
    # Ensure amplitude constraint
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    for i in range(N_STEPS):
        if amps[i] > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / amps[i]
            pulse[i] *= scale_factor
    
    # Ensure slew constraint - this is tricky, let's just check
    # For now, return the pulse
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second limit
    
    a, a_dag, n = _build_matrices()
    
    # Start with DRAG seed
    best_pulse = _generate_drag_pulse(beta=0.0)
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Try different beta values for DRAG
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4]
    
    for beta in betas:
        if time.time() > deadline - 60.0:
            break
        
        p = _generate_drag_pulse(beta=beta)
        if not _check_validity(p):
            continue
        
        score = _compute_score(p, a, a_dag, n)
        if score < best_score:
            best_score = score
            best_pulse = p.copy()
    
    # Simple local optimization: try small perturbations
    # Use a few random perturbations
    rng = np.random.RandomState(42)
    
    for iteration in range(200):
        if time.time() > deadline - 60.0:
            break
        
        # Generate a perturbation
        perturbation = rng.normal(0, 0.01, size=(N_STEPS, 2))
        
        # Apply perturbation
        new_pulse = best_pulse + perturbation
        
        # Check validity
        if not _check_validity(new_pulse):
            continue
        
        score = _compute_score(new_pulse, a, a_dag, n)
        if score < best_score:
            best_score = score
            best_pulse = new_pulse.copy()
    
    return best_pulse
# EVOLVE-BLOCK-END
