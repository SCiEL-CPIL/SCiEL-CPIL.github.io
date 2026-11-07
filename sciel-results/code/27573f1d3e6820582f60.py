"""Robust X_pi control pulse for a three-level transmon using parameterized DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    dt = 0.5  # ns
    N = 80  # number of steps
    total_time = N * dt  # 40 ns
    
    # Build the 3-level Hamiltonian components
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a  # number operator diag(0, 1, 2)
    I3 = np.eye(3)
    
    # Precompute Lindblad-related terms (constant)
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)
    diss_part = L1_term - 0.5 * (L1_diss_left + L1_diss_right) + L2_term - 0.5 * (L2_diss_left + L2_diss_right)
    
    def build_hamiltonian(I, Q, detuning, scale):
        """Build the Hamiltonian for given I, Q, detuning, scale."""
        H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - I3)
        H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return H
    
    def build_liouvillian(H):
        """Build the Liouvillian superoperator from Hamiltonian."""
        H_left = np.kron(I3, H)
        H_right = np.kron(H.T, I3)
        return -1j * (H_left - H_right) + diss_part
    
    def simulate_pulse(pulse_IQ, detuning=0.0, scale=1.0):
        """
        Simulate the pulse and return the final density matrix.
        pulse_IQ: (80, 2) array of [I, Q] values
        """
        rho_vec = np.zeros(9, dtype=complex)
        rho_vec[0] = 1.0  # |0><0|
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            H = build_hamiltonian(I_val, Q_val, detuning, scale)
            L = build_liouvillian(H)
            
            U = expm(dt * L)
            rho_vec = U @ rho_vec
        
        return rho_vec.reshape(3, 3)
    
    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and pure state."""
        psi = target_state.flatten()
        return np.real(np.conj(psi) @ rho @ psi)
    
    # Test states and targets
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
    ]
    
    # X_pi gate: -i * swap |0> and |1>
    target_states = []
    for state in states:
        target = np.zeros(3, dtype=complex)
        target[0] = -1j * state[1]
        target[1] = -1j * state[0]
        target[2] = state[2]
        target_states.append(target)
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    def compute_infidelity(pulse_IQ):
        """Compute mean + max infidelity over the 9 test cases."""
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                
                for state, target in zip(states, target_states):
                    fid = compute_fidelity(rho_final, target)
                    infid = 1.0 - fid
                    infidelities.append(infid)
        
        infidelities = np.array(infid)
        return np.mean(infid) + np.max(infid)
    
    def check_validity(pulse_IQ):
        """Check if the pulse satisfies all constraints."""
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / dt > slew_bound + 1e-12:
                return False
            prev = curr
        
        diff = np.linalg.norm(prev)
        if diff / dt > slew_bound + 1e-12:
            return False
        
        return True
    
    def make_pulse(params):
        """
        Create a parameterized DRAG pulse.
        params: [sigma, alpha, beta, amp_scale]
        sigma: Gaussian width (ns)
        alpha: DRAG coefficient
        beta: detuning compensation
        amp_scale: amplitude scaling
        """
        sigma, alpha, beta, amp_scale = params
        t = np.arange(N) * dt
        center = total_time / 2.0
        
        # Gaussian envelope
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset to make edges zero
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        # Normalize area to pi * amp_scale
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-15:
            gaussian *= (np.pi * amp_scale) / area
        
        I = gaussian.copy()
        
        # DRAG Q component
        dI_dt = np.gradient(I, dt)
        Q = alpha * dI_dt
        
        # Add detuning compensation to I
        I += beta * gaussian
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity by scaling down if needed
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        max_amp = np.max(amplitudes)
        if max_amp > amplitude_bound:
            scale_factor = amplitude_bound / max_amp
            pulse_IQ *= scale_factor
        
        # Check slew rate
        if not check_validity(pulse_IQ):
            # Further scale down for slew rate
            for _ in range(10):
                if check_validity(pulse_IQ):
                    break
                pulse_IQ *= 0.95
        
        return pulse_IQ
    
    # Optimization over 4 parameters
    def objective(params):
        """Objective function for optimization."""
        sigma, alpha, beta, amp_scale = params
        
        # Bounds
        if sigma < 1.0 or sigma > 20.0:
            return 1e10
        if alpha < -2.0 or alpha > 2.0:
            return 1e10
        if beta < -0.1 or beta > 0.1:
            return 1e10
        if amp_scale < 0.5 or amp_scale > 1.5:
            return 1e10
        
        pulse_IQ = make_pulse(params)
        
        if not check_validity(pulse_IQ):
            return 1e10
        
        return compute_infidelity(pulse_IQ)
    
    # Time management
    start_time = time.time()
    time_limit = 280.0  # Safety margin before 300s kill
    
    # Grid search over key parameters
    best_score = 1e10
    best_params = [6.67, 0.0, 0.0, 1.0]  # Default: sigma=40/6, no DRAG
    
    # Coarse grid
    sigma_values = [4.0, 5.0, 6.0, 6.67, 7.0, 8.0, 10.0]
    alpha_values = [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    beta_values = [-0.05, 0.0, 0.05]
    amp_scale_values = [0.95, 1.0, 1.05]
    
    for sigma in sigma_values:
        for alpha in alpha_values:
            for beta in beta_values:
                for amp_scale in amp_scale_values:
                    elapsed = time.time() - start_time
                    if elapsed > time_limit - 20:
                        break
                    
                    params = [sigma, alpha, beta, amp_scale]
                    score = objective(params)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
        
        if time.time() - start_time > time_limit - 20:
            break
    
    # Fine-tune around best
    if time.time() - start_time < time_limit - 30:
        sigma_best, alpha_best, beta_best, amp_best = best_params
        
        for ds in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for da in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                for db in [-0.02, 0.0, 0.02]:
                    elapsed = time.time() - start_time
                    if elapsed > time_limit - 15:
                        break
                    
                    params = [sigma_best + ds, alpha_best + da, beta_best + db, amp_best]
                    if params[0] < 1.0 or params[0] > 20.0:
                        continue
                    if params[1] < -2.0 or params[1] > 2.0:
                        continue
                    if params[2] < -0.1 or params[2] > 0.1:
                        continue
                    
                    score = objective(params)
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
            
            if time.time() - start_time > time_limit - 15:
                break
    
    # Generate final pulse
    best_pulse = make_pulse(best_params)
    
    # Final validity check
    if not check_validity(best_pulse):
        # Fallback to seed
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        best_pulse = np.column_stack([gaussian, np.zeros(N)])
    
    return best_pulse
# EVOLVE-BLOCK-END
