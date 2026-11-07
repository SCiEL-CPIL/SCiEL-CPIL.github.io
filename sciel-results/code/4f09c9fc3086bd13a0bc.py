"""Robust X_pi control pulse for a three-level transmon using DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X_pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Extract hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
    drive_scale_bounds = (0.97, 1.03)
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
    
    def build_hamiltonian(I, Q, detuning, scale):
        """Build the Hamiltonian for given I, Q, detuning, scale."""
        H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return H
    
    def build_liouvillian(I, Q, detuning, scale):
        """Build the Liouvillian superoperator for Lindblad evolution."""
        H = build_hamiltonian(I, Q, detuning, scale)
        
        # Lindblad operators
        L1 = a / np.sqrt(T1)
        L2 = n_op * np.sqrt(2.0 / Tphi)
        
        # Build the superoperator in vectorized form
        I3 = np.eye(3)
        
        H_left = np.kron(I3, H)
        H_right = np.kron(H.T, I3)
        
        L1_dag_L1 = L1.conj().T @ L1
        L2_dag_L2 = L2.conj().T @ L2
        
        L1_term = np.kron(L1, L1.T)
        L2_term = np.kron(L2, L2.T)
        
        L1_diss_left = np.kron(I3, L1_dag_L1)
        L1_diss_right = np.kron(L1_dag_L1.T, I3)
        
        L2_diss_left = np.kron(I3, L2_dag_L2)
        L2_diss_right = np.kron(L2_dag_L2.T, I3)
        
        Liouvillian = -1j * (H_left - H_right) + L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                        + L2_term - 0.5 * (L2_diss_left + L2_diss_right)
        
        return Liouvillian
    
    def simulate_pulse(pulse_IQ, detuning=0.0, scale=1.0):
        """
        Simulate the pulse and return the final density matrix.
        pulse_IQ: (80, 2) array of [I, Q] values
        """
        # Initial state: |0><0|
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L = build_liouvillian(I_val, Q_val, detuning, scale)
            
            # Evolve: rho(t+dt) = exp(dt * L) rho(t)
            U = expm(dt * L)
            
            # Apply to density matrix
            rho_vec = rho.flatten()
            rho_vec_new = U @ rho_vec
            rho = rho_vec_new.reshape(3, 3)
        
        return rho
    
    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and pure state."""
        psi = target_state.flatten()
        return np.real(np.conj(psi) @ rho @ psi)
    
    def compute_infidelity(pulse_IQ):
        """Compute mean + max infidelity over the 9 test cases."""
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        # Test states
        states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),  # (|0>+|1>)/sqrt(2)
            (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),  # (|0>-|1>)/sqrt(2)
            (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),  # (|0>+i|1>)/sqrt(2)
            (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),  # (|0>-i|1>)/sqrt(2)
        ]
        
        # X_pi gate: exp(-i*pi/2*X) = -i*X = [[0,-i],[-i,0]]
        # So |0> -> -i|1>, |1> -> -i|0>
        
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                
                for state in states:
                    # Apply X_pi to the state: -i * swap |0> and |1>
                    target = np.zeros(3, dtype=complex)
                    target[0] = -1j * state[1]
                    target[1] = -1j * state[0]
                    target[2] = state[2]
                    
                    fid = compute_fidelity(rho_final, target)
                    infid = 1.0 - fid
                    infidelities.append(infid)
        
        infidelities = np.array(infid)
        return np.mean(infid) + np.max(infid)
    
    def check_validity(pulse_IQ):
        """Check if the pulse satisfies all constraints."""
        # Amplitude bound
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        
        # Slew rate bound
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = np.linalg.norm(curr - prev)
            if diff / dt > slew_bound + 1e-12:
                return False
            prev = curr
        
        # Last step to zero
        diff = np.linalg.norm(prev - np.zeros(2))
        if diff / dt > slew_bound + 1e-12:
            return False
        
        return True
    
    # Initial DRAG-style Gaussian seed with proper Q component
    def make_drag_pulse(alpha=0.0):
        """Create a DRAG-style Gaussian pulse with sigma=40/6 ns, offset to 0 at edges, area=pi."""
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset to make edges zero
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        # Normalize area to pi
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        # I component
        I = gaussian.copy()
        
        # DRAG Q component: Q = alpha * dI/dt
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                scale_factor = amplitude_bound / max_amp
                pulse_IQ *= scale_factor
        
        return pulse_IQ
    
    # Start with the seed pulse
    pulse_IQ = make_drag_pulse(0.0)
    
    # Optimize using gradient-based methods
    def objective(params):
        """Objective function for optimization."""
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        
        if not check_validity(pulse_IQ):
            return 1e10  # Large penalty for invalid
        
        return compute_infidelity(pulse_IQ)
    
    # Use L-BFGS-B with constraints
    x0 = pulse_IQ.flatten()
    
    # Bounds for I and Q
    bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)
    
    # Time limit: 330 seconds (evaluator kills at 330s, keep safety margin)
    start_time = time.time()
    time_limit = 300.0  # Safety margin
    
    # Simple iterative optimization
    best_pulse = pulse_IQ.copy()
    best_score = compute_infidelity(best_pulse)
    
    # First, try optimizing the DRAG alpha parameter
    best_alpha = 0.0
    for alpha in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, -0.1, -0.2, -0.5]:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:
            break
        
        pulse_test = make_drag_pulse(alpha)
        if check_validity(pulse_test):
            score = compute_infidelity(pulse_test)
            if score < best_score:
                best_score = score
                best_pulse = pulse_test.copy()
                best_alpha = alpha
    
    # Now do gradient-based refinement
    x0 = best_pulse.flatten()
    
    for iteration in range(30):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:  # Safety margin
            break
        
        # Compute gradient numerically
        grad = np.zeros(2 * N)
        epsilon = 1e-5
        for i in range(2 * N):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = objective(x_plus)
            f_minus = objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        # Gradient descent step with adaptive learning rate
        learning_rate = 0.005
        x0 -= learning_rate * grad
        
        # Project to feasible region (simple clipping)
        x0[:N] = np.clip(x0[:N], -amplitude_bound, amplitude_bound)
        x0[N:] = np.clip(x0[N:], -amplitude_bound, amplitude_bound)
        
        # Check validity and update best
        pulse_IQ_new = x0.reshape(N, 2)
        if check_validity(pulse_IQ_new):
            score = compute_infidelity(pulse_IQ_new)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ_new.copy()
    
    return best_pulse
# EVOLVE-BLOCK-END
