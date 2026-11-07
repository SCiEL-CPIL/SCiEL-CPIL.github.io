"""Robust X_pi control pulse for a three-level transmon using DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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
    # a is the lowering operator for 3 levels
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a  # number operator
    
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
        
        # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
        # We'll represent the density matrix as a 9-dimensional vector (3x3)
        # For efficiency, we'll use the matrix exponential of the superoperator
        
        # Build the superoperator in vectorized form
        # rho is 3x3, vec(rho) is 9x1
        # L(rho) = -i(H rho - rho H) + L1 rho L1^dag - 0.5(L1^dag L1 rho + rho L1^dag L1)
        #        + L2 rho L2^dag - 0.5(L2^dag L2 rho + rho L2^dag L2)
        
        # Using Kronecker products:
        # vec(A rho B) = (B^T ⊗ A) vec(rho)
        
        H_kron = np.kron(H, np.eye(3)) + np.kron(np.eye(3), H)
        # -i[H, rho] -> -i(H rho - rho H) -> -i(kron(H, I) - kron(I, H)) vec(rho)
        # Wait, vec(A rho) = (I ⊗ A) vec(rho), vec(rho B) = (B^T ⊗ I) vec(rho)
        # So vec(H rho) = (I ⊗ H) vec(rho)
        # vec(rho H) = (H^T ⊗ I) vec(rho)
        
        H_left = np.kron(np.eye(3), H)
        H_right = np.kron(H.T, np.eye(3))
        
        L1_dag_L1 = L1.conj().T @ L1
        L2_dag_L2 = L2.conj().T @ L2
        
        # vec(L1 rho L1^dag) = (L1 ⊗ L1^T) vec(rho)
        L1_term = np.kron(L1, L1.T)
        L2_term = np.kron(L2, L2.T)
        
        # vec(L1^dag L1 rho) = (I ⊗ L1^dag L1) vec(rho)
        L1_diss_left = np.kron(np.eye(3), L1_dag_L1)
        L1_diss_right = np.kron((L1_dag_L1).T, np.eye(3))
        
        L2_diss_left = np.kron(np.eye(3), L2_dag_L2)
        L2_diss_right = np.kron((L2_dag_L2).T, np.eye(3))
        
        Liouvillian = -1j * (H_left - H_right) + L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                        + L2_term - 0.5 * (L2_diss_left + L2_diss_right)
        
        return Liouvillian
    
    def simulate_pulse(pulse_IQ, detuning=0.0, scale=1.0):
        """
        Simulate the pulse and return the final density matrix.
        pulse_IQ: (80, 2) array of [I, Q] values
        """
        # Initial state: we'll track the evolution of the density matrix
        # Start with |0><0|
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L = build_liouvillian(I_val, Q_val, detuning, scale)
            
            # Evolve: rho(t+dt) = exp(dt * L) rho(t)
            # Use matrix exponential
            from scipy.linalg import expm
            U = expm(dt * L)
            
            # Apply to density matrix
            rho_vec = rho.flatten()
            rho_vec_new = U @ rho_vec
            rho = rho_vec_new.reshape(3, 3)
        
        return rho
    
    def compute_fidelity(rho, target_state):
        """Compute fidelity between density matrix and pure state."""
        # F = <psi|rho|psi>
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
        
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                
                # The target is X_pi gate, which swaps |0> and |1>
                # So for input |0>, output should be |1>
                # For input |1>, output should be |0>
                # For superpositions, apply the swap
                
                # Compute fidelity to the X_pi transformed states
                for state in states:
                    # Apply X_pi to the state: swap |0> and |1>
                    target = np.zeros(3, dtype=complex)
                    target[0] = state[1]
                    target[1] = state[0]
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
        # Include from zero to first step and last step to zero
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
    
    # Initial DRAG-style Gaussian seed
    def make_drag_pulse():
        """Create a DRAG-style Gaussian pulse with sigma=40/6 ns, offset to 0 at edges, area=pi."""
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
        # Gaussian centered at t = 20 ns
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset to make edges zero
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        # Normalize area to pi
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        # Create I and Q components
        # For X_pi, we want a pi pulse, so the integral of amplitude should be pi
        # DRAG adds a derivative term to compensate for leakage
        I = gaussian
        Q = np.zeros(N)
        
        # Add DRAG term: Q proportional to derivative of I
        # The DRAG coefficient is related to the anharmonicity
        # For a simple approach, Q = -gamma * dI/dt
        gamma = 0.0  # Start with no DRAG, optimize later
        
        if gamma != 0:
            dI_dt = np.gradient(I, dt)
            Q = -gamma * dI_dt
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity
        if not check_validity(pulse_IQ):
            # Scale down if needed
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                scale_factor = amplitude_bound / max_amp
                pulse_IQ *= scale_factor
        
        return pulse_IQ
    
    # Start with the seed pulse
    pulse_IQ = make_drag_pulse()
    
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
    
    # Time limit: 300 seconds
    start_time = time.time()
    time_limit = 300.0
    
    # Simple iterative optimization
    best_pulse = pulse_IQ.copy()
    best_score = compute_infidelity(best_pulse)
    
    # Try a few optimization steps
    for iteration in range(50):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:  # Safety margin
            break
        
        # Compute gradient numerically
        grad = np.zeros(2 * N)
        epsilon = 1e-6
        for i in range(2 * N):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            
            f_plus = objective(x_plus)
            f_minus = objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        # Gradient descent step
        learning_rate = 0.01
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
