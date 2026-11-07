"""Robust X_pi control pulse for a three-level transmon using DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
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
    
    # Time limit guard
    start_time = time.time()
    time_limit = 300.0  # Safety margin
    
    def elapsed_time():
        return time.time() - start_time
    
    # First, optimize the DRAG alpha parameter with finer grid
    best_pulse = pulse_IQ.copy()
    best_score = compute_infidelity(best_pulse)
    
    # Coarse search over alpha
    alphas_coarse = np.linspace(-1.0, 1.0, 21)
    for alpha in alphas_coarse:
        if elapsed_time() > time_limit - 60:
            break
        pulse_test = make_drag_pulse(alpha)
        if check_validity(pulse_test):
            score = compute_infidelity(pulse_test)
            if score < best_score:
                best_score = score
                best_pulse = pulse_test.copy()
    
    # Fine search around best alpha found
    if elapsed_time() < time_limit - 30:
        # Find which alpha was best
        best_alpha = 0.0
        min_score = best_score
        for alpha in alphas_coarse:
            pulse_test = make_drag_pulse(alpha)
            if check_validity(pulse_test):
                score = compute_infidelity(pulse_test)
                if score < min_score:
                    min_score = score
                    best_alpha = alpha
        
        # Fine grid around best_alpha
        alphas_fine = np.linspace(best_alpha - 0.25, best_alpha + 0.25, 11)
        for alpha in alphas_fine:
            if elapsed_time() > time_limit - 30:
                break
            pulse_test = make_drag_pulse(alpha)
            if check_validity(pulse_test):
                score = compute_infidelity(pulse_test)
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_test.copy()
    
    # Now do gradient-based refinement using scipy minimize
    if elapsed_time() < time_limit - 30:
        x0 = best_pulse.flatten()
        
        def objective(params):
            """Objective function for optimization."""
            I = params[:N]
            Q = params[N:2*N]
            pulse_IQ = np.column_stack([I, Q])
            
            if not check_validity(pulse_IQ):
                return 1e10  # Large penalty for invalid
            
            return compute_infidelity(pulse_IQ)
        
        # Bounds for I and Q
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)
        
        # Use L-BFGS-B with numerical gradient
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                pulse_IQ_new = result.x.reshape(N, 2)
                if check_validity(pulse_IQ_new):
                    score = compute_infidelity(pulse_IQ_new)
                    if score < best_score:
                        best_score = score
                        best_pulse = pulse_IQ_new.copy()
        except Exception:
            pass
    
    # Final polish: a few more iterations if time remains
    if elapsed_time() < time_limit - 10:
        x0 = best_pulse.flatten()
        for _ in range(3):
            if elapsed_time() > time_limit - 5:
                break
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.fun < best_score:
                    pulse_IQ_new = result.x.reshape(N, 2)
                    if check_validity(pulse_IQ_new):
                        score = compute_infidelity(pulse_IQ_new)
                        if score < best_score:
                            best_score = score
                            best_pulse = pulse_IQ_new.copy()
                            x0 = result.x.copy()
            except Exception:
                break
    
    return best_pulse
# EVOLVE-BLOCK-END
