import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0  # Safety margin before 330s limit

    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0

    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # Basis states for fidelity calculation
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]

    # Target states for X gate (bit flip)
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2)
    ]

    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_op.conj().T @ a_op
    I3 = np.eye(3, dtype=complex)

    # Precompute drive matrices
    M_I = 0.5 * (a_op + a_dag)
    M_Q = 0.5 * 1j * (a_op - a_dag)

    # Precompute Lindblad superoperator (constant part)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    lindblad_super = np.zeros((9, 9), dtype=complex)
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        term1 = np.kron(Lk, Lk_dag.T)
        Lk_dag_Lk = Lk_dag @ Lk
        term2 = -0.5 * (np.kron(Lk_dag_Lk, I3) + np.kron(I3, Lk_dag_Lk.T))
        lindblad_super += term1 + term2

    def get_H(I, Q, delta, scale):
        H = np.zeros((3, 3), dtype=complex)
        H[0, 0] = 0
        H[1, 1] = delta
        H[2, 2] = 2 * delta + alpha
        
        H[0, 1] += scale * (I * M_I[0, 1] + Q * M_Q[0, 1])
        H[1, 0] += scale * (I * M_I[1, 0] + Q * M_Q[1, 0])
        H[1, 2] += scale * (I * M_I[1, 2] + Q * M_Q[1, 2])
        H[2, 1] += scale * (I * M_I[2, 1] + Q * M_Q[2, 1])
        
        return H

    def simulate_pulse(I_Q, delta, scale):
        """Simulate pulse with Lindblad and return list of infidelities for 6 states."""
        # Precompute U for each step
        U_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            H = get_H(I, Q, delta, scale)
            
            # Build Liouvillian: L = -i*H⊗I + lindblad
            H_super = -1j * np.kron(H, I3)
            L_total = H_super + dt * lindblad_super
            U = expm(L_total)
            U_steps.append(U)
        
        infidelities = []
        for idx in range(6):
            rho = np.outer(states[idx], states[idx].conj())
            rho = rho.flatten()
            
            for i in range(N):
                rho = U_steps[i] @ rho
            
            # Extract density matrix
            rho_mat = rho.reshape((3, 3))
            
            # Fidelity with target
            target = targets[idx]
            target_rho = np.outer(target, target.conj())
            
            # Fidelity = <target|rho|target>
            fid = np.real(np.vdot(target, rho_mat @ target))
            fid = np.clip(fid, 0, 1)
            infidelities.append(1 - fid)
        
        return infidelities

    def check_constraints(I_Q):
        """Check amplitude and slew constraints."""
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-10:
            return False
        
        # From 0 to step 1
        if amps[0] / dt > slew_bound + 1e-10:
            return False
        
        # Between steps
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-10:
            return False
        
        # From last step to 0
        if amps[-1] / dt > slew_bound + 1e-10:
            return False
        
        return True

    def score_pulse(I_Q):
        """Compute the scoring metric: mean of (mean_inf + max_inf) over 9 cases."""
        total_score = 0.0
        count = 0
        for delta in deltas:
            for scale in scales:
                infs = simulate_pulse(I_Q, delta, scale)
                mean_inf = np.mean(infs)
                max_inf = np.max(infs)
                total_score += (mean_inf + max_inf)
                count += 1
        return total_score / count

    def generate_pulse(sigma, eta, phase_shift=0.0, width_factor=1.0):
        """Generate DRAG-style Gaussian pulse with optional phase shift and width adjustment."""
        t = np.linspace(0, 40, N)
        t_c = 20.0
        sigma_eff = sigma * width_factor
        A = np.sqrt(np.pi / 2.0) / sigma_eff
        I = A * np.exp(-(t - t_c)**2 / (2 * sigma_eff**2))
        
        dI = I * (-(t - t_c) / (sigma_eff**2))
        Q = eta * dI
        
        # Apply phase shift
        I_Q = np.column_stack((I, Q))
        if abs(phase_shift) > 1e-10:
            I_new = I * np.cos(phase_shift) - Q * np.sin(phase_shift)
            Q_new = I * np.sin(phase_shift) + Q * np.cos(phase_shift)
            I_Q = np.column_stack((I_new, Q_new))
        
        return I_Q

    # Initial grid search for good starting point
    best_score = np.inf
    best_params = (10.0, 5.0, 0.0, 1.0)
    best_IQ = None

    # Try a range of sigma, eta, phase_shift, width_factor values
    for sigma in [6, 7, 8, 9, 10, 11, 12, 14, 16]:
        if time.time() > deadline - 60:
            break
        for eta in [-15, -10, -5, 0, 5, 10, 15]:
            for phase_shift in [-0.2, 0.0, 0.2]:
                for width_factor in [0.9, 1.0, 1.1]:
                    I_Q = generate_pulse(sigma, eta, phase_shift, width_factor)
                    if not check_constraints(I_Q):
                        continue
                    s = score_pulse(I_Q)
                    if s < best_score:
                        best_score = s
                        best_params = (sigma, eta, phase_shift, width_factor)
                        best_IQ = I_Q.copy()

    # If we found a valid pulse, try local refinement using finite differences
    if best_IQ is not None and time.time() < deadline - 30:
        # Simple gradient-free local search with adaptive step size
        current_IQ = best_IQ.copy()
        current_score = best_score
        step_size = 0.005
        max_iterations = 200
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            improved = False
            # Try small perturbations in I and Q
            for i in range(N):
                for j in range(2):
                    # Increase
                    old_val = current_IQ[i, j]
                    current_IQ[i, j] += step_size
                    if check_constraints(current_IQ):
                        s = score_pulse(current_IQ)
                        if s < current_score - 1e-12:
                            current_score = s
                            improved = True
                        else:
                            current_IQ[i, j] = old_val
                    else:
                        current_IQ[i, j] = old_val
                    
                    # Decrease
                    current_IQ[i, j] -= step_size
                    if check_constraints(current_IQ):
                        s = score_pulse(current_IQ)
                        if s < current_score - 1e-12:
                            current_score = s
                            improved = True
                        else:
                            current_IQ[i, j] = old_val
                    else:
                        current_IQ[i, j] = old_val
                
                if time.time() > deadline - 15:
                    break
            
            if not improved:
                # Reduce step size
                step_size *= 0.8
                if step_size < 1e-5:
                    break

        if current_score < best_score:
            best_score = current_score
            best_IQ = current_IQ.copy()

    # Final constraint check
    if best_IQ is None or not check_constraints(best_IQ):
        # Fallback to seed pulse
        sigma_seed = 40.0 / 6.0
        best_IQ = generate_pulse(sigma_seed, 0.0)
        if not check_constraints(best_IQ):
            I = np.full(N, np.pi / 40.0)
            Q = np.zeros(N)
            best_IQ = np.column_stack((I, Q))
            if not check_constraints(best_IQ):
                I = np.full(N, 0.01)
                Q = np.zeros(N)
                best_IQ = np.column_stack((I, Q))

    return best_IQ
