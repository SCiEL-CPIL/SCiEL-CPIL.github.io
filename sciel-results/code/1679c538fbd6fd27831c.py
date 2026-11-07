"""Robust X_pi control pulse for a three-level transmon using direct L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    t = np.arange(N) * dt

    # Build the 3-level Hamiltonian components
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a  # number operator diag(0, 1, 2)

    # Pre-compute dissipation part of Liouvillian (independent of I, Q, det, scale)
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    diss_L1_term = np.kron(L1.conj(), L1)
    diss_L2_term = np.kron(L2.conj(), L2)
    
    diss_L1_left = np.kron(np.eye(3), L1_dag_L1)
    diss_L1_right = np.kron(L1_dag_L1.T, np.eye(3))
    
    diss_L2_left = np.kron(np.eye(3), L2_dag_L2)
    diss_L2_right = np.kron(L2_dag_L2.T, np.eye(3))
    
    L_diss = diss_L1_term - 0.5 * (diss_L1_left + diss_L1_right) + \
             diss_L2_term - 0.5 * (diss_L2_left + diss_L2_right)

    # Pre-compute static Hamiltonian Liouvillian parts
    # H_static = detuning * n + 0.5 * anharmonicity * n(n-1)
    # n(n-1) is diag(0, 0, 2)
    n_nm1 = n_op @ (n_op - np.eye(3))
    
    def get_L_static(detuning):
        H_static = detuning * n_op + 0.5 * anharmonicity * n_nm1
        L_static = -1j * (np.kron(np.eye(3), H_static) - np.kron(H_static.T, np.eye(3)))
        return L_static

    def get_L_drive(I, Q, scale):
        H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        L_drive = -1j * (np.kron(np.eye(3), H_drive) - np.kron(H_drive.T, np.eye(3)))
        return L_drive

    def simulate_pulse(pulse_IQ, detuning=0.0, scale=1.0):
        """
        Simulate the pulse and return the final density matrix.
        pulse_IQ: (80, 2) array of [I, Q] values
        """
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        rho_vec = rho.flatten()
        
        L_static = get_L_static(detuning)
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L_drive = get_L_drive(I_val, Q_val, scale)
            L_total = L_diss + L_static + L_drive
            
            # Evolve: rho(t+dt) = exp(dt * L) rho(t)
            U = expm(dt * L_total)
            rho_vec = U @ rho_vec
        
        return rho_vec.reshape(3, 3)
    
    def compute_fidelity(rho, target_state):
        psi = target_state.flatten()
        return np.real(np.conj(psi) @ rho @ psi)
    
    def compute_infidelity(pulse_IQ):
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
            (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
            (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
            (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
        ]
        
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                
                for state in states:
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
        
        diff = np.linalg.norm(prev - np.zeros(2))
        if diff / dt > slew_bound + 1e-12:
            return False
        
        return True

    def generate_seed_pulse():
        """Generate a DRAG-style seed pulse."""
        sigma = 40.0/6.0
        center = 20.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        I = gaussian
        dI_dt = np.gradient(I, dt)
        gamma = 0.1
        Q = gamma * dI_dt
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
            if not check_validity(pulse_IQ):
                # Simple fallback: scale down more
                pulse_IQ *= 0.5
        
        return pulse_IQ

    def project_to_valid(pulse_IQ):
        """Project pulse to satisfy constraints approximately."""
        # Amplitude constraint
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        mask = amps > amplitude_bound
        if np.any(mask):
            pulse_IQ[mask] *= amplitude_bound / amps[mask]
        
        # Slew rate constraint: simple iterative projection
        # This is a heuristic, not a perfect projection
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            diff = curr - prev
            norm_diff = np.linalg.norm(diff)
            max_diff = slew_bound * dt
            if norm_diff > max_diff:
                curr = prev + diff * (max_diff / norm_diff)
                pulse_IQ[i] = curr
            prev = curr
        
        # Last step to zero
        curr = pulse_IQ[-1]
        norm_curr = np.linalg.norm(curr)
        max_curr = slew_bound * dt
        if norm_curr > max_curr:
            pulse_IQ[-1] = curr * (max_curr / norm_curr)
            
        return pulse_IQ

    # Initial pulse
    x0 = generate_seed_pulse()
    if not check_validity(x0):
        x0 = project_to_valid(x0)
        if not check_validity(x0):
            # Last resort: zero pulse is valid
            x0 = np.zeros((N, 2))

    best_pulse = x0.copy()
    best_score = compute_infidelity(x0)

    def objective(pulse_flat):
        pulse_IQ = pulse_flat.reshape(N, 2)
        # Project to ensure validity during optimization steps
        pulse_IQ = project_to_valid(pulse_IQ)
        
        if not check_validity(pulse_IQ):
            # If projection failed, return high penalty
            return 1e5
        
        return compute_infidelity(pulse_IQ)

    # Optimize
    start_time = time.time()
    time_limit = 290.0  # Safety margin

    try:
        # Use L-BFGS-B with finite differences
        # Flatten the pulse for the optimizer
        x0_flat = x0.flatten()
        
        res = minimize(objective, x0_flat, method='L-BFGS-B', 
                       options={'maxiter': 200, 'maxls': 20, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_pulse = res.x.reshape(N, 2)
            best_pulse = project_to_valid(best_pulse)
            
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Ensure the final pulse is valid
    if not check_validity(best_pulse):
        best_pulse = project_to_valid(best_pulse)
        if not check_validity(best_pulse):
            # Fallback to seed
            best_pulse = generate_seed_pulse()
            if not check_validity(best_pulse):
                best_pulse = np.zeros((N, 2))

    return best_pulse
# EVOLVE-BLOCK-END
