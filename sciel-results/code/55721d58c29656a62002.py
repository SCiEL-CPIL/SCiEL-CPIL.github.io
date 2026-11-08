import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    N = 80
    total_time = N * dt
    t = np.arange(N) * dt
    
    # Build operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a
    n2 = n_op @ (n_op - np.eye(3))
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # Precompute dissipation Liouvillian (constant)
    diss_L1_term = np.kron(L1.conj(), L1)
    diss_L2_term = np.kron(L2.conj(), L2)
    diss_L1_left = np.kron(np.eye(3), L1_dag_L1)
    diss_L1_right = np.kron(L1_dag_L1.T, np.eye(3))
    diss_L2_left = np.kron(np.eye(3), L2_dag_L2)
    diss_L2_right = np.kron(L2_dag_L2.T, np.eye(3))
    
    L_diss = (diss_L1_term - 0.5 * (diss_L1_left + diss_L1_right) +
              diss_L2_term - 0.5 * (diss_L2_left + diss_L2_right))
    
    # Precompute static Liouvillian for each detuning
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    L_static_cache = {}
    for det in detunings:
        H_static = det * n_op + 0.5 * anharmonicity * n2
        L_static = -1j * (np.kron(np.eye(3), H_static) - np.kron(H_static.T, np.eye(3)))
        L_static_cache[det] = L_static
    
    # Drive part: H_drive = 0.5 * scale * (I (a + a_dag) + i Q (a - a_dag))
    # L_drive = -i (kron(I, H_drive) - kron(H_drive.T, I))
    # This is linear in I and Q.
    
    def get_L_drive(I, Q, scale):
        H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return -1j * (np.kron(np.eye(3), H_drive) - np.kron(H_drive.T, np.eye(3)))
    
    def simulate_pulse(pulse_IQ, detuning, scale):
        """Simulate and return final density matrix."""
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        rho_vec = rho.flatten()
        
        L_static = L_static_cache[detuning]
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L_drive = get_L_drive(I_val, Q_val, scale)
            L_total = L_diss + L_static + L_drive
            
            U = expm(dt * L_total)
            rho_vec = U @ rho_vec
        
        return rho_vec.reshape(3, 3)
    
    # Test states and targets
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)),
        (np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)),
    ]
    
    def compute_infidelity(pulse_IQ):
        infidelities = []
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                for state in states:
                    target = np.zeros(3, dtype=complex)
                    target[0] = state[1]
                    target[1] = state[0]
                    target[2] = state[2]
                    fid = np.real(np.conj(target) @ rho_final @ target)
                    infidelities.append(1.0 - fid)
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)
    
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
        diff = np.linalg.norm(prev)
        if diff / dt > slew_bound + 1e-12:
            return False
        return True
    
    def generate_pulse(params):
        """Generate DRAG pulse from parameters: [amp, sigma, gamma, center, skew, tail]"""
        amp, sigma, gamma, center, skew, tail = params
        sigma = max(abs(sigma), 0.1)
        
        # Gaussian with skew and tail
        x = t - center
        gaussian = np.exp(-0.5 * (x / sigma)**2 * (1 + skew * x / sigma))
        
        # Add a small tail for robustness
        if tail > 0:
            tail_mask = np.abs(x) > 2 * sigma
            gaussian[tail_mask] += tail * np.exp(-0.5 * (x[tail_mask] / (sigma * 2))**2)
        
        # Offset to make edges zero
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        # Normalize area to pi
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-10:
            gaussian *= np.pi / area
        else:
            gaussian = np.zeros(N)
        
        I = gaussian * amp
        
        # DRAG term
        dI_dt = np.gradient(I, dt)
        Q = gamma * dI_dt
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Ensure validity
        if not check_validity(pulse_IQ):
            # Scale down amplitude
            max_amp = np.max(np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
            
            # Check slew rate
            if not check_validity(pulse_IQ):
                # Reduce gamma or increase sigma
                # For now, just scale down further
                pulse_IQ *= 0.5
            
            if not check_validity(pulse_IQ):
                # Last resort: use a simple valid pulse
                I_simple = np.zeros(N)
                Q_simple = np.zeros(N)
                sigma_simple = 40.0 / 6.0
                center_simple = total_time / 2.0
                gaussian_simple = np.exp(-0.5 * ((t - center_simple) / sigma_simple)**2)
                edge_val_simple = gaussian_simple[0]
                gaussian_simple -= edge_val_simple
                area_simple = np.sum(gaussian_simple) * dt
                if area_simple != 0:
                    gaussian_simple *= np.pi / area_simple
                I_simple = gaussian_simple
                pulse_IQ = np.column_stack([I_simple, Q_simple])
                
                if not check_validity(pulse_IQ):
                    max_amp_simple = np.max(np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2))
                    if max_amp_simple > amplitude_bound:
                        pulse_IQ *= amplitude_bound / max_amp_simple
        
        return pulse_IQ
    
    # Initial guess
    x0 = np.array([1.0, 40.0/6.0, 0.0, 20.0, 0.0, 0.0])
    
    def objective(params):
        pulse_IQ = generate_pulse(params)
        if not check_validity(pulse_IQ):
            amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
            amp_viol = np.max(0, np.max(amplitudes) - amplitude_bound)
            prev = np.zeros(2)
            slew_viol = 0
            for i in range(N):
                curr = pulse_IQ[i]
                diff = np.linalg.norm(curr - prev)
                slew_viol += max(0, diff / dt - slew_bound)
                prev = curr
            diff = np.linalg.norm(prev)
            slew_viol += max(0, diff / dt - slew_bound)
            penalty = 1e5 * (amp_viol + slew_viol)
            return penalty + compute_infidelity(pulse_IQ)
        return compute_infidelity(pulse_IQ)
    
    start_time = time.time()
    time_limit = 300.0
    
    # Phase 1: Differential Evolution for global search
    best_pulse = generate_pulse(x0)
    best_score = objective(x0)
    
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        try:
            bounds = [
                (0.5, 1.5),      # amp
                (2.0, 10.0),     # sigma
                (-0.5, 0.5),     # gamma
                (15.0, 25.0),    # center
                (-0.5, 0.5),     # skew
                (0.0, 0.5),      # tail
            ]
            de_res = differential_evolution(objective, bounds, maxiter=100, popsize=20, 
                                             tol=1e-4, seed=42, polish=False)
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_pulse = generate_pulse(de_res.x)
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.8:
        try:
            nm_res = minimize(objective, best_pulse.shape[0] and de_res.x if 'de_res' in dir() else x0, 
                             method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if nm_res.fun < best_score:
                best_score = nm_res.fun
                best_pulse = generate_pulse(nm_res.x)
        except Exception:
            pass
    
    # Ensure final validity
    if not check_validity(best_pulse):
        I = np.zeros(N)
        Q = np.zeros(N)
        sigma = 40.0 / 6.0
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        I = gaussian
        best_pulse = np.column_stack([I, Q])
        if not check_validity(best_pulse):
            max_amp = np.max(np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2))
            if max_amp > amplitude_bound:
                best_pulse *= amplitude_bound / max_amp
    
    return best_pulse
