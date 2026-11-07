"""Robust X_pi control pulse for a three-level transmon using DRAG optimization."""
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
    # L(rho) = -i[H, rho] + L1 rho L1^dag - 0.5{L1^dag L1, rho} + L2 rho L2^dag - 0.5{L2^dag L2, rho}
    # H = H_static + H_drive
    # L = L_static + L_drive
    
    # Static part of H: detuning * n + 0.5 * anharmonicity * n(n-1)
    # Note: detuning varies in test cases, so H_static depends on detuning.
    # But dissipation is constant.
    
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # Vectorized terms for dissipation
    # vec(L1 rho L1^dag) = (L1^T ⊗ L1) vec(rho) ? No, vec(A X B) = (B^T ⊗ A) vec(X)
    # So vec(L1 rho L1^dag) = ((L1^dag)^T ⊗ L1) vec(rho) = (L1.conj() ⊗ L1) vec(rho)
    # Wait, L1 is complex. (B^T ⊗ A). B = L1^dag. B^T = (L1^dag)^T = L1.conj().
    # So term is kron(L1.conj(), L1).
    
    diss_L1_term = np.kron(L1.conj(), L1)
    diss_L2_term = np.kron(L2.conj(), L2)
    
    # vec(L1^dag L1 rho) = (I ⊗ L1^dag L1) vec(rho)
    # vec(rho L1^dag L1) = ((L1^dag L1)^T ⊗ I) vec(rho)
    diss_L1_left = np.kron(np.eye(3), L1_dag_L1)
    diss_L1_right = np.kron(L1_dag_L1.T, np.eye(3))
    
    diss_L2_left = np.kron(np.eye(3), L2_dag_L2)
    diss_L2_right = np.kron(L2_dag_L2.T, np.eye(3))
    
    L_diss = diss_L1_term - 0.5 * (diss_L1_left + diss_L1_right) + \
             diss_L2_term - 0.5 * (diss_L2_left + diss_L2_right)
    
    # Static Hamiltonian part (without drive)
    # H_static = detuning * n + 0.5 * anharmonicity * n(n-1)
    # Liouvillian static part: -i(H_static rho - rho H_static)
    # vec(H_static rho) = (I ⊗ H_static) vec(rho)
    # vec(rho H_static) = (H_static^T ⊗ I) vec(rho)
    
    def get_L_drive(I, Q, scale):
        """Get the drive part of the Liouvillian."""
        # H_drive = 0.5 * scale * (I (a + a_dag) + i Q (a - a_dag))
        H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        L_drive = -1j * (np.kron(np.eye(3), H_drive) - np.kron(H_drive.T, np.eye(3)))
        return L_drive
    
    def get_L_static(detuning):
        """Get the static part of the Liouvillian (detuning and anharmonicity)."""
        H_static = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        L_static = -1j * (np.kron(np.eye(3), H_static) - np.kron(H_static.T, np.eye(3)))
        return L_static
    
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
        
        infidelities = []
        
        for det in detunings:
            for scale in scales:
                rho_final = simulate_pulse(pulse_IQ, det, scale)
                
                # The target is X_pi gate, which swaps |0> and |1>
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

    def generate_pulse(params):
        """Generate a DRAG-style pulse from parameters."""
        # params: [amp, sigma, gamma, center]
        amp = params[0]
        sigma = params[1]
        gamma = params[2]
        center = params[3]
        
        # Gaussian envelope
        # Ensure sigma is positive
        sigma = max(sigma, 0.1)
        
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        
        # Offset to make edges zero
        edge_val = gaussian[0]
        gaussian -= edge_val
        
        # Normalize area to pi (for X_pi gate)
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        
        # Scale by amp (amp should be 1.0 for nominal, but we optimize it)
        # Actually, let's treat amp as a scaling factor for the normalized pulse
        # The normalized pulse has area pi. So the integral is pi * amp.
        # For X_pi, we want area pi. So amp should be 1.
        # But we might want to optimize amp slightly? No, area is fixed by physics.
        # Let's fix amp to 1 and optimize sigma, gamma, center.
        # Or, keep amp as a free parameter and penalize if area != pi?
        # Better: Fix area to pi, optimize shape.
        
        I = gaussian * amp
        
        # DRAG term: Q = gamma * dI/dt
        dI_dt = np.gradient(I, dt)
        Q = gamma * dI_dt
        
        pulse_IQ = np.column_stack([I, Q])
        
        # Check validity and scale if needed
        if not check_validity(pulse_IQ):
            # Scale down to satisfy constraints
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                scale_factor = amplitude_bound / max_amp
                pulse_IQ *= scale_factor
            
            # Check slew rate again
            # Simple check: if slew rate is too high, reduce amplitude or increase sigma
            # For now, just scale down if amplitude is violated
            # Slew rate violation is harder to fix simply, so we rely on optimizer to avoid it
            # Or we can add a penalty in the objective function
            
            if not check_validity(pulse_IQ):
                # Return a valid pulse (e.g., zero) or heavily penalize
                # To ensure we return a valid pulse, we can return the seed pulse
                # But it's better to let the optimizer handle it with a penalty
                pass
        
        return pulse_IQ

    # Initial parameters: amp=1, sigma=40/6, gamma=0, center=20
    x0 = np.array([1.0, 40.0/6.0, 0.0, 20.0])
    
    def objective(params):
        pulse_IQ = generate_pulse(params)
        if not check_validity(pulse_IQ):
            # Penalty for invalidity
            # Calculate violation magnitude
            amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
            amp_viol = np.max(0, np.max(amplitudes) - amplitude_bound)
            
            prev = np.zeros(2)
            slew_viol = 0
            for i in range(N):
                curr = pulse_IQ[i]
                diff = np.linalg.norm(curr - prev)
                slew_viol += max(0, diff / dt - slew_bound)
                prev = curr
            diff = np.linalg.norm(prev - np.zeros(2))
            slew_viol += max(0, diff / dt - slew_bound)
            
            penalty = 1e5 * (amp_viol + slew_viol)
            return penalty + compute_infidelity(pulse_IQ)
        
        return compute_infidelity(pulse_IQ)
    
    # Optimize
    start_time = time.time()
    time_limit = 300.0
    
    best_pulse = generate_pulse(x0)
    best_score = objective(x0)
    
    # Use Nelder-Mead for small parameter space
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_pulse = generate_pulse(res.x)
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        
    # Ensure the final pulse is valid
    if not check_validity(best_pulse):
        # Fallback to a valid pulse
        # Generate a simple valid pulse
        I = np.zeros(N)
        Q = np.zeros(N)
        # Simple rectangular pulse? No, must be smooth.
        # Use the seed pulse
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        I = gaussian
        Q = np.zeros(N)
        best_pulse = np.column_stack([I, Q])
        
        if not check_validity(best_pulse):
            # Scale down
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                best_pulse *= amplitude_bound / max_amp
            
            if not check_validity(best_pulse):
                # Last resort: zero pulse (invalid for gate, but valid for constraints)
                # Actually, zero pulse is valid for constraints but bad for infidelity
                # We must return a valid pulse.
                # If seed is invalid, we need to make it valid.
                # The seed should be valid.
                pass

    return best_pulse
# EVOLVE-BLOCK-END
