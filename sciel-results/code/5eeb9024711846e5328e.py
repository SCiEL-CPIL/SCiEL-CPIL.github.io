import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0  # 310s safety margin (330s limit, 20s buffer)
    
    # Parameters
    dt = 0.5
    n_steps = 80
    T_total = 40.0
    
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    
    # Time samples
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0
    
    # Precompute operators (constant across all evaluations)
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    
    # Anharmonicity term: 0.5 * anharmonicity * n * (n - I)
    anham_term = 0.5 * anharmonicity * n_op @ (n_op - I3)
    
    # Lindblad operators and their superoperator contributions
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    # Precompute constant Lindblad superoperator part
    lindblad_super = (np.kron(L1.conj(), L1) - 0.5 * (np.kron(I3, L1_dag_L1) + np.kron(L1_dag_L1.T, I3))
                     + np.kron(L2.conj(), L2) - 0.5 * (np.kron(I3, L2_dag_L2) + np.kron(L2_dag_L2.T, I3)))
    
    # Hamiltonian basis terms
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    
    def ham_super(H_comp):
        return -1j * (np.kron(I3, H_comp) - np.kron(H_comp.T, I3))
    
    n_super = ham_super(n_op)
    anham_super = ham_super(anham_term)
    I_drive_super = ham_super(0.5 * a_plus_adag)
    Q_drive_super = ham_super(0.5 * 1j * a_minus_adag)
    
    # Detuning and scale cases
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]
    
    # Precompute Liouvillian base for each (det, scale) combination
    liouvillian_bases = {}
    for det in detunings:
        for sc in scales:
            base = det * n_super + anham_super + lindblad_super
            liouvillian_bases[(det, sc)] = base
    
    # Initial states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    
    # X gate
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    target_states = [X @ psi for psi in initial_states]
    
    def check_validity(pulse_IQ):
        """Check amplitude and slew constraints."""
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amp_bound + 1e-10):
            return False
        
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(pulse_IQ[i] - prev)
            if diff / dt > slew_bound + 1e-10:
                return False
            prev = pulse_IQ[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        if diff / dt > slew_bound + 1e-10:
            return False
        return True
    
    def simulate_one_case(pulse_IQ, det, sc, rho_init):
        """Simulate one case and return final density matrix."""
        base = liouvillian_bases[(det, sc)]
        rho = rho_init.copy()
        
        for i in range(n_steps):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            L_super = base + sc * (I_val * I_drive_super + Q_val * Q_drive_super)
            
            rho_flat = rho.flatten()
            rho_flat = expm(L_super * dt) @ rho_flat
            rho = rho_flat.reshape((3, 3))
        
        return rho
    
    def evaluate_pulse(pulse_IQ):
        """Return the score (lower is better)."""
        if not check_validity(pulse_IQ):
            return float('inf')
        
        scores = []
        for det in detunings:
            for sc in scales:
                fidelities = []
                for idx, psi in enumerate(initial_states):
                    rho_init = np.outer(psi, psi.conj())
                    rho_final = simulate_one_case(pulse_IQ, det, sc, rho_init)
                    
                    # Fidelity to target state |phi> = X|psi>
                    phi = target_states[idx]
                    F = np.vdot(phi, rho_final @ phi).real
                    F = max(0.0, min(1.0, F))
                    fidelities.append(F)
                
                mean_fid = np.mean(fidelities)
                infid = 1.0 - mean_fid
                scores.append(infid)
        
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        return mean_score + max_score
    
    # Generate seed pulse (Gaussian with area pi)
    sigma_seed = T_total / 6.0
    A_seed = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A_seed * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    Q_seed = np.zeros(n_steps)
    pulse_seed = np.column_stack([I_seed, Q_seed])
    
    # Evaluate seed
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)
    
    # DRAG pulse generation
    def make_drag_pulse(sigma, beta, area_scale=1.0):
        """Generate a DRAG pulse with given sigma and beta."""
        A = area_scale * np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        return np.column_stack([I_base, Q_drag])
    
    # Local optimization around seed
    sigma_vals = [T_total / 6.0, T_total / 5.8, T_total / 6.2, T_total / 5.5, T_total / 6.5]
    beta_vals = [0.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
    
    evaluated = set()
    evaluated.add(('seed',))
    
    for sigma in sigma_vals:
        if time.time() > deadline - 10.0:
            break
        for beta in beta_vals:
            if time.time() > deadline - 10.0:
                break
            key = (sigma, beta)
            if key in evaluated:
                continue
            evaluated.add(key)
            
            pulse_candidate = make_drag_pulse(sigma, beta)
            
            if not check_validity(pulse_candidate):
                continue
            
            score = evaluate_pulse(pulse_candidate)
            if score < best_score:
                best_score = score
                best_pulse = pulse_candidate.copy()
    
    # If we have time, try area scaling
    if time.time() < deadline - 30.0:
        for area_scale in [0.98, 1.0, 1.02]:
            if time.time() > deadline - 10.0:
                break
            key = ('area', area_scale)
            if key in evaluated:
                continue
            evaluated.add(key)
            
            pulse_candidate = make_drag_pulse(sigma_seed, 0.0, area_scale)
            
            if not check_validity(pulse_candidate):
                continue
            
            score = evaluate_pulse(pulse_candidate)
            if score < best_score:
                best_score = score
                best_pulse = pulse_candidate.copy()
    
    return best_pulse
