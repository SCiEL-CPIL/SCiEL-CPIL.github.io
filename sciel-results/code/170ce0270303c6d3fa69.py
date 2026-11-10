"""Pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # 20s safety margin before 330s kill
    
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_min, scale_max = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    # Build 3-level operators
    a_op = np.zeros((3, 3), dtype=complex)
    a_op[0, 1] = np.sqrt(1.0)
    a_op[1, 2] = np.sqrt(2.0)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    def build_hamiltonian(detuning, scale, I, Q):
        H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H
    
    def lindblad_superop(H, L1, L2):
        dim = 3
        H_comm = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.T))
        
        def lindblad_term(L):
            L_dag = L.conj().T
            LdL = L_dag @ L
            term = (np.kron(L.conj(), L) + np.kron(L, L_dag) 
                   - 0.5 * (np.kron(LdL, np.eye(dim)) + np.kron(np.eye(dim), LdL)))
            return term
        
        L1_term = lindblad_term(L1)
        L2_term = lindblad_term(L2)
        
        return H_comm + L1_term + L2_term
    
    # Precompute Liouvillian for each step and each (det, scale) combination
    # Since I and Q vary, we can't fully precompute, but we can precompute the static parts
    
    # Initial states for fidelity computation
    initial_states = [
        np.array([1.0, 0, 0], dtype=complex),
        np.array([0, 1.0, 0], dtype=complex),
        np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex),
        np.array([1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex),
        np.array([1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=complex),
        np.array([1/np.sqrt(2), -1j/np.sqrt(2), 0], dtype=complex),
    ]
    
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    # Precompute target states
    target_states = [X_mat @ psi for psi in initial_states]
    
    # Precompute initial density matrices
    init_rho_vecs = [np.outer(psi, psi.conj()).flatten() for psi in initial_states]
    
    def simulate_and_score(I_arr, Q_arr, detuning, scale):
        """Simulate all 6 initial states and return mean fidelity."""
        dim = 3
        fids = []
        
        for rho_init in init_rho_vecs:
            rho_v = rho_init.copy()
            for i in range(n_steps):
                H = build_hamiltonian(detuning, scale, I_arr[i], Q_arr[i])
                L = lindblad_superop(H, L1, L2)
                expL = sla.expm(L * dt)
                rho_v = expL @ rho_v
            
            rho_final = rho_v.reshape(dim, dim)
            psi_t = target_states[len(fids)]
            fid = np.real(np.vdot(psi_t, rho_final @ psi_t))
            fids.append(fid)
        
        return np.mean(fids)
    
    def compute_score(I_arr, Q_arr):
        """Compute mean + max infidelity over 9 (det, scale) combinations."""
        detunings = [-detuning_bound, 0.0, detuning_bound]
        scales = [scale_min, 1.0, scale_max]
        
        infs = []
        for det in detunings:
            for sc in scales:
                fid = simulate_and_score(I_arr, Q_arr, det, sc)
                infs.append(1.0 - fid)
        
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)
    
    def is_valid(I_arr, Q_arr):
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        if np.any(amps > amp_bound + 1e-12):
            return False
        # Slew from zero to step 0
        if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / dt > slew_bound + 1e-12:
            return False
        # Slew between steps
        for i in range(n_steps - 1):
            dI = I_arr[i+1] - I_arr[i]
            dQ = Q_arr[i+1] - Q_arr[i]
            if math.sqrt(dI*dI + dQ*dQ) / dt > slew_bound + 1e-12:
                return False
        # Slew from last step to zero
        if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / dt > slew_bound + 1e-12:
            return False
        return True
    
    def make_drag_pulse(sigma_t, alpha, area_scale=1.0):
        """Generate a DRAG-style Gaussian pulse."""
        t = np.arange(n_steps) * dt
        center = T / 2
        t_centered = t - center
        gauss = np.exp(-t_centered**2 / (2 * sigma_t**2))
        
        # Normalize to have area = pi * area_scale
        area = np.sum(gauss) * dt
        I = gauss * (np.pi * area_scale / area)
        
        # DRAG Q component
        dI = np.gradient(I, dt)
        Q = -alpha * dI
        
        # Check amplitude bound
        amps = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I *= scale
            Q *= scale
        
        # Check slew and scale down if needed
        if not is_valid(I, Q):
            for _ in range(20):
                I *= 0.95
                Q *= 0.95
                if is_valid(I, Q):
                    break
        
        return I, Q
    
    # Start with a good DRAG seed
    sigma_t = 40.0 / 6.0
    alpha = 1.0 / (4.0 * abs(anharmonicity))
    I0, Q0 = make_drag_pulse(sigma_t, alpha)
    
    # Compute seed score
    seed_score = compute_score(I0, Q0)
    best_I = I0.copy()
    best_Q = Q0.copy()
    best_score = seed_score
    
    # Grid search over sigma and alpha
    sigma_range = np.linspace(40.0/8.0, 40.0/4.0, 15)
    alpha_range = np.linspace(0.3/(4*abs(anharmonicity)), 2.0/(4*abs(anharmonicity)), 15)
    
    for sigma in sigma_range:
        for alpha in alpha_range:
            if time.time() > deadline - 30:
                break
            I_var, Q_var = make_drag_pulse(sigma, alpha)
            if not is_valid(I_var, Q_var):
                continue
            score = compute_score(I_var, Q_var)
            if score < best_score:
                best_score = score
                best_I = I_var.copy()
                best_Q = Q_var.copy()
        if time.time() > deadline - 30:
            break
    
    # Local optimization around the best found
    if time.time() < deadline - 40:
        # Try small perturbations
        np.random.seed(42)
        for iteration in range(50):
            if time.time() > deadline - 15:
                break
            
            # Random perturbation
            pert_I = best_I + np.random.randn(n_steps) * 0.01
            pert_Q = best_Q + np.random.randn(n_steps) * 0.01
            
            # Project to satisfy constraints
            # Amplitude bound
            amps = np.sqrt(pert_I**2 + pert_Q**2)
            mask = amps > amp_bound
            if np.any(mask):
                scale = amp_bound / amps[mask]
                pert_I[mask] *= scale
                pert_Q[mask] *= scale
            
            # Slew constraint: iteratively project
            for _ in range(5):
                # Check and fix slew from zero
                amp0 = np.sqrt(pert_I[0]**2 + pert_Q[0]**2)
                if amp0 / dt > slew_bound:
                    scale = slew_bound * dt / amp0
                    pert_I[0] *= scale
                    pert_Q[0] *= scale
                
                # Between steps
                for i in range(n_steps - 1):
                    dI = pert_I[i+1] - pert_I[i]
                    dQ = pert_Q[i+1] - pert_Q[i]
                    norm = math.sqrt(dI*dI + dQ*dQ)
                    if norm / dt > slew_bound:
                        scale = slew_bound * dt / norm
                        pert_I[i+1] = pert_I[i] + dI * scale
                        pert_Q[i+1] = pert_Q[i] + dQ * scale
                
                # From last to zero
                amp_last = np.sqrt(pert_I[-1]**2 + pert_Q[-1]**2)
                if amp_last / dt > slew_bound:
                    scale = slew_bound * dt / amp_last
                    pert_I[-1] *= scale
                    pert_Q[-1] *= scale
            
            if not is_valid(pert_I, pert_Q):
                continue
            
            score = compute_score(pert_I, pert_Q)
            if score < best_score:
                best_score = score
                best_I = pert_I.copy()
                best_Q = pert_Q.copy()
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
