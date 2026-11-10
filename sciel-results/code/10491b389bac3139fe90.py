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
    deadline = start_time + 320.0  # 10s safety margin
    
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
    # a = diag(sqrt([1,2]),1) means a|1> = sqrt(1)|0>, a|2> = sqrt(2)|1>
    # In basis |0>,|1>,|2>:
    a_op = np.zeros((3, 3), dtype=complex)
    a_op[0, 1] = np.sqrt(1.0)  # a|1> = |0>
    a_op[1, 2] = np.sqrt(2.0)  # a|2> = sqrt(2)|1>
    a_dag = a_op.conj().T
    
    n_op = a_dag @ a_op  # number operator: diag(0, 1, 2)
    
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)  # amplitude damping
    L2 = n_op * np.sqrt(2.0 / Tphi)  # pure dephasing
    
    def build_hamiltonian(detuning, scale, I, Q):
        """Build the Hamiltonian for given parameters."""
        # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I(a+a†) + iQ(a-a†))
        H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
        H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H
    
    def lindblad_superop(H, L1, L2):
        """Build the Liouvillian superoperator for Lindblad master equation.
        dρ/dt = -i[H,ρ] + L1 ρ L1† + L1† ρ L1 - 0.5{L1†L1,ρ} + L2 ρ L2† + L2† ρ L2 - 0.5{L2†L2,ρ}
        Vectorize ρ as column-stacked vector.
        """
        dim = 3
        # -i[H,ρ] term: -i(H⊗I - I⊗H^T)
        H_comm = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.T))
        
        # Lindblad terms for L
        def lindblad_term(L):
            L_dag = L.conj().T
            # L ρ L†: np.kron(L, L†)
            # L† ρ L: np.kron(L, L)  -- wait, need to be careful
            # Vectorization: vec(L ρ L†) = (L† ⊗ L) vec(ρ)
            # vec(L† ρ L) = (L ⊗ L†) vec(ρ)  -- no
            # Actually: vec(A ρ B) = (B^T ⊗ A) vec(ρ)
            # So vec(L ρ L†) = ((L†)^T ⊗ L) vec(ρ) = (L^* ⊗ L) vec(ρ)
            # vec(L† ρ L) = (L^T ⊗ L†) vec(ρ)
            # Anti-commutator term: -0.5{L†L, ρ} = -0.5(L†L ρ + ρ L†L)
            # vec(L†L ρ) = (I ⊗ L†L) vec(ρ)
            # vec(ρ L†L) = ((L†L)^T ⊗ I) vec(ρ) = (L†L ⊗ I) vec(ρ) since L†L is Hermitian
            LdL = L_dag @ L
            term = np.kron(L.conj(), L) + np.kron(L, L_dag) - 0.5 * (np.kron(LdL, np.eye(dim)) + np.kron(np.eye(dim), LdL))
            return term
        
        L1_term = lindblad_term(L1)
        L2_term = lindblad_term(L2)
        
        return H_comm + L1_term + L2_term
    
    def simulate_pulse(I_arr, Q_arr, detuning, scale):
        """Simulate the pulse and return the final density matrix (9-dim vector)."""
        dim = 3
        rho_vec = np.zeros(dim * dim, dtype=complex)
        rho_vec[0] = 1.0  # |0><0|
        
        for i in range(n_steps):
            H = build_hamiltonian(detuning, scale, I_arr[i], Q_arr[i])
            L = lindblad_superop(H, L1, L2)
            # Exponentiate: rho_{i+1} = exp(L * dt) * rho_i
            # Use matrix exponential
            expL = sla.expm(L * dt)
            rho_vec = expL @ rho_vec
        
        return rho_vec
    
    def fidelity_to_X(rho_vec):
        """Compute fidelity to X gate for the given state evolution.
        We need to compute fidelity for 5 initial states and average.
        """
        dim = 3
        # Target: X gate on the qubit subspace |0>,|1>
        # X|0> = |1>, X|1> = |0>
        # For the 5 states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
        # Wait, the task says 4 states: |0>, |1>, (|0>+-|1>)/sqrt(2), (|0>+-i|1>)/sqrt(2)
        # That's actually 5 states if we count both + and - for each.
        # Let me re-read: "fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
        # So 5 states total.
        
        initial_states = []
        # |0>
        s0 = np.zeros(dim, dtype=complex)
        s0[0] = 1.0
        initial_states.append(s0)
        # |1>
        s1 = np.zeros(dim, dtype=complex)
        s1[1] = 1.0
        initial_states.append(s1)
        # (|0>+|1>)/sqrt(2)
        sp = np.zeros(dim, dtype=complex)
        sp[0] = 1.0 / np.sqrt(2)
        sp[1] = 1.0 / np.sqrt(2)
        initial_states.append(sp)
        # (|0>-|1>)/sqrt(2)
        sm = np.zeros(dim, dtype=complex)
        sm[0] = 1.0 / np.sqrt(2)
        sm[1] = -1.0 / np.sqrt(2)
        initial_states.append(sm)
        # (|0>+i|1>)/sqrt(2)
        sip = np.zeros(dim, dtype=complex)
        sip[0] = 1.0 / np.sqrt(2)
        sip[1] = 1j / np.sqrt(2)
        initial_states.append(sip)
        # (|0>-i|1>)/sqrt(2)
        sim = np.zeros(dim, dtype=complex)
        sim[0] = 1.0 / np.sqrt(2)
        sim[1] = -1j / np.sqrt(2)
        initial_states.append(sim)
        
        # Target states after X gate
        X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
        
        fidelities = []
        for psi in initial_states:
            rho_init_vec = np.outer(psi, psi.conj()).flatten()
            # Simulate
            H_det = build_hamiltonian(0, 1.0, 0, 0)  # placeholder, we'll simulate properly
            # Actually, we need to simulate with the actual pulse
            # Let's refactor: simulate each initial state
            dim3 = 3
            rho_v = rho_init_vec.copy()
            for i in range(n_steps):
                H = build_hamiltonian(0, 1.0, I_arr[i], Q_arr[i])  # nominal for now
                L = lindblad_superop(H, L1, L2)
                expL = sla.expm(L * dt)
                rho_v = expL @ rho_v
            
            # Compute fidelity
            rho_final = rho_v.reshape(dim, dim)
            psi_target = X_mat @ psi
            # Fidelity = <psi_target|rho_final|psi_target>
            fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
            fidelities.append(fid)
        
        return np.mean(fidelities)
    
    def compute_infidelity(I_arr, Q_arr):
        """Compute mean and max infidelity over 9 corners."""
        detunings = [-detuning_bound, 0.0, detuning_bound]
        scales = [scale_min, 1.0, scale_max]
        
        infs = []
        for det in detunings:
            for sc in scales:
                # Simulate all 5 initial states
                dim = 3
                initial_states = [
                    np.array([1.0, 0, 0], dtype=complex),
                    np.array([0, 1.0, 0], dtype=complex),
                    np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex),
                    np.array([1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex),
                    np.array([1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=complex),
                    np.array([1/np.sqrt(2), -1j/np.sqrt(2), 0], dtype=complex),
                ]
                
                X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
                
                fids = []
                for psi in initial_states:
                    rho_v = np.outer(psi, psi.conj()).flatten()
                    for i in range(n_steps):
                        H = build_hamiltonian(det, sc, I_arr[i], Q_arr[i])
                        L = lindblad_superop(H, L1, L2)
                        expL = sla.expm(L * dt)
                        rho_v = expL @ rho_v
                    rho_final = rho_v.reshape(dim, dim)
                    psi_t = X_mat @ psi
                    fid = np.real(np.vdot(psi_t, rho_final @ psi_t))
                    fids.append(fid)
                
                inf = 1.0 - np.mean(fids)
                infs.append(inf)
        
        infs = np.array(infs)
        return np.mean(infs), np.max(infs)
    
    def is_valid(I_arr, Q_arr):
        """Check amplitude and slew constraints."""
        # Amplitude bound
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        if np.any(amps > amp_bound + 1e-12):
            return False
        # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
        # From zero to step 1
        p0 = np.array([0.0, 0.0])
        p1 = np.array([I_arr[0], Q_arr[0]])
        if np.linalg.norm(p1 - p0) / dt > slew_bound + 1e-12:
            return False
        # Between steps
        for i in range(n_steps - 1):
            pi = np.array([I_arr[i], Q_arr[i]])
            pip1 = np.array([I_arr[i+1], Q_arr[i+1]])
            if np.linalg.norm(pip1 - pi) / dt > slew_bound + 1e-12:
                return False
        # From last step to zero
        plast = np.array([I_arr[-1], Q_arr[-1]])
        if np.linalg.norm(p0 - plast) / dt > slew_bound + 1e-12:
            return False
        return True
    
    # DRAG-style Gaussian seed
    def make_seed():
        t = np.arange(n_steps) * dt
        center = T / 2
        sigma = (40 / 6) / 2  # sigma in time, but need to think about this
        # Actually sigma = 40/6 ns? Let me use sigma_t = 40/6
        sigma_t = 40.0 / 6.0
        # Gaussian pulse with area pi
        # I(t) = A * exp(-t^2/(2*sigma^2))
        # Area = A * sigma * sqrt(2*pi) = pi => A = 1/(sigma*sqrt(2))
        # But we need to offset to 0 at edges
        t_centered = t - center
        gauss = np.exp(-t_centered**2 / (2 * sigma_t**2))
        # Normalize so that the integral is pi
        # Integral of exp(-x^2/(2σ²)) dx = σ√(2π)
        area = np.sum(gauss) * dt
        I_seed = gauss * (np.pi / area)
        Q_seed = np.zeros(n_steps)
        
        # DRAG: add Q component proportional to derivative of I
        # Q(t) = -alpha * dI/dt
        # Choose alpha to suppress leakage
        dI = np.gradient(I_seed, dt)
        # Typical DRAG alpha for transmon with anharmonicity ~0.25 GHz
        # alpha ~ 1/(4*|anharmonicity|) or similar
        # Let's use a reasonable value
        alpha = 1.0 / (4.0 * abs(anharmonicity))
        Q_seed = -alpha * dI
        
        # Project to satisfy constraints
        # Scale down if needed
        amps = np.sqrt(I_seed**2 + Q_seed**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I_seed *= scale
            Q_seed *= scale
        
        # Check slew
        if not is_valid(I_seed, Q_seed):
            # Scale down further
            for _ in range(10):
                I_seed *= 0.95
                Q_seed *= 0.95
                if is_valid(I_seed, Q_seed):
                    break
        
        return I_seed, Q_seed
    
    I0, Q0 = make_seed()
    
    # Check validity of seed
    if not is_valid(I0, Q0):
        # Further scale down
        scale = 0.5
        I0 = I0 * scale
        Q0 = Q0 * scale
    
    # Compute seed score
    seed_mean, seed_max = compute_infidelity(I0, Q0)
    seed_score = seed_mean + seed_max
    
    best_I = I0.copy()
    best_Q = Q0.copy()
    best_score = seed_score
    
    # Try a few simple variations
    variations = [
        (I0, Q0),  # seed
    ]
    
    # Try adjusting the DRAG alpha
    for alpha_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        t = np.arange(n_steps) * dt
        center = T / 2
        sigma_t = 40.0 / 6.0
        t_centered = t - center
        gauss = np.exp(-t_centered**2 / (2 * sigma_t**2))
        area = np.sum(gauss) * dt
        I_var = gauss * (np.pi / area)
        dI = np.gradient(I_var, dt)
        alpha = alpha_mult / (4.0 * abs(anharmonicity))
        Q_var = -alpha * dI
        
        # Project
        amps = np.sqrt(I_var**2 + Q_var**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I_var *= scale
            Q_var *= scale
        
        if is_valid(I_var, Q_var):
            mean_inf, max_inf = compute_infidelity(I_var, Q_var)
            score = mean_inf + max_inf
            if score < best_score:
                best_score = score
                best_I = I_var.copy()
                best_Q = Q_var.copy()
    
    # Check time
    if time.time() > deadline - 10:
        return np.column_stack([best_I, best_Q])
    
    # Try varying the Gaussian width
    for sigma_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        t = np.arange(n_steps) * dt
        center = T / 2
        sigma_t = (40.0 / 6.0) * sigma_mult
        t_centered = t - center
        gauss = np.exp(-t_centered**2 / (2 * sigma_t**2))
        area = np.sum(gauss) * dt
        I_var = gauss * (np.pi / area)
        dI = np.gradient(I_var, dt)
        alpha = 1.0 / (4.0 * abs(anharmonicity))
        Q_var = -alpha * dI
        
        amps = np.sqrt(I_var**2 + Q_var**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I_var *= scale
            Q_var *= scale
        
        if is_valid(I_var, Q_var):
            mean_inf, max_inf = compute_infidelity(I_var, Q_var)
            score = mean_inf + max_inf
            if score < best_score:
                best_score = score
                best_I = I_var.copy()
                best_Q = Q_var.copy()
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
