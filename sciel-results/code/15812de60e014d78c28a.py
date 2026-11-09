"""Rewritten program for robust control pulses for a three-level transmon."""
import numpy as np
import time
from scipy.integrate import solve_ivp

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    dt = 0.5  # ns
    n_steps = 80
    total_time = 40.0  # ns
    
    # Extract from hardware_model (assuming it's a dict or similar)
    if hasattr(hardware_model, '__getitem__'):
        try:
            anharmonicity = hardware_model.get('anharmonicity', -2*np.pi*0.25)
            amp_bound = hardware_model.get('amplitude_bound', 0.4)
            slew_bound = hardware_model.get('slew_bound', 0.12)
            detuning_bounds = hardware_model.get('detuning_bounds', (-2*np.pi*0.002, 2*np.pi*0.002))
            scale_bounds = hardware_model.get('drive_scale_bounds', (0.97, 1.03))
            T1 = hardware_model.get('T1', 25000.0)
            Tphi = hardware_model.get('Tphi', 40000.0)
        except:
            anharmonicity = -2*np.pi*0.25
            amp_bound = 0.4
            slew_bound = 0.12
            detuning_bounds = (-2*np.pi*0.002, 2*np.pi*0.002)
            scale_bounds = (0.97, 1.03)
            T1 = 25000.0
            Tphi = 40000.0
    else:
        anharmonicity = -2*np.pi*0.25
        amp_bound = 0.4
        slew_bound = 0.12
        detuning_bounds = (-2*np.pi*0.002, 2*np.pi*0.002)
        scale_bounds = (0.97, 1.03)
        T1 = 25000.0
        Tphi = 40000.0
    
    # Start with DRAG-style Gaussian seed
    t = np.linspace(0, total_time, n_steps + 1)[:-1]  # 80 time points
    center = total_time / 2.0
    sigma = 40.0 / 6.0  # ~6.67 ns
    
    # Gaussian pulse shape
    gauss = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    
    # Normalize area to pi (for X-pi gate)
    area = np.sum(gauss) * dt
    scale_factor = np.pi / area
    I_seed = scale_factor * gauss
    Q_seed = np.zeros(n_steps)
    
    # Apply amplitude bound
    amp = np.sqrt(I_seed**2 + Q_seed**2)
    mask = amp > amp_bound
    if np.any(mask):
        I_seed[mask] *= amp_bound / amp[mask]
        Q_seed[mask] *= amp_bound / amp[mask]
    
    # Apply slew rate constraint
    # Check from zero into step 1
    if np.sqrt(I_seed[0]**2 + Q_seed[0]**2) / dt > slew_bound:
        I_seed[0] *= slew_bound * dt / np.sqrt(I_seed[0]**2 + Q_seed[0]**2)
        Q_seed[0] *= slew_bound * dt / np.sqrt(I_seed[0]**2 + Q_seed[0]**2)
    
    # Check between steps
    for i in range(n_steps - 1):
        dI = I_seed[i+1] - I_seed[i]
        dQ = Q_seed[i+1] - Q_seed[i]
        dist = np.sqrt(dI**2 + dQ**2)
        if dist / dt > slew_bound:
            # Scale down the next step to satisfy constraint
            factor = slew_bound * dt / dist
            I_seed[i+1] = I_seed[i] + dI * factor
            Q_seed[i+1] = Q_seed[i] + dQ * factor
    
    # Check from last step to zero
    if np.sqrt(I_seed[-1]**2 + Q_seed[-1]**2) / dt > slew_bound:
        I_seed[-1] *= slew_bound * dt / np.sqrt(I_seed[-1]**2 + Q_seed[-1]**2)
        Q_seed[-1] *= slew_bound * dt / np.sqrt(I_seed[-1]**2 + Q_seed[-1]**2)
    
    # Build the pulse array
    pulse_array = np.column_stack([I_seed, Q_seed])
    
    # Optimize using a simple gradient-free method to improve robustness
    # We'll try to minimize infidelity across the 9 test cases
    
    def compute_fidelity(pulse_arr, detuning, scale):
        """
        Simulate the three-level system and compute fidelity to X gate.
        """
        # Build Hamiltonian components
        # a = [[0,1,0],[0,0,0],[0,0,0]] in the |0>,|1>,|2> basis? 
        # Actually for a transmon, the ladder operator a acts on the computational subspace
        # Let's use the 3-level model: |0>, |1>, |2>
        # a = |0><1| + sqrt(2)|1><2|  (in the basis |0>,|1>,|2>)
        
        a_op = np.array([
            [0, 1, 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ], dtype=complex)
        
        a_dag = a_op.conj().T
        n_op = a_dag @ a_op  # number operator
        
        # Hamiltonian: H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
        
        def liouvillian(t, rho_flat):
            """
            Liouvillian superoperator for the master equation.
            d rho/dt = -i[H, rho] + L1(rho) + L2(rho)
            where L1 involves a/sqrt(T1) and L2 involves n*sqrt(2/Tphi)
            """
            rho = rho_flat.reshape(3, 3)
            
            # Get current pulse values
            idx = int(t / dt)
            if idx >= n_steps:
                idx = n_steps - 1
            I_val = pulse_arr[idx, 0]
            Q_val = pulse_arr[idx, 1]
            
            # Build Hamiltonian
            H = detuning * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
            drive = I_val * (a_op + a_dag) + 1j * Q_val * (a_op - a_dag)
            H += 0.5 * scale * drive
            
            # Coherent evolution
            d_rho_coherent = -1j * (H @ rho - rho @ H)
            
            # Dissipation terms
            # L1: a / sqrt(T1)
            L1 = a_op / np.sqrt(T1)
            d_rho_L1 = L1 @ rho @ L1.conj().T - 0.5 * (L1.conj().T @ L1 @ rho + rho @ L1.conj().T @ L1)
            
            # L2: n * sqrt(2/Tphi)
            L2 = n_op * np.sqrt(2.0 / Tphi)
            d_rho_L2 = L2 @ rho @ L2.conj().T - 0.5 * (L2.conj().T @ L2 @ rho + rho @ L2.conj().T @ L2)
            
            d_rho = d_rho_coherent + d_rho_L1 + d_rho_L2
            return d_rho.ravel()
        
        # Initial states
        states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
            (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
            (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
            (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2)  # (|0>-i|1>)/sqrt(2)
        ]
        
        fidelities = []
        for state in states:
            rho0 = np.outer(state, state.conj())
            rho0_flat = rho0.ravel()
            
            sol = solve_ivp(liouvillian, [0, total_time], rho0_flat, method='DOP853', rtol=1e-10, atol=1e-12)
            if not sol.success:
                return 0.0  # Failure
            
            rho_final = sol.y[:, -1].reshape(3, 3)
            
            # Target state after X-pi gate
            # X-pi swaps |0> and |1>, leaves |2> unchanged
            # So target state is: |0> -> |1>, |1> -> |0>, etc.
            if np.allclose(state, np.array([1, 0, 0], dtype=complex)):
                target = np.array([0, 1, 0], dtype=complex)
            elif np.allclose(state, np.array([0, 1, 0], dtype=complex)):
                target = np.array([1, 0, 0], dtype=complex)
            else:
                # For superpositions, apply X-pi
                target = state.copy()
                target[0], target[1] = target[1], target[0]
            
            target_rho = np.outer(target, target.conj())
            
            # Fidelity
            fidelity = np.real(np.trace(rho_final @ target_rho))
            fidelity = max(0.0, min(1.0, fidelity))
            fidelities.append(fidelity)
        
        mean_fidelity = np.mean(fidelities)
        infidelity = 1.0 - mean_fidelity
        return infidelity
    
    # Evaluate seed
    detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    def total_score(pulse_arr):
        """Compute the score: mean + max of infidelities over 9 cases"""
        infidelities = []
        for det in detunings:
            for sc in scales:
                inf = compute_fidelity(pulse_arr, det, sc)
                infidelities.append(inf)
        return np.mean(infidelities) + np.max(infidelities)
    
    # Try a few simple improvements
    best_pulse = pulse_array.copy()
    best_score = total_score(best_pulse)
    
    # Try adjusting the Gaussian width
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second deadline
    
    for sigma_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        sigma_test = sigma * sigma_mult
        gauss_test = np.exp(-0.5 * ((t - center) / sigma_test) ** 2)
        area_test = np.sum(gauss_test) * dt
        scale_factor_test = np.pi / area_test
        I_test = scale_factor_test * gauss_test
        Q_test = np.zeros(n_steps)
        
        # Apply amplitude bound
        amp_test = np.sqrt(I_test**2 + Q_test**2)
        mask = amp_test > amp_bound
        if np.any(mask):
            I_test[mask] *= amp_bound / amp_test[mask]
            Q_test[mask] *= amp_bound / amp_test[mask]
        
        # Apply slew rate constraint
        if np.sqrt(I_test[0]**2 + Q_test[0]**2) / dt > slew_bound:
            I_test[0] *= slew_bound * dt / np.sqrt(I_test[0]**2 + Q_test[0]**2)
            Q_test[0] *= slew_bound * dt / np.sqrt(I_test[0]**2 + Q_test[0]**2)
        
        for i in range(n_steps - 1):
            dI = I_test[i+1] - I_test[i]
            dQ = Q_test[i+1] - Q_test[i]
            dist = np.sqrt(dI**2 + dQ**2)
            if dist / dt > slew_bound:
                factor = slew_bound * dt / dist
                I_test[i+1] = I_test[i] + dI * factor
                Q_test[i+1] = Q_test[i] + dQ * factor
        
        if np.sqrt(I_test[-1]**2 + Q_test[-1]**2) / dt > slew_bound:
            I_test[-1] *= slew_bound * dt / np.sqrt(I_test[-1]**2 + Q_test[-1]**2)
            Q_test[-1] *= slew_bound * dt / np.sqrt(I_test[-1]**2 + Q_test[-1]**2)
        
        pulse_test = np.column_stack([I_test, Q_test])
        score_test = total_score(pulse_test)
        
        if score_test < best_score:
            best_score = score_test
            best_pulse = pulse_test
    
    return best_pulse
