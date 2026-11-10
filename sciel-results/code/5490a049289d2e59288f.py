import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import math
import time

def pulse(target_gate, hardware_model):
    # Extract parameters from hardware_model if available, else use defaults from task description
    # Task says hardware_model holds these values.
    # We assume the structure is a dict or similar. If not, we fall back to constants.
    try:
        chi = hardware_model.get('anharmonicity', -2 * math.pi * 0.25)
        amp_bound = hardware_model.get('amplitude_bound', 0.4)
        slew_bound = hardware_model.get('slew_bound', 0.12)
        detuning_bounds = hardware_model.get('detuning_bounds', [-2 * math.pi * 0.002, 2 * math.pi * 0.002])
        scale_bounds = hardware_model.get('scale_bounds', [0.97, 1.03])
        T1 = hardware_model.get('T1', 25000.0)
        Tphi = hardware_model.get('Tphi', 40000.0)
    except:
        chi = -2 * math.pi * 0.25
        amp_bound = 0.4
        slew_bound = 0.12
        detuning_bounds = [-2 * math.pi * 0.002, 2 * math.pi * 0.002]
        scale_bounds = [0.97, 1.03]
        T1 = 25000.0
        Tphi = 40000.0

    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    
    # Precompute static parts of Liouvillian?
    # L(rho) = -i [H0, rho] + ...
    # H0 depends on detuning.
    
    # Lindblad operators
    L1 = a / math.sqrt(T1)
    L2 = n * math.sqrt(2.0 / Tphi)
    
    # Precompute dissipator matrices (independent of I, Q, detuning)
    # D[L](rho) = L rho L^dag - 0.5 L^dag L rho - 0.5 rho L^dag L
    def get_dissipator_matrix(L_op):
        L_dag = L_op.conj().T
        L_dag_L = L_dag @ L_op
        return np.kron(L_op.T, L_op) - 0.5 * np.kron(I3, L_dag_L) - 0.5 * np.kron(L_dag_L.T, I3)
    
    D1_mat = get_dissipator_matrix(L1)
    D2_mat = get_dissipator_matrix(L2)
    D_total = D1_mat + D2_mat

    # Precompute n and n(n-1) matrices
    n_mat = n
    nn_minus_1 = n @ (n - I3)
    
    # Hamiltonian parts
    # H = delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I(a+a_dag) + i Q(a-a_dag))
    # H_drive = 0.5 * scale * (I * (a+a_dag) + i Q * (a-a_dag))
    # Let A_plus = a + a_dag, A_minus = a - a_dag
    A_plus = a + a_dag
    A_minus = a - a_dag
    
    def simulate_pulse(I, Q, detuning, scale):
        # I, Q are arrays of length n_steps
        rho_vec = np.zeros(9, dtype=complex)
        rho_vec[0] = 1.0 # |0><0|
        
        for k in range(n_steps):
            I_k = I[k]
            Q_k = Q[k]
            
            # H = delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I_k * A_plus + 1j * Q_k * A_minus)
            H = (detuning * n_mat + 0.5 * chi * nn_minus_1 + 
                 0.5 * scale * (I_k * A_plus + 1j * Q_k * A_minus))
            
            # Liouvillian
            # L = -i (I \otimes H - H^T \otimes I) + D_total
            L_mat = -1j * (np.kron(I3, H) - np.kron(H.T, I3)) + D_total
            
            # Evolve
            # Use expm
            U = sla.expm(L_mat * dt)
            rho_vec = U @ rho_vec
        
        return rho_vec.reshape(3, 3)

    def calculate_score(I, Q):
        # 9 cases
        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        # Initial states
        psi_0 = np.array([1, 0, 0], dtype=complex)
        psi_1 = np.array([0, 1, 0], dtype=complex)
        psi_plus = np.array([1, 1, 0], dtype=complex) / math.sqrt(2)
        psi_minus = np.array([1, -1, 0], dtype=complex) / math.sqrt(2)
        psi_i_plus = np.array([1, 1j, 0], dtype=complex) / math.sqrt(2)
        psi_i_minus = np.array([1, -1j, 0], dtype=complex) / math.sqrt(2)
        
        # Target states (X gate applied)
        # X|0> = |1>, X|1> = |0>, X(|0>+|1>)/sqrt2 = (|1>+|0>)/sqrt2 = same
        # X(|0>-|1>)/sqrt2 = (|1>-|0>)/sqrt2 = -psi_minus
        # X(|0>+i|1>)/sqrt2 = (|1>+i|0>)/sqrt2 = i (|0>-i|1>)/sqrt2?
        # Let's just compute X @ psi
        X_mat = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
        
        target_states = [
            X_mat @ psi_0,
            X_mat @ psi_1,
            X_mat @ psi_plus,
            X_mat @ psi_minus,
            X_mat @ psi_i_plus,
            X_mat @ psi_i_minus
        ]
        
        initial_states = [psi_0, psi_1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
        
        infidelities = []
        
        for delta in detunings:
            for s in scales:
                # Simulate all 6 initial states?
                # We can simulate the evolution operator U for the pulse,
                # then apply it to each initial state.
                # But the simulation above evolves a density matrix.
                # It's more efficient to evolve the state vector if we assume pure states?
                # The task says "Lindblad operators... exponentiates the Liouvillian".
                # So we must use density matrices.
                
                # However, we can simulate the superoperator U_total = prod(exp(L_k dt))
                # Then rho_final = U_total @ rho_initial
                # This saves 6x simulation time per hardware case.
                
                # Let's refactor simulate to return the superoperator
                pass

        # Refactored simulation for speed
        def get_superoperator(I, Q, detuning, scale):
            U_total = np.eye(9, dtype=complex)
            for k in range(n_steps):
                I_k = I[k]
                Q_k = Q[k]
                H = (detuning * n_mat + 0.5 * chi * nn_minus_1 + 
                     0.5 * scale * (I_k * A_plus + 1j * Q_k * A_minus))
                L_mat = -1j * (np.kron(I3, H) - np.kron(H.T, I3)) + D_total
                U_step = sla.expm(L_mat * dt)
                U_total = U_step @ U_total
            return U_total

        for delta in detunings:
            for s in scales:
                U_tot = get_superoperator(I, Q, delta, s)
                
                case_infid = 0.0
                for i in range(6):
                    rho_init = np.outer(initial_states[i], initial_states[i].conj()).reshape(9)
                    rho_final_vec = U_tot @ rho_init
                    rho_final = rho_final_vec.reshape(3, 3)
                    
                    target = target_states[i]
                    F = np.real(np.vdot(target, rho_final @ target))
                    infid = 1.0 - F
                    case_infid += infid
                
                case_infid /= 6.0
                infidelities.append(case_infid)
        
        return np.mean(infidelities) + np.max(infidelities)

    def generate_pulse(params):
        sigma, alpha = params
        if sigma <= 0:
            return np.zeros((n_steps, 2))
        
        t = (np.arange(n_steps) - (n_steps - 1) / 2.0) * dt
        # Area = pi
        A = math.sqrt(math.pi) / sigma
        I = A * np.exp(-t**2 / sigma**2)
        # dI/dt = A * exp(...) * (-2t/sigma^2)
        dIdt = -2 * A * t / sigma**2 * np.exp(-t**2 / sigma**2)
        Q = -alpha * dIdt
        
        pulse_arr = np.column_stack((I, Q))
        
        # Check constraints
        # Amplitude
        mags = np.sqrt(I**2 + Q**2)
        if np.max(mags) > amp_bound:
            # Scale down
            scale_factor = amp_bound / np.max(mags)
            I *= scale_factor
            Q *= scale_factor
            pulse_arr = np.column_stack((I, Q))
            # Re-check area? If scaled, area is not pi.
            # The task says "area pi" for the seed.
            # If we scale down, we lose area.
            # It's better to optimize sigma such that constraints are met without scaling.
            # If scaling is needed, it's a penalty.
            pass
            
        # Slew rate
        # ||p[i+1] - p[i]|| / dt <= slew_bound
        # p is (I, Q)
        diffs = np.diff(pulse_arr, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        max_slew = np.max(norms) / dt
        # Also check from 0 to first and last to 0
        first_norm = np.linalg.norm(pulse_arr[0]) / dt
        last_norm = np.linalg.norm(pulse_arr[-1]) / dt
        max_slew = max(max_slew, first_norm, last_norm)
        
        if max_slew > slew_bound:
            # Penalty
            pass

        return pulse_arr

    def objective(params):
        pulse_arr = generate_pulse(params)
        I = pulse_arr[:, 0]
        Q = pulse_arr[:, 1]
        
        # Check constraints strictly
        mags = np.sqrt(I**2 + Q**2)
        if np.max(mags) > amp_bound + 1e-9:
            return 1.0 # High penalty
        
        diffs = np.diff(pulse_arr, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        max_slew = np.max(norms) / dt
        first_norm = np.linalg.norm(pulse_arr[0]) / dt
        last_norm = np.linalg.norm(pulse_arr[-1]) / dt
        max_slew = max(max_slew, first_norm, last_norm)
        
        if max_slew > slew_bound + 1e-9:
            return 1.0 # High penalty
            
        return calculate_score(I, Q)

    # Optimization
    # Start with seed-like parameters
    # Seed: sigma = 40/6 = 6.66, alpha = ?
    # Standard DRAG alpha is often ~ 0.1 - 0.5
    x0 = [6.66, 0.2]
    
    bounds = [(2.0, 12.0), (-1.0, 1.0)]
    
    start_time = time.time()
    best_score = float('inf')
    best_params = x0
    
    # Use Nelder-Mead or Powell
    # Powell is robust for unconstrained
    res = opt.minimize(objective, x0, method='Powell', bounds=bounds, 
                        options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if res.success and res.fun < best_score:
        best_params = res.x
        best_score = res.fun
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 300:
        # Return best found
        pass
    
    I, Q = generate_pulse(best_params).T
    return np.column_stack((I, Q))

pulse("X_pi", {})
