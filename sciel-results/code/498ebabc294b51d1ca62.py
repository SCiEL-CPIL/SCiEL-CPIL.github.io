import numpy as np
from scipy.linalg import expm
import time
from scipy.optimize import minimize, differential_evolution

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
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
    
    # The 6 states as specified
    sqrt2 = np.sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2
    ]
    
    # Target states for X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([-1, 1, 0], dtype=complex) / sqrt2,
        np.array([1j, 1, 0], dtype=complex) / sqrt2,
        np.array([-1j, 1, 0], dtype=complex) / sqrt2
    ]
    
    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Precompute constant parts
    H_anharm = 0.5 * alpha * (n_op @ (n_op - np.eye(3)))
    
    # Dissipator superoperator (constant)
    dim = 3
    d2 = dim * dim
    L_diss_super = np.zeros((d2, d2), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L1_term = np.kron(L_op, L_op.conj())
        L2_term = 0.5 * (np.kron(L_dag @ L_op, np.eye(dim)) + np.kron(np.eye(dim), (L_dag @ L_op).T))
        L_diss_super += L1_term - L2_term
    
    # Precompute identity for kron
    I3 = np.eye(3)
    
    def simulate_pulse(I_Q, delta, scale):
        """
        Simulates the pulse and returns mean infidelity over 6 states.
        """
        # Precompute Liouvillian steps
        L_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
            H = delta * n_op + H_anharm + drive
            H_term = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
            L_total = H_term + L_diss_super
            L_steps.append(expm(L_total * dt))
        
        # Simulate all 6 states
        total_infidelity = 0.0
        for idx in range(6):
            psi0 = states[idx]
            rho0 = np.outer(psi0, psi0.conj())
            rho_vec = rho0.flatten()
            
            for i in range(N):
                rho_vec = L_steps[i] @ rho_vec
            
            rho = rho_vec.reshape(3, 3)
            target = targets[idx]
            proj = np.outer(target, target.conj())
            fid = np.real(np.trace(rho @ proj))
            fid = np.clip(fid, 0, 1)
            total_infidelity += (1.0 - fid)
        
        return total_infidelity / 6.0
    
    def check_constraints(I_Q):
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-12:
            return False
        # Slew
        if np.sqrt(I_Q[0,0]**2 + I_Q[0,1]**2) / dt > slew_bound + 1e-12:
            return False
        if np.sqrt(I_Q[-1,0]**2 + I_Q[-1,1]**2) / dt > slew_bound + 1e-12:
            return False
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-12:
            return False
        return True
    
    def calculate_score(I_Q):
        """
        Calculates the score: mean + max of the 9 cases.
        """
        case_scores = []
        for delta in deltas:
            for scale in scales:
                mean_inf = simulate_pulse(I_Q, delta, scale)
                # We need max infidelity too, but simulate_pulse returns mean.
                # Let's modify to return both or recompute.
                # For speed, let's just use mean as proxy first, or modify simulate_pulse.
                # Actually, the task says "infidelity = 1 - mean fidelity... Score = mean + max of the 9."
                # This implies for each case, we compute infidelity (which is mean fidelity deficit?).
                # "infidelity = 1 - mean fidelity to X over [states]"
                # So infidelity_case = 1 - (1/6) * sum(fid_i)
                # Then Score = mean(infidelity_case) + max(infidelity_case)
                # My simulate_pulse returns mean_inf which is infidelity_case.
                case_scores.append(mean_inf)
        
        return np.mean(case_scores) + np.max(case_scores)
    
    def generate_pulse(sigma, eta, center=20.0):
        t = np.linspace(0, 40, N)
        A = np.sqrt(np.pi / 2) / sigma
        I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
        dI = -(t - center) / (sigma**2) * I
        Q = eta * dI
        I_Q = np.column_stack((I, Q))
        return I_Q
    
    # Seed pulse
    sigma_seed = 40.0 / 6.0
    I_Q_seed = generate_pulse(sigma_seed, 0.0)
    
    # Check seed
    if not check_constraints(I_Q_seed):
        # Adjust seed if necessary
        pass
    
    best_I_Q = I_Q_seed.copy()
    best_score = calculate_score(best_I_Q)
    
    # Objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e10
        sigma, eta = params
        if sigma <= 0.5 or sigma > 50:
            return 1e10
        I_Q = generate_pulse(sigma, eta)
        if not check_constraints(I_Q):
            return 1e10
        return calculate_score(I_Q)
    
    # Phase 1: Coarse Grid Search
    # Time budget: 50s
    grid_deadline = start_time + 50.0
    
    sigmas = [5, 6, 7, 8, 9, 10, 12, 15]
    etas = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    
    for s in sigmas:
        for e in etas:
            if time.time() > grid_deadline:
                break
            score = objective([s, e])
            if score < best_score:
                best_score = score
                best_I_Q = generate_pulse(s, e)
        if time.time() > grid_deadline:
            break
    
    # Phase 2: Local Optimization (Nelder-Mead)
    # Time budget: 100s
    local_deadline = start_time + 150.0
    
    # Start from best grid point
    # Try a few local optimizations from different starting points
    start_points = [
        [sigma_seed, 0.0],
        [8.0, -1.0],
        [10.0, -2.0],
        [best_I_Q.shape[0] and 8.0, 0.0] # dummy, just to fill list
    ]
    
    # Identify the best grid point parameters if we tracked them, otherwise use defaults
    # Let's just run NM from the current best if we can infer params, or from a few good guesses.
    # Since we don't track which (s,e) gave best_score in the loop above easily without extra vars,
    # let's just run NM from a few diverse points.
    
    nm_starts = [
        [8.0, -1.0],
        [10.0, -2.0],
        [6.0, -0.5],
        [12.0, -1.5]
    ]
    
    for sp in nm_starts:
        if time.time() > local_deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_I_Q = generate_pulse(res.x[0], res.x[1])
        except Exception as e:
            pass
    
    # Phase 3: Global Optimization (Differential Evolution) - if time permits
    # Time budget: remaining - 10s safety
    global_deadline = deadline - 10.0
    
    if time.time() < global_deadline:
        try:
            bounds = [(5, 20), (-5, 5)]
            # Use a small population for speed
            res_de = differential_evolution(objective, bounds, maxiter=20, popsize=10, 
                                            tol=1e-4, atol=1e-5, seed=42, polish=False,
                                            update_interval=1)
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_I_Q = generate_pulse(res_de.x[0], res_de.x[1])
        except Exception as e:
            pass
    
    # Final Polish: Very tight local optimization on the best found
    if time.time() < deadline - 5:
        # Infer best params? We don't have them directly if best came from DE or Grid.
        # Let's just run NM on the best_I_Q directly? No, NM needs params.
        # Let's assume the best pulse is close to a (sigma, eta) form.
        # We can try to fit or just run NM from a few points near the best known good ones.
        # Actually, if best_I_Q is from DE, we have res_de.x.
        # If from Grid, we lost the params.
        # Let's just run a final NM from a few generic good points to see if we can squeeze more.
        
        # Let's try to recover params from best_I_Q roughly?
        # Or just run NM from the best grid point we likely found (around 8-10, -1 to -2).
        
        final_starts = [
            [8.0, -1.0],
            [9.0, -1.5],
            [10.0, -2.0]
        ]
        
        for sp in final_starts:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective, sp, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_I_Q = generate_pulse(res.x[0], res.x[1])
            except Exception as e:
                pass
    
    # Final Constraint Check
    if not check_constraints(best_I_Q):
        best_I_Q = I_Q_seed.copy()
    
    return best_I_Q
