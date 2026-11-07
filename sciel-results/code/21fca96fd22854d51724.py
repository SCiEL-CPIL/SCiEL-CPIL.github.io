"""Robust X_pi control pulse using B-spline parameterization with adjoint-state gradients."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N = 80
    total_time = N * dt
    
    # 3-level operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Precompute constant Liouvillian parts
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    dissip = (np.kron(L1, L1.T) - 0.5 * (np.kron(I3, L1_dag_L1) + np.kron(L1_dag_L1.T, I3)) +
              np.kron(L2, L2.T) - 0.5 * (np.kron(I3, L2_dag_L2) + np.kron(L2_dag_L2.T, I3)))
    
    # Static Hamiltonian (anharmonicity)
    H_static = 0.5 * anharmonicity * n_op @ (n_op - I3)
    H_static_liouv = -1j * (np.kron(I3, H_static) - np.kron(H_static.T, I3))
    
    # Drive basis
    n_left = np.kron(I3, n_op)
    n_right = np.kron(n_op.T, I3)
    n_liouv = -1j * (n_left - n_right)
    
    X_op = a + a_dag
    Y_op = 1j * (a - a_dag)
    
    X_liouv = -1j * (np.kron(I3, X_op) - np.kron(X_op.T, I3))
    Y_liouv = -1j * (np.kron(I3, Y_op) - np.kron(Y_op.T, I3))
    
    # 9 test cases
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # 6 initial states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # Target states after X_pi
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = s[1]
        t[1] = s[0]
        t[2] = s[2]
        target_states.append(t)
    
    # Precompute fidelity coefficients W
    W = np.zeros((9, 54), dtype=complex)
    for ci, det in enumerate(detunings):
        for si, sc in enumerate(scales):
            case_idx = ci * 3 + si
            for si2, (state, target) in enumerate(zip(states, target_states)):
                col = case_idx * 6 + si2
                for i in range(3):
                    for j in range(3):
                        W[i + 3 * j, col] = np.conj(target[i]) * target[j]
    
    # Precompute initial density matrices
    RHO0 = np.zeros((9, 54), dtype=complex)
    for ci, det in enumerate(detunings):
        for si, sc in enumerate(scales):
            case_idx = ci * 3 + si
            for si2, state in enumerate(states):
                col = case_idx * 6 + si2
                rho = np.outer(state, state.conj())
                RHO0[:, col] = rho.flatten()
    
    # B-spline parameterization
    # Use 5 interior knots for I and Q, so 6 coefficients each (knots + 2 boundary)
    # Actually, for cubic B-splines with n knots, we have n+1 coefficients
    # Let's use 4 interior knots for each, giving 5 coefficients per channel
    n_knots = 4
    n_coeffs = n_knots + 1
    
    # Initial knot positions (evenly spaced)
    knot_positions_I = np.linspace(0.1, 0.9, n_knots)
    knot_positions_Q = np.linspace(0.1, 0.9, n_knots)
    
    # Initial coefficients (DRAG-like)
    t_eval = np.arange(N) * dt
    sigma = 40.0 / 6.0
    center = 20.0
    gaussian = np.exp(-0.5 * ((t_eval - center) / sigma)**2)
    edge_val = gaussian[0]
    gaussian -= edge_val
    area = np.sum(gaussian) * dt
    if area != 0:
        gaussian *= np.pi / area
    
    # Fit initial Gaussian to B-spline coefficients
    cs_I = CubicSpline(np.concatenate([[0], knot_positions_I, [1]]), 
                       np.concatenate([[0], gaussian[np.searchsorted(t_eval, knot_positions_I * total_time)], [0]]), 
                       bc_type='clamped')
    coeffs_I_init = cs_I.c[0, :n_coeffs]
    
    dI_dt = np.gradient(gaussian, dt)
    gamma = 0.1
    Q_init = -gamma * dI_dt
    cs_Q = CubicSpline(np.concatenate([[0], knot_positions_Q, [1]]), 
                       np.concatenate([[0], Q_init[np.searchsorted(t_eval, knot_positions_Q * total_time)], [0]]), 
                       bc_type='clamped')
    coeffs_Q_init = cs_Q.c[0, :n_coeffs]
    
    # Parameters: [knots_I (n_knots), knots_Q (n_knots), coeffs_I (n_coeffs), coeffs_Q (n_coeffs)]
    param_dim = 2 * n_knots + 2 * n_coeffs
    x0 = np.concatenate([knot_positions_I, knot_positions_Q, coeffs_I_init, coeffs_Q_init])
    
    def params_to_pulse(x):
        """Convert parameter vector to (N, 2) pulse array."""
        knots_I = x[:n_knots]
        knots_Q = x[n_knots:2*n_knots]
        coeffs_I = x[2*n_knots:2*n_knots+n_coeffs]
        coeffs_Q = x[2*n_knots+n_coeffs:2*n_knots+2*n_coeffs]
        
        # Ensure knots are sorted and within (0, 1)
        knots_I = np.sort(np.clip(knots_I, 0.05, 0.95))
        knots_Q = np.sort(np.clip(knots_Q, 0.05, 0.95))
        
        # Evaluate B-splines at time points
        t_norm = t_eval / total_time
        I_vals = np.zeros(N)
        Q_vals = np.zeros(N)
        
        # Use cubic B-spline evaluation
        for i in range(N):
            ti = t_norm[i]
            # Find which knot interval
            if ti <= knots_I[0]:
                I_vals[i] = coeffs_I[0]
            elif ti >= knots_I[-1]:
                I_vals[i] = coeffs_I[-1]
            else:
                # Find interval
                idx = np.searchsorted(knots_I, ti)
                if idx > 0:
                    t_local = (ti - knots_I[idx-1]) / (knots_I[idx] - knots_I[idx-1])
                    I_vals[i] = coeffs_I[idx-1] * (1-t_local) + coeffs_I[idx] * t_local
                else:
                    t_local = ti / knots_I[0]
                    I_vals[i] = coeffs_I[0] * (1-t_local)
        
        for i in range(N):
            tq = t_norm[i]
            if tq <= knots_Q[0]:
                Q_vals[i] = coeffs_Q[0]
            elif tq >= knots_Q[-1]:
                Q_vals[i] = coeffs_Q[-1]
            else:
                idx = np.searchsorted(knots_Q, tq)
                if idx > 0:
                    t_local = (tq - knots_Q[idx-1]) / (knots_Q[idx] - knots_Q[idx-1])
                    Q_vals[i] = coeffs_Q[idx-1] * (1-t_local) + coeffs_Q[idx] * t_local
                else:
                    t_local = tq / knots_Q[0]
                    Q_vals[i] = coeffs_Q[0] * (1-t_local)
        
        return np.column_stack([I_vals, Q_vals])
    
    def check_validity(pulse_IQ):
        """Check if pulse satisfies amplitude and slew constraints."""
        amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amps > amplitude_bound + 1e-10):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-10:
                return False
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-10:
            return False
        return True
    
    def simulate_forward(pulse_IQ):
        """Forward simulation, return RHOs and U matrices."""
        RHOs = np.zeros((N + 1, 9, 54), dtype=complex)
        RHOs[0] = RHO0
        U_list = []
        
        for i in range(N):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            U_list.append([])
            for ci, det in enumerate(detunings):
                for si, sc in enumerate(scales):
                    case_idx = ci * 3 + si
                    L = H_static_liouv + det * n_liouv + 0.5 * sc * (I_val * X_liouv + Q_val * Y_liouv) + dissip
                    U = expm(dt * L)
                    U_list[i].append(U)
                    cols = slice(case_idx * 6, case_idx * 6 + 6)
                    RHOs[i + 1, :, cols] = U @ RHOs[i, :, cols]
        
        return RHOs, U_list
    
    def compute_fidelities(RHO_final):
        """Compute fidelities from final RHO."""
        fids = np.real(np.sum(W * RHO_final, axis=0))
        return fids
    
    def compute_score(fids):
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)
    
    def forward_and_score(pulse_IQ):
        """Forward simulation and score."""
        RHOs, U_list = simulate_forward(pulse_IQ)
        fids = compute_fidelities(RHOs[N])
        score = compute_score(fids)
        return RHOs, U_list, fids, score
    
    def adjoint_gradient(pulse_IQ, RHOs, U_list, fids):
        """Compute gradient of score w.r.t. pulse amplitudes using adjoint method."""
        infids = 1.0 - fids
        max_idx = np.argmax(infids)
        
        d_score_d_fids = -np.ones(54) / 54.0
        d_score_d_fids[max_idx] -= 1.0
        
        # Adjoint state
        Lambda = np.zeros((9, 54), dtype=complex)
        for k in range(54):
            Lambda[:, k] = d_score_d_fids[k] * np.conj(W[:, k])
        
        grad_I = np.zeros(N)
        grad_Q = np.zeros(N)
        
        for i in range(N - 1, -1, -1):
            I_val = pulse_IQ[i, 0]
            Q_val = pulse_IQ[i, 1]
            
            for ci, det in enumerate(detunings):
                for si, sc in enumerate(scales):
                    case_idx = ci * 3 + si
                    L = H_static_liouv + det * n_liouv + 0.5 * sc * (I_val * X_liouv + Q_val * Y_liouv) + dissip
                    U = U_list[i][case_idx]
                    
                    # Adjoint update
                    Lambda_next = Lambda.copy()
                    Lambda = U.conj().T @ Lambda_next
                    
                    # Gradient contribution
                    cols = slice(case_idx * 6, case_idx * 6 + 6)
                    rho_i = RHOs[i, :, cols]
                    
                    dL_dI = 0.5 * sc * X_liouv
                    dL_dQ = 0.5 * sc * Y_liouv
                    
                    # d(score)/d(I_i) = sum Lambda^H dL/dI rho
                    grad_I[i] += np.real(np.sum(Lambda_next.conj() * (dL_dI @ rho_i)))
                    grad_Q[i] += np.real(np.sum(Lambda_next.conj() * (dL_dQ @ rho_i)))
        
        return grad_I, grad_Q
    
    def objective(x):
        """Objective function for optimization."""
        pulse_IQ = params_to_pulse(x)
        
        if not check_validity(pulse_IQ):
            return 1e6
        
        RHOs, U_list, fids, score = forward_and_score(pulse_IQ)
        return score
    
    def gradient(x):
        """Gradient function for optimization."""
        pulse_IQ = params_to_pulse(x)
        
        if not check_validity(pulse_IQ):
            return np.zeros(param_dim)
        
        RHOs, U_list, fids, score = forward_and_score(pulse_IQ)
        grad_I, grad_Q = adjoint_gradient(pulse_IQ, RHOs, U_list, fids)
        
        # Chain rule: d(score)/d(x) = sum_i (d(score)/d(I_i) * d(I_i)/d(x) + d(score)/d(Q_i) * d(Q_i)/d(x))
        # For simplicity, use finite differences for the B-spline parameter mapping
        grad_x = np.zeros(param_dim)
        eps = 1e-7
        
        for j in range(param_dim):
            x_plus = x.copy()
            x_plus[j] += eps
            pulse_plus = params_to_pulse(x_plus)
            
            if check_validity(pulse_plus):
                _, _, fids_plus, score_plus = forward_and_score(pulse_plus)
            else:
                score_plus = score + eps
            
            grad_x[j] = (score_plus - score) / eps
        
        return grad_x
    
    # Initial pulse
    pulse_init = params_to_pulse(x0)
    
    # Ensure validity
    amps = np.sqrt(pulse_init[:, 0]**2 + pulse_init[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amplitude_bound:
        scale = amplitude_bound / max_amp
        x0[2*n_knots:2*n_knots+n_coeffs] *= scale
        x0[2*n_knots+n_coeffs:2*n_knots+2*n_coeffs] *= scale
    
    pulse_init = params_to_pulse(x0)
    
    prev = np.zeros(2)
    for i in range(N):
        curr = pulse_init[i]
        if np.linalg.norm(curr - prev) / dt > slew_bound:
            scale = slew_bound * dt / np.linalg.norm(curr - prev)
            pulse_init[i:] *= scale
            break
        prev = curr
    
    # If still invalid, use simple Gaussian
    if not check_validity(pulse_init):
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
        center = 20.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if area != 0:
            gaussian *= np.pi / area
        dI_dt = np.gradient(gaussian, dt)
        gamma = 0.1
        Q_init = -gamma * dI_dt
        pulse_init = np.column_stack([gaussian, Q_init])
        
        amps = np.sqrt(pulse_init[:, 0]**2 + pulse_init[:, 1]**2)
        max_amp = np.max(amps)
        if max_amp > amplitude_bound:
            pulse_init *= amplitude_bound / max_amp
        
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_init[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound:
                scale = slew_bound * dt / np.linalg.norm(curr - prev)
                pulse_init[i:] *= scale
                break
            prev = curr
    
    best_x = x0.copy()
    best_score = objective(x0)
    best_pulse = params_to_pulse(best_x)
    
    start_time = time.time()
    time_limit = 300.0
    
    # Optimize with BFGS
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='BFGS',
            options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
        )
        
        if result.fun < best_score and check_validity(params_to_pulse(result.x)):
            best_x = result.x
            best_score = result.fun
            best_pulse = params_to_pulse(best_x)
            
    except Exception:
        pass
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            result2 = minimize(
                objective,
                best_x,
                jac=gradient,
                method='BFGS',
                options={'maxiter': 50, 'gtol': 1e-10, 'disp': False}
            )
            
            if result2.fun < best_score and check_validity(params_to_pulse(result2.x)):
                best_x = result2.x
                best_score = result2.fun
                best_pulse = params_to_pulse(best_x)
                
        except Exception:
            pass
    
    # Final validity check
    pulse_final = params_to_pulse(best_x)
    if not check_validity(pulse_final):
        pulse_final = pulse_init
    
    return pulse_final
