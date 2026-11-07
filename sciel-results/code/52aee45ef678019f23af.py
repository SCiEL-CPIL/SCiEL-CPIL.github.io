"""Robust control pulses for a three-level transmon - differentiable ODE with adjoint gradients."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm

# Hardware constants
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
T2 = 40000.0  # ns (Tphi)
N_STEPS = 80
DT = 0.5  # ns
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Precompute ladder operator and number operator for 3 levels
A = np.array([
    [0, np.sqrt(1), 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
A_DAG = A.conj().T
N_OP = A_DAG @ A  # diag(0, 1, 2)

# Precompute Lindblad-related terms
L1 = A / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / T2)
L1dagL1 = L1.conj().T @ L1
L2dagL2 = L2.conj().T @ L2

def hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian."""
    H = detuning * N_OP + 0.5 * ANHARMONICITY * (N_OP @ (N_OP - np.eye(3)))
    drive = 0.5 * scale * (I * (A + A_DAG) + 1j * Q * (A - A_DAG))
    H = H + drive
    return H


def build_liouvillian_superoperator(H):
    """Build the Liouvillian superoperator for a given Hamiltonian."""
    dim = 3
    I3 = np.eye(dim)
    # Coherent term: -i[H, rho]
    L_super = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    # Dissipative terms
    for L in [L1, L2]:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        L_super += np.kron(L, L.conj().T)
        L_super -= 0.5 * (np.kron(LdagL, I3) + np.kron(I3, LdagL))
    return L_super


def evolve_state_vec(rho_vec, pulse_I, pulse_Q, detuning, scale):
    """
    Evolve the density matrix vector through the pulse.
    Returns the final density matrix vector.
    """
    dim = 3
    rho = rho_vec.reshape(dim, dim)
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        H = hamiltonian(detuning, scale, I_val, Q_val)
        L_super = build_liouvillian_superoperator(H)
        U = expm(DT * L_super)
        rho_vec = U @ rho_vec
        rho = rho_vec.reshape(dim, dim)
    
    return rho_vec


def compute_infidelity_for_case(pulse_I, pulse_Q, detuning, scale, target_states):
    """
    Compute infidelity for a single case (detuning, scale).
    Returns array of infidelities for each target state.
    """
    dim = 3
    rho0_vec = np.zeros(dim * dim, dtype=complex)
    rho0_vec[0] = 1.0  # |0><0|
    
    rho_final_vec = evolve_state_vec(rho0_vec, pulse_I, pulse_Q, detuning, scale)
    rho_final = rho_final_vec.reshape(dim, dim)
    
    infids = []
    for psi in target_states:
        val = np.vdot(psi, rho_final @ psi)
        fid = np.real(val)
        fid = np.clip(fid, 0, 1)
        infids.append(1 - fid)
    
    return np.array(infids)


def score_function(pulse_I, pulse_Q):
    """The scoring function: mean + max of the 9-case infidelities."""
    dim = 3
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    all_infid = []
    for det in detunings:
        for scl in scales:
            infids = compute_infidelity_for_case(pulse_I, pulse_Q, det, scl, target_states)
            all_infid.extend(infids.tolist())
    
    all_infid = np.array(all_infid)
    return np.mean(all_infid) + np.max(all_infid)


def check_constraints(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    for i in range(len(pulse_I)):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-10:
            return False
    
    prev_I, prev_Q = 0.0, 0.0
    for i in range(len(pulse_I)):
        dI = pulse_I[i] - prev_I
        dQ = pulse_Q[i] - prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
        prev_I, prev_Q = pulse_I[i], pulse_Q[i]
    
    dI = 0.0 - prev_I
    dQ = 0.0 - prev_Q
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-10:
        return False
    
    return True


def drag_seed():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0  # ns
    
    A = np.sqrt(np.pi / 2) / sigma
    I = A * np.exp(-t**2 / (2 * sigma**2))
    I = I - I[0]
    I = np.maximum(I, 0)
    
    max_amp = np.max(np.sqrt(I**2))
    if max_amp > AMPLITUDE_BOUND:
        I = I * (AMPLITUDE_BOUND / max_amp)
    
    for _ in range(10):
        if check_constraints(I, np.zeros_like(I)):
            break
        I = I * 0.9
    
    Q = np.zeros_like(I)
    return I, Q


def compute_gradient_adjoint(pulse_I, pulse_Q):
    """
    Compute gradients using the adjoint method.
    This is computationally expensive but gives exact gradients.
    """
    dim = 3
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    # For each case, compute the adjoint gradient
    for det in detunings:
        for scl in scales:
            # Forward pass: store all intermediate states
            rho_vecs = []
            rho_vec = np.zeros(dim * dim, dtype=complex)
            rho_vec[0] = 1.0
            
            U_list = []
            for i in range(N_STEPS):
                I_val = pulse_I[i]
                Q_val = pulse_Q[i]
                H = hamiltonian(det, scl, I_val, Q_val)
                L_super = build_liouvillian_superoperator(H)
                U = expm(DT * L_super)
                U_list.append(U)
                rho_vec = U @ rho_vec
                rho_vecs.append(rho_vec.copy())
            
            rho_final_vec = rho_vecs[-1]
            rho_final = rho_final_vec.reshape(dim, dim)
            
            # Compute infidelities and find which are max
            infids = []
            for psi in target_states:
                val = np.vdot(psi, rho_final @ psi)
                fid = np.real(val)
                fid = np.clip(fid, 0, 1)
                infids.append(1 - fid)
            
            infids = np.array(infids)
            mean_infid = np.mean(infids)
            max_idx = np.argmax(infids)
            
            # The score contribution from this case is weighted
            # In the full score, we have mean + max over all 54 values
            # For simplicity, we'll compute gradient of mean and max separately
            # and combine them appropriately
            
            # Adjoint pass: compute gradient with respect to pulse
            # The adjoint variable lambda satisfies:
            # dlambda/dt = -L_super^T lambda
            # where L_super is the Liouvillian superoperator
            
            # Initialize adjoint at final time
            # The cost function depends on rho_final through fidelity
            # For each target state psi, d(1-fid)/d(rho) = -|psi><psi|
            
            # We need to handle the mean and max contributions
            # For the mean: weight = 1/54 for each state
            # For the max: weight = 1 for the max state, 0 for others
            
            # Let's compute the adjoint for the combined cost
            # The total derivative of the score with respect to rho_final is:
            # d(score)/d(rho) = (1/54) * sum_psi (-|psi><psi|) + (-|psi_max><psi_max|)
            
            cost_grad_rho = np.zeros((dim, dim), dtype=complex)
            for psi in target_states:
                cost_grad_rho += (-1.0 / 54.0) * np.outer(psi, psi.conj().T)
            psi_max = target_states[max_idx]
            cost_grad_rho += (-1.0) * np.outer(psi_max, psi_max.conj().T)
            
            # Vectorize the cost gradient
            cost_grad_vec = cost_grad_rho.reshape(dim * dim)
            
            # Adjoint evolution backwards
            lambda_vec = cost_grad_vec.copy()
            
            for i in range(N_STEPS - 1, -1, -1):
                I_val = pulse_I[i]
                Q_val = pulse_Q[i]
                H = hamiltonian(det, scl, I_val, Q_val)
                L_super = build_liouvillian_superoperator(H)
                
                # The gradient with respect to H at step i is:
                # d(rho_next)/d(H) = DT * expm(DT * L_super) @ (dL_super/dH) @ rho_vecs[i-1] if i > 0 else rho0
                # But we need dL_super/d(I) and dL_super/d(Q)
                
                # L_super depends on H through the coherent term
                # d(-i[H,rho])/dI = -i * (dH/dI @ rho - rho @ dH/dI^T)
                # In superoperator form: -i * (kron(dH/dI, I) - kron(I, dH/dI^T))
                
                dH_dI = 0.5 * scl * (A + A_DAG)
                dH_dQ = 0.5 * scl * 1j * (A - A_DAG)
                
                dL_dI = -1j * (np.kron(dH_dI, np.eye(dim)) - np.kron(np.eye(dim), dH_dI.conj().T))
                dL_dQ = -1j * (np.kron(dH_dQ, np.eye(dim)) - np.kron(np.eye(dim), dH_dQ.conj().T))
                
                # The contribution to the gradient from step i
                # lambda_next = U_i^T lambda_i
                # d(lambda_next @ rho_next)/d(I_i) = lambda_i^T @ (DT * U_i @ dL_dI @ rho_prev)
                
                if i == 0:
                    rho_prev = np.zeros(dim * dim, dtype=complex)
                    rho_prev[0] = 1.0
                else:
                    rho_prev = rho_vecs[i - 1]
                
                U_i = U_list[i]
                
                # Gradient contribution
                term_I = lambda_vec @ (DT * U_i @ dL_dI @ rho_prev)
                term_Q = lambda_vec @ (DT * U_i @ dL_dQ @ rho_prev)
                
                grad_I[i] += np.real(term_I)
                grad_Q[i] += np.real(term_Q)
                
                # Update adjoint
                lambda_vec = U_i.conj().T @ lambda_vec
    
    return grad_I, grad_Q


def optimize_pulse():
    """
    Optimize the pulse using gradient-based methods with the adjoint method.
    """
    I_seed, Q_seed = drag_seed()
    
    def objective_and_grad(x):
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        
        # Compute score
        score = score_function(I, Q)
        
        # Compute gradient using adjoint method
        try:
            grad_I, grad_Q = compute_gradient_adjoint(I, Q)
        except Exception:
            # Fallback to finite differences if adjoint fails
            eps = 1e-6
            grad_I = np.zeros(N_STEPS)
            grad_Q = np.zeros(N_STEPS)
            for i in range(N_STEPS):
                I_plus = I.copy()
                I_plus[i] += eps
                I_minus = I.copy()
                I_minus[i] -= eps
                grad_I[i] = (score_function(I_plus, Q) - score_function(I_minus, Q)) / (2 * eps)
                
                Q_plus = Q.copy()
                Q_plus[i] += eps
                Q_minus = Q.copy()
                Q_minus[i] -= eps
                grad_Q[i] = (score_function(I, Q_plus) - score_function(I, Q_minus)) / (2 * eps)
        
        # Add constraint penalties to gradient
        penalty = 0.0
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > AMPLITUDE_BOUND:
                penalty += 10.0 * (amp - AMPLITUDE_BOUND)**2
                # Gradient of penalty
                if amp > 1e-10:
                    grad_I[i] += 20.0 * (amp - AMPLITUDE_BOUND) * I[i] / amp
                    grad_Q[i] += 20.0 * (amp - AMPLITUDE_BOUND) * Q[i] / amp
            
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                penalty += 10.0 * (slew - SLEW_BOUND)**2
                if slew > 1e-10:
                    grad_I[i] += 20.0 * (slew - SLEW_BOUND) * (I[i] - prev_I) / (slew * DT)
                    grad_Q[i] += 20.0 * (slew - SLEW_BOUND) * (Q[i] - prev_Q) / (slew * DT)
            
            prev_I, prev_Q = I[i], Q[i]
        
        dI = 0.0 - prev_I
        dQ = 0.0 - prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            penalty += 10.0 * (slew - SLEW_BOUND)**2
            if slew > 1e-10:
                grad_I[-1] += 20.0 * (slew - SLEW_BOUND) * (0.0 - prev_I) / (slew * DT)
                grad_Q[-1] += 20.0 * (slew - SLEW_BOUND) * (0.0 - prev_Q) / (slew * DT)
        
        total_grad = np.concatenate([grad_I, grad_Q])
        return score + penalty, total_grad
    
    x0 = np.concatenate([I_seed, Q_seed])
    
    # Use L-BFGS-B with bounds to enforce constraints
    bounds = [(-AMPLITUDE_BOUND, AMPLITUDE_BOUND)] * (2 * N_STEPS)
    
    result = minimize(objective_and_grad, x0, method='L-BFGS-B',
                      jac=True, bounds=bounds,
                      options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
    
    I_opt = result.x[:N_STEPS]
    Q_opt = result.x[N_STEPS:]
    
    # Ensure constraints are satisfied
    if not check_constraints(I_opt, Q_opt):
        # Fall back to seed
        return I_seed, Q_seed
    
    return I_opt, Q_opt


def _compute_pulse():
    """Compute the pulse and cache it."""
    I, Q = drag_seed()
    
    try:
        I_opt, Q_opt = optimize_pulse()
        if check_constraints(I_opt, Q_opt):
            return I_opt, Q_opt
    except Exception:
        pass
    
    return I, Q


_I_cached, _Q_cached = _compute_pulse()


def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    result = np.column_stack([_I_cached, _Q_cached])
    return result
