"""Robust X-pi gate pulse for 3-level transmon using basis expansion and optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _build_hamiltonian(detuning, scale, I, Q, anharmonicity):
    """Build the 3x3 Hamiltonian for the transmon system."""
    # a is the lowering operator for 3 levels
    # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    
    # n = a^dagger @ a (number operator)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # H = detuning*n + 0.5*anharmonicity*n*(n-1) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
    H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    return H

def _build_liouvillian(H, T1, Tphi):
    """Build the Liouvillian superoperator for the Lindblad master equation."""
    dim = H.shape[0]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Build Liouvillian: dρ/dt = -i[H,ρ] + sum_k (L_k ρ L_k† - 0.5{L_k† L_k, ρ})
    # Vectorize: ρ -> vec(ρ), Liouvillian acts on vec(ρ)
    # -i[H,ρ] -> -i(I⊗H† - H⊗I)vec(ρ) in column-stacking convention
    # Actually, using row-major: vec(AρB) = (B^T ⊗ A)vec(ρ)
    # [H,ρ] = Hρ - ρH -> (I⊗H - H^T⊗I)vec(ρ)
    
    I3 = np.eye(3)
    L_comm = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    for L in [L1, L2]:
        L_dag = L.conj().T
        # LρL† -> (L†^T ⊗ L)vec(ρ) = (L.conj() ⊗ L)vec(ρ)
        # 0.5{L†L, ρ} -> 0.5(L†L ρ + ρ L†L) -> 0.5(I⊗L†L - (L†L)^T⊗I)
        L_dagL = L_dag @ L
        L_lind = np.kron(L.conj(), L) - 0.5 * (np.kron(I3, L_dagL) - np.kron(L_dagL.T, I3))
        L_comm += L_lind
    
    return L_comm

def _evolve_state(rho_init, pulse_IQ, detuning, scale, anharmonicity, T1, Tphi, dt):
    """Evolve the density matrix through the pulse using the Liouvillian."""
    n_steps = len(pulse_IQ)
    rho = rho_init.copy()
    
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        
        H = _build_hamiltonian(detuning, scale, I_val, Q_val, anharmonicity)
        L = _build_liouvillian(H, T1, Tphi)
        
        # Exponentiate: rho_next = exp(L*dt) vec(rho)
        # Use scipy.linalg.expm for the matrix exponential
        from scipy.linalg import expm
        L_exp = expm(L * dt)
        rho = (L_exp @ rho).reshape(3, 3)
    
    return rho

def _fidelity(rho, state):
    """Compute fidelity |<state|rho|state>|^2 for pure state."""
    return np.abs(state.conj().T @ rho @ state).real

def _compute_score(pulse_IQ, hardware_model):
    """Compute the infidelity score for the given pulse."""
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    dt = 0.5  # ns
    
    # Test cases: detuning in {-2pi*0.002e-3, 0, 2pi*0.002e-3}, scale in {0.97, 1.0, 1.03}
    detunings = [-2 * np.pi * 0.002e-3, 0.0, 2 * np.pi * 0.002e-3]
    scales = [0.97, 1.0, 1.03]
    
    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # X gate: |0> -> |1>, |1> -> |0>
    # Fidelity to X: |<X|rho|state>|^2 where X|0>=|1>, X|1>=|0>
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            case_infidelities = []
            for state in states:
                rho_init = np.outer(state, state.conj())
                rho_final = _evolve_state(rho_init, pulse_IQ, det, sc, anharmonicity, T1, Tphi, dt)
                
                # Target state after X gate
                if np.allclose(state, states[0]):  # |0>
                    target = states[1]  # |1>
                elif np.allclose(state, states[1]):  # |1>
                    target = states[0]  # |0>
                else:
                    # For superposition states, X swaps |0> and |1>
                    target = state.copy()
                    target[0], target[1] = state[1], state[0]
                    target = target / np.linalg.norm(target)
                
                fid = _fidelity(rho_final, target)
                infidelities_case = 1.0 - fid
                case_infidelities.append(infidelities_case)
            
            mean_inf = np.mean(case_infidelities)
            max_inf = np.max(case_infidelities)
            infidelities.append(mean_inf + max_inf)
    
    return np.mean(infidelities)

def _generate_pulse_from_params(params):
    """Generate pulse from basis expansion parameters."""
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt
    
    # Hermite-Gaussian basis: H_n(x) * exp(-x^2/2)
    # Use 4 basis functions
    sigma = 40.0 / 6.0
    t_center = 20.0
    
    # Normalize basis functions
    basis = []
    for k in range(4):
        x = (t - t_center) / sigma
        # Hermite polynomial
        if k == 0:
            Hn = np.ones_like(x)
        elif k == 1:
            Hn = 2 * x
        elif k == 2:
            Hn = 4 * x**2 - 2
        elif k == 3:
            Hn = 8 * x**3 - 12 * x
        else:
            Hn = np.zeros_like(x)
        
        g = np.exp(-x**2 / 2)
        basis.append(Hn * g)
    
    # Combine with parameters
    pulse_IQ = np.zeros((n_steps, 2))
    for k in range(4):
        pulse_IQ[:, 0] += params[2*k] * basis[k]
        pulse_IQ[:, 1] += params[2*k+1] * basis[k]
    
    # Enforce amplitude bound
    amp = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    mask = amp > 0.4
    if np.any(mask):
        scale_factor = 0.4 / amp[mask]
        pulse_IQ[mask, 0] *= scale_factor
        pulse_IQ[mask, 1] *= scale_factor
    
    # Enforce slew rate bound
    # ||p[i+1] - p[i]|| / dt <= 0.12
    for i in range(n_steps - 1):
        diff = pulse_IQ[i+1] - pulse_IQ[i]
        norm_diff = np.linalg.norm(diff)
        if norm_diff > 0.12 * dt:
            scale = 0.12 * dt / norm_diff
            pulse_IQ[i+1] = pulse_IQ[i] + diff * scale
    
    # Enforce slew from zero to step 0
    norm_first = np.linalg.norm(pulse_IQ[0])
    if norm_first > 0.12 * dt:
        pulse_IQ[0] *= 0.12 * dt / norm_first
    
    # Enforce slew from last step to zero
    norm_last = np.linalg.norm(pulse_IQ[-1])
    if norm_last > 0.12 * dt:
        pulse_IQ[-1] *= 0.12 * dt / norm_last
    
    return pulse_IQ

def _seed_pulse():
    """Generate the DRAG-style seed pulse."""
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt
    
    sigma = 40.0 / 6.0
    t_center = 20.0
    
    # Gaussian with area pi
    x = (t - t_center) / sigma
    g = np.exp(-x**2 / 2)
    
    # Normalize to have area pi
    area = np.trapz(g, t)
    if area > 0:
        g *= np.pi / area
    
    # DRAG correction: derivative term
    # dI/dt proportional to -Q, so Q ~ -dI/dt
    dI_dt = np.gradient(g, dt)
    q_pulse = -dI_dt * 0.5  # Some DRAG coefficient
    
    pulse_IQ = np.column_stack([g, q_pulse])
    
    # Enforce constraints
    amp = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    mask = amp > 0.4
    if np.any(mask):
        scale_factor = 0.4 / amp[mask]
        pulse_IQ[mask, 0] *= scale_factor
        pulse_IQ[mask, 1] *= scale_factor
    
    for i in range(n_steps - 1):
        diff = pulse_IQ[i+1] - pulse_IQ[i]
        norm_diff = np.linalg.norm(diff)
        if norm_diff > 0.12 * dt:
            scale = 0.12 * dt / norm_diff
            pulse_IQ[i+1] = pulse_IQ[i] + diff * scale
    
    norm_first = np.linalg.norm(pulse_IQ[0])
    if norm_first > 0.12 * dt:
        pulse_IQ[0] *= 0.12 * dt / norm_first
    
    norm_last = np.linalg.norm(pulse_IQ[-1])
    if norm_last > 0.12 * dt:
        pulse_IQ[-1] *= 0.12 * dt / norm_last
    
    return pulse_IQ

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Uses basis expansion and differential evolution for robust optimization.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second deadline
    
    # Start with seed pulse
    best_pulse = _seed_pulse()
    best_score = _compute_score(best_pulse, hardware_model)
    
    # Check if we should proceed with optimization
    if time.time() > deadline - 60.0:
        return best_pulse
    
    # Define bounds for parameters (8 parameters for 4 basis functions, I and Q each)
    bounds = [(-0.3, 0.3)] * 8
    
    # Objective function for differential evolution
    def objective(params):
        p = _generate_pulse_from_params(params)
        return _compute_score(p, hardware_model)
    
    # Use differential evolution for global optimization
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-4,
            polish=True,
            callback=lambda xk, convergence: None
        )
        
        if time.time() < deadline - 30.0:
            candidate_pulse = _generate_pulse_from_params(result.x)
            candidate_score = _compute_score(candidate_pulse, hardware_model)
            
            if candidate_score < best_score:
                best_pulse = candidate_pulse
                best_score = candidate_score
    except Exception:
        pass
    
    return best_pulse
# EVOLVE-BLOCK-END
