import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5

def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    """Build the 9x9 Liouvillian superoperator."""
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    I3 = np.eye(3, dtype=complex)
    
    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))
        
    return L_total

def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n):
    """
    Simulate the pulse for 6 input states simultaneously.
    Returns mean fidelity.
    """
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Initial density matrices vectorized
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    # Stack into a 9x6 matrix
    states = np.column_stack(initial_states)
    
    # Precompute propagators for each step
    propagators = []
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        propagators.append(U_super)
    
    # Apply propagators
    for U_super in propagators:
        states = U_super @ states
        
    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        
        # Extract rho_out
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
        
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False
    
    return True

def _compute_score(pulse, a, a_dag, n):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _legendre_basis(t):
    """
    Generate the first 8 Legendre polynomials evaluated at t,
    where t is in [0, 1].
    """
    # t is in [0, 1], map to [-1, 1] for standard Legendre
    x = 2 * t - 1
    
    P = np.zeros((N_STEPS, 8))
    for k in range(8):
        if k == 0:
            P[:, k] = 1.0
        elif k == 1:
            P[:, k] = x
        else:
            P[:, k] = (2*k - 1) * x * P[:, k-1] - (k - 1) * P[:, k-2]
    
    return P

def _pulse_from_legendre(coeffs):
    """
    Generate a pulse from 8 Legendre coefficients.
    coeffs: array of shape (8, 2) where column 0 is I coefficients, column 1 is Q coefficients.
    """
    t = np.linspace(0, 1, N_STEPS)
    P = _legendre_basis(t)
    
    I = P @ coeffs[:, 0]
    Q = P @ coeffs[:, 1]
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    return pulse

def _check_legendre_validity(pulse):
    """Check if the pulse satisfies all constraints."""
    return _check_validity(pulse)

def _objective_legendre(coeffs, a, a_dag, n):
    """
    Objective function for Legendre coefficient optimization.
    coeffs: array of shape (16,) = 8 I coeffs + 8 Q coeffs
    """
    coeffs_2d = coeffs.reshape(8, 2)
    pulse = _pulse_from_legendre(coeffs_2d)
    
    if not _check_validity(pulse):
        return 1e10
    
    score = _compute_score(pulse, a, a_dag, n)
    return score

def _generate_initial_guess_legendre():
    """
    Generate an initial guess for Legendre coefficients based on a DRAG-like shape.
    """
    # Start with a Gaussian-like pulse
    t = np.linspace(0, 1, N_STEPS)
    gauss = np.exp(-0.5 * ((t - 0.5) / 0.08)**2)
    # Offset to 0 at edges
    edge_val = gauss[0]
    gauss = gauss - edge_val
    # Normalize area to pi
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss / area * np.pi
    
    # Derivative for Q
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss[1:] - gauss[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I_target = gauss
    Q_target = 0.2 * gauss  # Simple DRAG-like Q
    
    # Project onto Legendre basis
    P = _legendre_basis(t)
    
    # Least squares fit
    coeffs_I, _, _, _ = np.linalg.lstsq(P, I_target, rcond=None)
    coeffs_Q, _, _, _ = np.linalg.lstsq(P, Q_target, rcond=None)
    
    coeffs = np.column_stack([coeffs_I, coeffs_Q])
    return coeffs.flatten()

def pulse(target_gate, hardware_model):
    """Main entry point with Legendre-based optimization."""
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    a, a_dag, n = _build_matrices()
    
    # Generate initial guess
    coeffs_init = _generate_initial_guess_legendre()
    pulse_init = _pulse_from_legendre(coeffs_init.reshape(8, 2))
    
    # Ensure initial pulse is valid
    if not _check_validity(pulse_init):
        # Scale down
        amps = np.sqrt(pulse_init[:, 0]**2 + pulse_init[:, 1]**2)
        max_amp = np.max(amps)
        if max_amp > AMPLITUDE_BOUND:
            pulse_init *= AMPLITUDE_BOUND / max_amp
        # Fix slew
        for i in range(N_STEPS - 1):
            diff = np.linalg.norm(pulse_init[i+1] - pulse_init[i])
            if diff / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / diff
                pulse_init[i+1] = pulse_init[i] + scale * (pulse_init[i+1] - pulse_init[i])
    
    best_pulse = pulse_init.copy()
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Optimization using L-BFGS-B
    try:
        result = minimize(
            _objective_legendre,
            coeffs_init,
            args=(a, a_dag, n),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
        )
        
        if result.success or result.fun < best_score:
            coeffs_opt = result.x
            pulse_opt = _pulse_from_legendre(coeffs_opt.reshape(8, 2))
            
            if _check_validity(pulse_opt):
                score_opt = result.fun
                if score_opt < best_score:
                    best_score = score_opt
                    best_pulse = pulse_opt.copy()
            else:
                # If invalid, try to fix
                amps = np.sqrt(pulse_opt[:, 0]**2 + pulse_opt[:, 1]**2)
                max_amp = np.max(amps)
                if max_amp > AMPLITUDE_BOUND:
                    pulse_opt *= AMPLITUDE_BOUND / max_amp
                for i in range(N_STEPS - 1):
                    diff = np.linalg.norm(pulse_opt[i+1] - pulse_opt[i])
                    if diff / DT > SLEW_BOUND:
                        scale = SLEW_BOUND * DT / diff
                        pulse_opt[i+1] = pulse_opt[i] + scale * (pulse_opt[i+1] - pulse_opt[i])
                
                if _check_validity(pulse_opt):
                    score_opt = _compute_score(pulse_opt, a, a_dag, n)
                    if score_opt < best_score:
                        best_score = score_opt
                        best_pulse = pulse_opt.copy()
    except Exception as e:
        pass  # Keep the initial pulse if optimization fails
    
    # Local refinement around the best pulse
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(42)
        n_local = 50
        for i in range(n_local):
            if time.time() > deadline - 20.0:
                break
            
            # Perturb the pulse
            noise = rng.normal(0, 0.02, size=best_pulse.shape)
            p = best_pulse + noise
            
            # Enforce constraints
            amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
            max_amp = np.max(amps)
            if max_amp > AMPLITUDE_BOUND:
                p *= AMPLITUDE_BOUND / max_amp
            
            for j in range(N_STEPS - 1):
                diff = np.linalg.norm(p[j+1] - p[j])
                if diff / DT > SLEW_BOUND:
                    scale = SLEW_BOUND * DT / diff
                    p[j+1] = p[j] + scale * (p[j+1] - p[j])
            
            if _check_validity(p):
                s = _compute_score(p, a, a_dag, n)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()
    
    # Ensure final pulse is valid
    if not _check_validity(best_pulse):
        # Fallback
        best_pulse = _pulse_from_legendre(_generate_initial_guess_legendre().reshape(8, 2))
        amps = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
        max_amp = np.max(amps)
        if max_amp > AMPLITUDE_BOUND:
            best_pulse *= AMPLITUDE_BOUND / max_amp
    
    return best_pulse
