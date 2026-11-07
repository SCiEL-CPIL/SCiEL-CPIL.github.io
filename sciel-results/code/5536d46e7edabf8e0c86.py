import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
from numpy.polynomial.legendre import legvander

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
N_COEFFS = 8

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
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
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

def _legendre_basis():
    """Create Legendre polynomial evaluation matrix for N_STEPS points in [0,1]."""
    # Map step indices to [0, 1]
    t = np.linspace(0, 1, N_STEPS)
    # legvander gives V[i, j] = P_j(t_i)
    V = legvander(t, N_COEFFS - 1)
    return V

def _pulse_from_legendre(coeffs):
    """Convert 16 coefficients (8 for I, 8 for Q) to pulse array."""
    V = _legendre_basis()
    I_coeffs = coeffs[:N_COEFFS]
    Q_coeffs = coeffs[N_COEFFS:]
    I = V @ I_coeffs
    Q = V @ Q_coeffs
    pulse = np.column_stack([I, Q])
    return pulse

def _check_legendre_validity(pulse):
    """Check validity with a small tolerance for optimization."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-9):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-9:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-9:
        return False
    
    return True

def _objective_legendre(coeffs, a, a_dag, n):
    """Objective function for optimization."""
    pulse = _pulse_from_legendre(coeffs)
    
    # Check validity with strict bounds
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-6):
        return 1e6 + np.max(amps - AMPLITUDE_BOUND)
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-6:
        return 1e6 + (slew0 - SLEW_BOUND)
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-6:
            return 1e6 + (diff - SLEW_BOUND)
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-6:
        return 1e6 + (slew_last - SLEW_BOUND)
    
    score = _compute_score(pulse, a, a_dag, n)
    return score

def _generate_initial_guess_legendre(a, a_dag, n):
    """Generate an initial guess in Legendre coefficient space."""
    # Start with a simple DRAG-like pulse and convert to Legendre
    t = np.arange(N_STEPS) * DT
    sigma = 5.0
    t_center = 20.0
    A = 0.25
    beta = 0.15
    
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = gauss
    
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = A * gauss_norm
    Q = beta * A * dgauss
    
    # Fit to Legendre basis
    V = _legendre_basis()
    # Least squares fit
    I_coeffs, _, _, _ = np.linalg.lstsq(V, I, rcond=None)
    Q_coeffs, _, _, _ = np.linalg.lstsq(V, Q, rcond=None)
    
    coeffs = np.concatenate([I_coeffs, Q_coeffs])
    
    # Check if valid
    pulse = _pulse_from_legendre(coeffs)
    if _check_legendre_validity(pulse):
        return coeffs
    
    # If not valid, try simpler coefficients
    coeffs = np.zeros(2 * N_COEFFS)
    coeffs[0] = 0.2  # Constant term for I
    coeffs[N_COEFFS] = 0.05  # Small constant for Q
    return coeffs

def _enforce_constraints_legendre(coeffs):
    """Enforce constraints by scaling if necessary."""
    pulse = _pulse_from_legendre(coeffs)
    
    # Check amplitude
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        coeffs *= scale_factor
        pulse = _pulse_from_legendre(coeffs)
    
    # Check slew
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i])
        if diff / DT > SLEW_BOUND:
            # Need to reduce variation
            break
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    max_slew = slew0
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        max_slew = max(max_slew, diff)
    slew_last = np.linalg.norm(pulse[-1]) / DT
    max_slew = max(max_slew, slew_last)
    
    if max_slew > SLEW_BOUND:
        scale_factor = SLEW_BOUND / max_slew
        coeffs *= scale_factor
        pulse = _pulse_from_legendre(coeffs)
    
    return coeffs

def pulse(target_gate, hardware_model):
    """Main entry point with Legendre basis optimization."""
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    a, a_dag, n = _build_matrices()
    rng = np.random.default_rng(42)
    
    # Get initial guess
    best_coeffs = _generate_initial_guess_legendre(a, a_dag, n)
    best_pulse = _pulse_from_legendre(best_coeffs)
    
    # Ensure validity
    if not _check_validity(best_pulse):
        best_coeffs = _enforce_constraints_legendre(best_coeffs)
        best_pulse = _pulse_from_legendre(best_coeffs)
    
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # Try multiple random restarts with L-BFGS-B
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        # Generate random initial coefficients
        if restart == 0:
            coeffs0 = best_coeffs.copy()
        else:
            # Random perturbation
            coeffs0 = best_coeffs + rng.normal(0, 0.05, size=2 * N_COEFFS)
            coeffs0 = _enforce_constraints_legendre(coeffs0)
        
        # Optimize
        try:
            result = minimize(
                _objective_legendre,
                coeffs0,
                args=(a, a_dag, n),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result.success or result.fun < best_score:
                new_coeffs = _enforce_constraints_legendre(result.x)
                new_pulse = _pulse_from_legendre(new_coeffs)
                
                if _check_validity(new_pulse):
                    new_score = _compute_score(new_pulse, a, a_dag, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_pulse = new_pulse
                        best_coeffs = new_coeffs
        except Exception:
            pass
    
    # Final local refinement
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                _objective_legendre,
                best_coeffs,
                args=(a, a_dag, n),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                new_coeffs = _enforce_constraints_legendre(result.x)
                new_pulse = _pulse_from_legendre(new_coeffs)
                
                if _check_validity(new_pulse):
                    new_score = _compute_score(new_pulse, a, a_dag, n)
                    if new_score < best_score:
                        best_score = new_score
                        best_pulse = new_pulse
                        best_coeffs = new_coeffs
        except Exception:
            pass
    
    # Ensure final pulse is valid
    if not _check_validity(best_pulse):
        # Fallback
        best_coeffs = _enforce_constraints_legendre(best_coeffs)
        best_pulse = _pulse_from_legendre(best_coeffs)
    
    return best_pulse
