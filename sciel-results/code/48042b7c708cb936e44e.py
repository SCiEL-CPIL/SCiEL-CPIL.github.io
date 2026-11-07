"""Rewritten program: robust control pulses for a three-level transmon.
Uses Legendre-basis parameterization with L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# Constants
N_STEPS = 80
DT = 0.5  # ns
T_TOTAL = N_STEPS * DT  # 40 ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
ANHARMONICITY = -2 * np.pi * 0.25
T1 = 25000.0  # ns
T2 = 40000.0  # ns  # Tphi

# 3-level system
# a = diag(sqrt([1,2]), 1) in basis |0>,|1>,|2>
A = np.diag([np.sqrt(1.0), np.sqrt(2.0), 1.0])
A_DAG = A.conj().T
N_OP = A_DAG @ A  # diag([0,1,2])

def liouvillian(I, Q, detuning, scale):
    """Construct the Liouvillian superoperator for given pulse parameters."""
    # Hamiltonian
    # H = detuning*n + 0.5*anharm*n*(n-1) + 0.5*scale*(I*(a+a^dag) + i*Q*(a-a^dag))
    H = detuning * N_OP + 0.5 * ANHARMONICITY * (N_OP @ (N_OP - np.eye(3)))
    drive = 0.5 * scale * (I * (A + A_DAG) + 1j * Q * (A - A_DAG))
    H = H + drive
    
    # Lindblad operators
    gamma1 = 1.0 / T1
    gamma_phi = 2.0 / T2 - 1.0 / T1  # 1/T2 - 1/T1 = 2/Tphi - 1/T1
    
    L1 = np.sqrt(gamma1) * A
    L2 = np.sqrt(max(gamma_phi, 0.0)) * N_OP
    
    # Build Liouvillian: L(rho) = -i[H,rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # Vectorize: rho -> vec(rho) (column-major)
    # -i[H,rho] -> -i (H ⊗ I - I ⊗ H^T) vec(rho)
    # L_k rho L_k^dag -> (L_k^dag^T ⊗ L_k) vec(rho)
    # -0.5 L_k^dag L_k rho -> -0.5 (I ⊗ L_k^dag L_k) vec(rho)
    # -0.5 rho L_k^dag L_k -> -0.5 ((L_k^dag L_k)^T ⊗ I) vec(rho)
    
    I3 = np.eye(3)
    dim = 9
    
    L_mat = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    for L in [L1, L2]:
        LdagL = L.conj().T @ L
        L_mat += np.kron(L.conj().T.T, L) - 0.5 * np.kron(I3, LdagL) - 0.5 * np.kron(LdagL.T, I3)
    
    return L_mat

def vec_to_rho(v):
    """Convert vector to density matrix (column-major)."""
    return v.reshape(3, 3, order='F')

def rho_to_vec(rho):
    """Convert density matrix to vector (column-major)."""
    return rho.flatten(order='F')

def evolve_pulse(pulse_I, pulse_Q, detuning, scale):
    """Evolve the state for a given pulse and return final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # start in |0>
    rho_vec = rho_to_vec(rho)
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        L_mat = liouvillian(I_val, Q_val, detuning, scale)
        # Exponentiate: rho_vec_{i+1} = expm(L_mat * dt) @ rho_vec_i
        # For efficiency, we can use the matrix exponential
        rho_vec = expm(L_mat * DT) @ rho_vec
    
    return vec_to_rho(rho_vec)

def fidelity(rho, target):
    """Compute fidelity F = <psi|rho|psi> for pure target state."""
    return np.real(np.vdot(target, rho @ target))

def compute_score(pulse_I, pulse_Q):
    """Compute the scoring metric: mean + max infidelity over 9 cases."""
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [0.97, 1.0, 1.03]
    
    # Target states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_ip = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_im = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_ip, psi_im]
    
    # X gate: |0> -> |1>, |1> -> |0>
    # Fidelity to X means: after applying pulse, state should be X|psi>
    # So we compare final state with X applied to initial state
    
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            case_infs = []
            for psi in target_states:
                target = X_gate @ psi
                rho_final = evolve_pulse(pulse_I, pulse_Q, det, sc)
                fid = fidelity(rho_final, target)
                case_infs.append(1.0 - fid)
            infidelities.append(np.mean(case_infs))
    
    return np.mean(infidelities) + np.max(infidelities)

def check_validity(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
    # Amplitude bound
    amps = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.max(amps) > AMPLITUDE_BOUND + 1e-12:
        return False
    
    # Slew bound
    # From zero to step 1
    if np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT > SLEW_BOUND + 1e-12:
        return False
    # Between steps
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-12:
            return False
    # From last step to zero
    if np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT > SLEW_BOUND + 1e-12:
        return False
    
    return True

def legendre_to_pulse(coeffs_I, coeffs_Q):
    """Convert Legendre coefficients to pulse values."""
    t = np.linspace(-1, 1, N_STEPS)
    pulse_I = np.zeros(N_STEPS)
    pulse_Q = np.zeros(N_STEPS)
    for k, (cI, cQ) in enumerate(zip(coeffs_I, coeffs_Q)):
        Pk = np.polynomial.legendre.legval(t, [0]*k + [1])
        pulse_I += cI * Pk
        pulse_Q += cQ * Pk
    return pulse_I, pulse_Q

def pulse_to_legendre(pulse_I, pulse_Q):
    """Convert pulse to Legendre coefficients (least squares)."""
    t = np.linspace(-1, 1, N_STEPS)
    V = np.vstack([np.polynomial.legendre.legval(t, [0]*k + [1]) for k in range(8)])
    coeffs_I = np.linalg.lstsq(V, pulse_I, rcond=None)[0]
    coeffs_Q = np.linalg.lstsq(V, pulse_Q, rcond=None)[0]
    return coeffs_I, coeffs_Q

def make_seed_pulse():
    """Create DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, T_TOTAL, N_STEPS)
    sigma = 40.0 / 6.0
    # Gaussian centered at T_TOTAL/2
    g = np.exp(-((t - T_TOTAL/2)**2) / (2 * sigma**2))
    # Offset to make area = pi and zero at edges
    # Actually, the seed is described as: Gaussian (sigma 40/6 ns, offset to 0 at the edges, area pi)
    # Let's create a Gaussian that is zero at t=0 and t=T_TOTAL
    # Use a shifted/scaled Gaussian
    center = T_TOTAL / 2
    g = np.exp(-((t - center)**2) / (2 * sigma**2))
    # Subtract edge values to make it zero at edges
    g_edge = g[0]  # approximately equal to g[-1] by symmetry
    g = g - g_edge
    # Normalize area to pi
    area = np.sum(g) * DT
    if abs(area) > 1e-12:
        g = g * (np.pi / area)
    
    pulse_I = g
    pulse_Q = np.zeros(N_STEPS)
    
    # Check and clip
    amps = np.abs(pulse_I)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        pulse_I = pulse_I * (AMPLITUDE_BOUND / max_amp)
    
    # Check slew
    for i in range(N_STEPS - 1):
        d = abs(pulse_I[i+1] - pulse_I[i]) / DT
        if d > SLEW_BOUND:
            # Reduce amplitude
            scale = SLEW_BOUND * DT / max(abs(pulse_I[i+1] - pulse_I[i]), 1e-12)
            pulse_I = pulse_I * min(scale, 1.0)
            break
    
    return pulse_I, pulse_Q

def optimize_pulse():
    """Optimize the pulse using Legendre basis and L-BFGS-B."""
    start_time = time.time()
    deadline = start_time + 300.0  # 300s limit
    
    # Create seed
    seed_I, seed_Q = make_seed_pulse()
    
    # Convert to Legendre coefficients
    coeffs_I, coeffs_Q = pulse_to_legendre(seed_I, seed_Q)
    
    # Combine into single vector
    x0 = np.concatenate([coeffs_I, coeffs_Q])
    
    def objective(x):
        cI = x[:8]
        cQ = x[8:]
        pulse_I, pulse_Q = legendre_to_pulse(cI, cQ)
        
        # Check validity
        if not check_validity(pulse_I, pulse_Q):
            return 1e6
        
        return compute_score(pulse_I, pulse_Q)
    
    # Initial evaluation
    best_score = objective(x0)
    best_x = x0.copy()
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30.0:
        # Not enough time for more optimization
        pulse_I, pulse_Q = legendre_to_pulse(best_x[:8], best_x[8:])
        return pulse_I, pulse_Q
    
    # Try a second round with different initial conditions
    # Add some random perturbations
    np.random.seed(42)
    for trial in range(3):
        if time.time() > deadline - 60.0:
            break
        
        x_trial = best_x + np.random.randn(16) * 0.01
        try:
            result = minimize(
                objective, x_trial, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            pass
    
    pulse_I, pulse_Q = legendre_to_pulse(best_x[:8], best_x[8:])
    return pulse_I, pulse_Q

# Pre-compute the pulse
_pulse_I, _pulse_Q = optimize_pulse()

def pulse(target_gate, hardware_model):
    """Return the pulse as an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gates
        return np.zeros((N_STEPS, 2))
    
    result = np.column_stack([_pulse_I, _pulse_Q])
    return result
# EVOLVE-BLOCK-END
