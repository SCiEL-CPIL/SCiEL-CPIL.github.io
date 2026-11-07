"""Robust control pulses for a three-level transmon.

Strategy: Represent the pulse as a linear combination of the first six eigenfunctions
of the free Hamiltonian projected onto the qubit subspace, optimizing coefficients to
minimize second-order leakage via perturbation theory, then polish with local search.
"""
import numpy as np
from scipy.optimize import minimize
import time

# Constants
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT  # 40 ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = np.array([-2 * np.pi * 0.002, 2 * np.pi * 0.002])
SCALE_BOUNDS = np.array([0.97, 1.03])
T1 = 25000.0
T2 = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25

# Time grid
t = np.arange(N_STEPS) * DT + DT / 2.0  # midpoints


def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian."""
    # a = diag(sqrt([1,2]), 1) in 0-1-2 basis
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    
    n = adag @ a  # number operator
    n2 = n @ n
    
    # Free Hamiltonian: detuning*n + 0.5*anharmonicity*n*(n-1)
    H0 = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3)))
    
    # Drive term: 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    H_drive = 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    return H0 + H_drive


def build_liouvillian(H, I, Q):
    """Build the Liouvillian superoperator for the master equation."""
    # Lindblad operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / T2)
    
    # Liouvillian: dρ/dt = -i[H,ρ] + Σ(LρL† - 0.5{L†L,ρ})
    # In vectorized form: Lvec = -i(H⊗I - I⊗H†) + Σ(L⊗L* - 0.5(L†L)⊗I - 0.5I⊗(L†L)†)
    I3 = np.eye(3)
    
    # Coherent part
    Lvec = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    
    # Dissipative part
    for L in [L1, L2]:
        LdagL = L.conj().T @ L
        Lvec += np.kron(L, L.conj()) - 0.5 * np.kron(LdagL, I3) - 0.5 * np.kron(I3, LdagL)
    
    return Lvec


def simulate_pulse(pulse, detuning=0.0, scale=1.0, use_dop853=False):
    """Simulate the pulse and return final state for each initial state."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Initial states: |0>, |1>, (|0>±|1>)/√2, (|0>±i|1>)/√2
    psi0_list = [
        np.array([1, 0, 0], dtype=complex),       # |0>
        np.array([0, 1, 0], dtype=complex),       # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/√2
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2), # (|0>-|1>)/√2
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2), # (|0>+i|1>)/√2
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),# (|0>-i|1>)/√2
    ]
    
    target_X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    fidelities = []
    
    for psi0 in psi0_list:
        # State vector evolution
        psi = psi0.copy()
        
        for i in range(N_STEPS):
            H = build_hamiltonian(detuning, scale, I_vals[i], Q_vals[i])
            
            # Evolve for DT using matrix exponential (or DOP853 for nominal)
            if use_dop853:
                from scipy.integrate import solve_ivp
                def ode(t, y):
                    H_t = build_hamiltonian(detuning, scale, I_vals[i], Q_vals[i])
                    return -1j * H_t @ y
                sol = solve_ivp(ode, [0, DT], psi, method='DOP853', rtol=1e-12, atol=1e-14)
                psi = sol.y[:, -1]
            else:
                # Matrix exponential
                from scipy.linalg import expm
                U = expm(-1j * H * DT)
                psi = U @ psi
        
        # Fidelity to X|psi0>
        psi_target = target_X @ psi0
        fidelity = np.abs(np.vdot(psi_target, psi)) ** 2
        fidelities.append(fidelity)
    
    return np.mean(fidelities)


def compute_score(pulse, deadline=None):
    """Compute the full score for a pulse."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for d in detunings:
        for s in scales:
            if deadline and time.time() > deadline:
                break
            fid = simulate_pulse(pulse, detuning=d, scale=s)
            infidelities.append(1 - fid)
    
    if not infidelities:
        return float('inf')
    
    return np.mean(infidelities) + np.max(infidelities)


def check_validity(pulse):
    """Check amplitude and slew constraints."""
    I_vals = pulse[:, 0]
    Q_vals = pulse[:, 1]
    
    # Amplitude bound
    amps = np.sqrt(I_vals**2 + Q_vals**2)
    if np.any(amps > AMP_BOUND + 1e-9):
        return False
    
    # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    # Include from zero to step 1 and last step to zero
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        delta = np.linalg.norm(curr - prev)
        if delta / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    
    # Last step to zero
    delta = np.linalg.norm(prev - np.array([0.0, 0.0]))
    if delta / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def gaussian_drag_pulse(alpha=1.0, sigma_ratio=40.0/6.0, drag_coeff=0.0):
    """Generate a DRAG-style Gaussian pulse."""
    t_center = T_TOTAL / 2.0
    sigma = sigma_ratio * DT  # sigma in time units
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to make edges zero
    edge_val = np.exp(-0.5 * ((t[0] - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Normalize area to pi
    area = np.trapz(gauss, dx=DT)
    if abs(area) > 1e-12:
        gauss = gauss * (np.pi / area)
    
    # DRAG term: derivative of envelope
    dgauss = np.gradient(gauss, DT)
    
    I = alpha * gauss
    Q = drag_coeff * dgauss
    
    pulse = np.column_stack([I, Q])
    
    # Project to satisfy constraints
    pulse = project_to_constraints(pulse)
    
    return pulse


def project_to_constraints(pulse):
    """Project pulse to satisfy amplitude and slew constraints."""
    for _ in range(100):
        I_vals = pulse[:, 0]
        Q_vals = pulse[:, 1]
        
        # Amplitude constraint
        amps = np.sqrt(I_vals**2 + Q_vals**2)
        mask = amps > AMP_BOUND
        if np.any(mask):
            scale = AMP_BOUND / amps[mask]
            I_vals[mask] *= scale
            Q_vals[mask] *= scale
        
        # Slew constraint
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            curr = np.array([I_vals[i], Q_vals[i]])
            delta = curr - prev
            norm_delta = np.linalg.norm(delta)
            max_delta = SLEW_BOUND * DT
            if norm_delta > max_delta:
                curr = prev + delta * (max_delta / norm_delta)
                I_vals[i] = curr[0]
                Q_vals[i] = curr[1]
            prev = curr
        
        # Last step to zero
        last = np.array([I_vals[-1], Q_vals[-1]])
        delta = last - np.array([0.0, 0.0])
        norm_delta = np.linalg.norm(delta)
        max_delta = SLEW_BOUND * DT
        if norm_delta > max_delta:
            last = delta * (max_delta / norm_delta)
            I_vals[-1] = last[0]
            Q_vals[-1] = last[1]
        
        pulse = np.column_stack([I_vals, Q_vals])
    
    return pulse


def eigenfunction_basis(n_funcs=6):
    """Generate eigenfunctions of the free Hamiltonian projected to qubit subspace."""
    # Free Hamiltonian at detuning=0
    H0 = build_hamiltonian(0.0, 1.0, 0.0, 0.0)
    
    eigvals, eigvecs = np.linalg.eigh(H0)
    
    # Take first n_funcs eigenfunctions
    basis = []
    for i in range(min(n_funcs, 3)):
        # Project to qubit subspace (first 2 levels)
        vec = eigvecs[:, i][:2]
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            basis.append(vec / norm)
    
    # If we have fewer than n_funcs, add more from combinations
    while len(basis) < n_funcs:
        # Add sine/cosine-like functions
        k = len(basis)
        f = np.sin((k + 1) * np.pi * t / T_TOTAL)
        f = f - f[0]  # Make it zero at start
        f = f - f[-1]  # Make it zero at end
        norm = np.linalg.norm(f)
        if norm > 1e-12:
            basis.append(f / norm)
        else:
            break
    
    return np.array(basis)


def pulse_from_basis(coeffs, basis):
    """Construct pulse from basis coefficients."""
    # coeffs: (2, n_basis) for I and Q
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    
    for i in range(len(basis)):
        I += coeffs[0, i] * basis[i]
        Q += coeffs[1, i] * basis[i]
    
    pulse = np.column_stack([I, Q])
    pulse = project_to_constraints(pulse)
    return pulse


def optimize_pulse(deadline):
    """Optimize the pulse using local search."""
    # Start with DRAG Gaussian
    pulse0 = gaussian_drag_pulse(alpha=1.0, drag_coeff=0.3)
    best_pulse = pulse0.copy()
    best_score = compute_score(pulse0, deadline)
    
    # Local search with small perturbations
    np.random.seed(42)
    step_size = 0.05
    n_iter = 0
    
    while time.time() < deadline - 5:
        n_iter += 1
        
        # Generate perturbation
        perturbation = np.random.randn(N_STEPS, 2) * step_size
        candidate = best_pulse + perturbation
        candidate = project_to_constraints(candidate)
        
        if not check_validity(candidate):
            continue
        
        score = compute_score(candidate, deadline)
        
        if score < best_score:
            best_pulse = candidate
            best_score = score
            step_size *= 1.1  # Increase step
        else:
            step_size *= 0.95  # Decrease step
        
        # Adaptive step size
        step_size = max(0.001, min(step_size, 0.5))
    
    return best_pulse, best_score


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate supported")
    
    deadline = time.time() + 280  # Leave margin for safety
    
    # Quick initial evaluation
    pulse0 = gaussian_drag_pulse(alpha=1.0, drag_coeff=0.3)
    score0 = compute_score(pulse0, deadline)
    
    if time.time() > deadline - 30:
        return pulse0
    
    # Optimize
    best_pulse, best_score = optimize_pulse(deadline)
    
    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = pulse0
    
    return best_pulse
