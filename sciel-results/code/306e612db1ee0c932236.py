"""Rewritten program: optimized DRAG pulse for three-level transmon X-pi gate."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Precompute basis operators
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T

# n = a^dag a
n_op = a_dag @ a_op

# Lindblad operators
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian base terms (coherent + dissipative)
# H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
# The Liouvillian in the Liouville space is:
# L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})

# Precompute the superoperator components
def _comm_super(H):
    """Compute the commutator superoperator [H, .] as a matrix."""
    H_flat = H.reshape(9, 1)
    H_dag_flat = H.conj().T.reshape(9, 1)
    # [H, rho] = H @ rho - rho @ H
    # In vectorized form: (I ⊗ H) - (H^T ⊗ I)
    # Actually, vec(H@rho - rho@H) = (I ⊗ H) vec(rho) - (H^T ⊗ I) vec(rho)
    # = (I ⊗ H - H^T ⊗ I) vec(rho)
    # But we need -i times this
    return -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))

def _dissipative_super(L):
    """Compute the dissipative superoperator for a single Lindblad operator."""
    # L rho L^dag - 0.5 {L^dag L, rho}
    L_dag = L.conj().T
    L_dag_L = L_dag @ L
    # vec(L rho L^dag) = (L^dag^T ⊗ L) vec(rho)
    # vec(0.5(L^dag L rho + rho L^dag L)) = 0.5 (I ⊗ L^dag L + (L^dag L)^T ⊗ I) vec(rho)
    return (np.kron(L_dag.T, L) - 0.5 * (np.kron(np.eye(3), L_dag_L) + np.kron(L_dag_L.T, np.eye(3))))

# Precompute dissipative part (independent of pulse)
D_super = _dissipative_super(L1) + _dissipative_super(L2)

# Precompute the static Hamiltonian part (detuning + anharmonicity)
# H_static = detuning * n + 0.5 * anharmonicity * n*(n-1)
# n*(n-1) = n^2 - n
n2_op = n_op @ n_op
n_minus_1 = n_op - np.eye(3)
n_times_n_minus_1 = n_op @ n_minus_1

def build_liouvillian(detuning, scale, I_val, Q_val):
    """Build the full Liouvillian superoperator for given parameters."""
    # Static part
    H_static = detuning * n_op + 0.5 * ANHARMONICITY * n_times_n_minus_1
    # Drive part
    H_drive = 0.5 * scale * (I_val * (a_op + a_dag) + 1j * Q_val * (a_op - a_dag))
    H_total = H_static + H_drive
    C_super = _comm_super(H_total)
    return C_super + D_super

def simulate_pulse(pulse_array, detuning=0.0, scale=1.0):
    """Simulate the pulse and return the final density matrix."""
    I_vals = pulse_array[:, 0]
    Q_vals = pulse_array[:, 1]
    
    # Initial state: identity/3 (maximally mixed) - but we need to track each basis state
    # Actually, we need to compute fidelity for specific input states
    # Let's simulate the unitary part + dissipation for each initial state
    
    # For efficiency, simulate the full Liouvillian evolution
    # Start with rho = I/3
    rho = np.eye(3) / 3.0
    rho_vec = rho.reshape(9)
    
    for i in range(N_STEPS):
        L = build_liouvillian(detuning, scale, I_vals[i], Q_vals[i])
        # Evolve for dt
        # Use matrix exponential for the Liouvillian
        # rho_vec(t+dt) = exp(L * dt) * rho_vec(t)
        # For small dt, use first-order approximation or matrix exp
        # Since dt is small (0.5 ns), use expm
        from scipy.linalg import expm
        expm_L_dt = expm(L * DT)
        rho_vec = expm_L_dt @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho

def compute_fidelity(pulse_array, detuning=0.0, scale=1.0):
    """Compute mean fidelity for the given pulse and parameters."""
    # Simulate for each input state
    # Input states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    
    # For each initial state |psi>, the target is X|psi>
    # Fidelity = |<X|psi>|^2 where we evolve |psi> and compare to X|psi>
    
    # We need to simulate the evolution of each state
    # For a pure state |psi>, rho = |psi><psi|
    
    states = []
    # |0>
    states.append(np.array([1, 0, 0], dtype=complex))
    # |1>
    states.append(np.array([0, 1, 0], dtype=complex))
    # (|0>+|1>)/sqrt(2)
    states.append(np.array([1, 1, 0], dtype=complex) / np.sqrt(2))
    # (|0>-|1>)/sqrt(2)
    states.append(np.array([1, -1, 0], dtype=complex) / np.sqrt(2))
    # (|0>+i|1>)/sqrt(2)
    states.append(np.array([1, 1j, 0], dtype=complex) / np.sqrt(2))
    # (|0>-i|1>)/sqrt(2)
    states.append(np.array([1, -1j, 0], dtype=complex) / np.sqrt(2))
    
    fidelities = []
    for psi in states:
        rho_initial = np.outer(psi, psi.conj())
        rho_vec = rho_initial.reshape(9)
        
        for i in range(N_STEPS):
            L = build_liouvillian(detuning, scale, pulse_array[i, 0], pulse_array[i, 1])
            from scipy.linalg import expm
            expm_L_dt = expm(L * DT)
            rho_vec = expm_L_dt @ rho_vec
        
        rho_final = rho_vec.reshape(3, 3)
        
        # Target state: X|psi>
        # X = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
        X_op = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]], dtype=complex)
        psi_target = X_op @ psi
        rho_target = np.outer(psi_target, psi_target.conj())
        
        # Fidelity = Tr(rho_final @ rho_target) for pure target
        # Actually, for pure target state, fidelity = <psi_target|rho_final|psi_target>
        fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
        fidelities.append(max(0, min(1, fid)))
    
    return np.mean(fidelities)

def score_pulse(pulse_array):
    """Compute the score (mean + max of infidelities over 9 cases)."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fid = compute_fidelity(pulse_array, det, sc)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    infid_arr = np.array(infidelities)
    return np.mean(infid_arr) + np.max(infid_arr)

def check_validity(pulse_array):
    """Check if the pulse satisfies all constraints."""
    # Amplitude bound
    amps = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-9):
        return False
    
    # Slew bound
    # From zero to step 1
    p0 = np.array([0.0, 0.0])
    p1 = pulse_array[0]
    if np.linalg.norm(p1 - p0) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        diff = pulse_array[i+1] - pulse_array[i]
        if np.linalg.norm(diff) / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    plast = pulse_array[-1]
    if np.linalg.norm(plast - p0) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def generate_seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - DURATION/2) / sigma)**2)
    
    # Offset to 0 at edges
    # Subtract the value at the edges
    edge_val = gauss[0]  # Should be small but let's check
    gauss = gauss - edge_val
    
    # Scale to have area pi
    # Area = sum(gauss * dt)
    area = np.sum(gauss) * DT
    gauss = gauss * (np.pi / area) if area > 0 else gauss
    
    # The pulse is in I channel, Q = 0 for seed
    pulse = np.zeros((N_STEPS, 2))
    pulse[:, 0] = gauss
    pulse[:, 1] = 0.0
    
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 300.0  # 300 seconds
    
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Generate seed
    pulse_seed = generate_seed_pulse()
    
    # Check if seed is valid
    if not check_validity(pulse_seed):
        # Scale down to satisfy constraints
        scale_factor = 1.0
        for _ in range(10):
            pulse_test = pulse_seed * scale_factor
            if check_validity(pulse_test):
                break
            scale_factor *= 0.9
        pulse_seed = pulse_seed * scale_factor
    
    best_pulse = pulse_seed.copy()
    best_score = score_pulse(pulse_seed)
    
    # Optimize using Nelder-Mead
    # Parameters: optimize the Gaussian parameters and add DRAG component
    # Let's parameterize: amplitude, width, DRAG amplitude
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            return best_score  # Return current best if near deadline
        
        amp, width, drag_amp = params
        
        # Generate pulse
        t = np.arange(N_STEPS) * DT
        t_center = DURATION / 2
        
        # Gaussian envelope
        gauss = np.exp(-0.5 * ((t - t_center) / width)**2)
        # Offset to 0 at edges
        edge_val = gauss[0]
        gauss = gauss - edge_val
        # Scale to area pi
        area = np.sum(gauss) * DT
        if area > 0:
            gauss = gauss * (np.pi / area)
        
        # DRAG component: derivative of Gaussian
        drag = -np.gradient(gauss, DT)
        
        pulse = np.zeros((N_STEPS, 2))
        pulse[:, 0] = amp * gauss
        pulse[:, 1] = drag_amp * drag
        
        # Check validity
        if not check_validity(pulse):
            return 1e6
        
        return score_pulse(pulse)
    
    # Initial parameters
    x0 = [1.0, 40.0/6.0, 0.0]
    
    # Bounds for parameters
    bounds = [(0.1, AMPLITUDE_BOUND), (5.0, 15.0), (-0.2, 0.2)]
    
    # Run optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_score:
            amp, width, drag_amp = result.x
            t = np.arange(N_STEPS) * DT
            t_center = DURATION / 2
            gauss = np.exp(-0.5 * ((t - t_center) / width)**2)
            edge_val = gauss[0]
            gauss = gauss - edge_val
            area = np.sum(gauss) * DT
            if area > 0:
                gauss = gauss * (np.pi / area)
            drag = -np.gradient(gauss, DT)
            new_pulse = np.zeros((N_STEPS, 2))
            new_pulse[:, 0] = amp * gauss
            new_pulse[:, 1] = drag_amp * drag
            
            if check_validity(new_pulse):
                best_pulse = new_pulse
                best_score = result.fun
    except Exception:
        pass
    
    # Additional refinement: local search around best pulse
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Try small perturbations
        for _ in range(20):
            if time.time() - start_time > deadline - 20.0:
                break
            
            # Random perturbation
            noise = np.random.normal(0, 0.01, (N_STEPS, 2))
            candidate = best_pulse + noise
            
            if check_validity(candidate):
                cand_score = score_pulse(candidate)
                if cand_score < best_score:
                    best_pulse = candidate
                    best_score = cand_score
    
    return best_pulse
