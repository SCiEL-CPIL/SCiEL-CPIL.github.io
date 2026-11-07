"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

# Constants
N_STEPS = 80
DT = 0.5  # ns
T_TOTAL = N_STEPS * DT  # 40 ns

# Hardware model parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12  # rad/ns^2
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
TPHI = 40000.0  # ns

# 3-level system
# a = diag(sqrt([1, 2]), 1)
a = np.diag([np.sqrt(1), np.sqrt(2), 1])
a_dag = a.conj().T
n = a_dag @ a  # n = diag([0, 1, 2])

def build_hamiltonian(I, Q, detuning, scale):
    """Build the Hamiltonian for given I, Q, detuning, and scale."""
    # H = detuning * n + 0.5 * anharmonicity * n * (n - 1) + 0.5 * scale * (I * (a + a_dag) + i * Q * (a - a_dag))
    H0 = detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3))
    H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H0 + H_drive

def build_liouvillian(H, detuning, scale):
    """Build the Liouvillian superoperator for the Lindblad equation."""
    dim = 3
    # Lindblad operators: L1 = a / sqrt(T1), L2 = n * sqrt(2 / Tphi)
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    # Liouvillian: L(rho) = -i/2 [H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
    # In vectorized form: L = -i/2 (H ⊗ I - I ⊗ H^T) + sum_k (L_k ⊗ L_k^* - 1/2 (L_k^dag L_k ⊗ I + I ⊗ (L_k^dag L_k)^T))
    
    H_vec = H.reshape(-1)
    I3 = np.eye(dim)
    
    # Coherent part: -i/2 (H ⊗ I - I ⊗ H^T)
    L_coherent = -0.5j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Dissipative part
    L_diss = np.zeros((dim * dim, dim * dim), dtype=complex)
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        L_diss += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag_L, I3) + np.kron(I3, L_dag_L.T))
    
    return L_coherent + L_diss

def evolve_pulse(I_arr, Q_arr, detuning, scale):
    """Evolve the pulse and return the final density matrix."""
    dim = 3
    # Initial state: |0><0|
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        H = build_hamiltonian(I_arr[i], Q_arr[i], detuning, scale)
        L = build_liouvillian(H, detuning, scale)
        
        # Evolve for DT
        rho_vec = rho.reshape(-1)
        rho_vec_new = expm(L * DT) @ rho_vec
        rho = rho_vec_new.reshape(dim, dim)
    
    return rho

def compute_fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure target state."""
    # Fidelity for pure target: F = <psi|rho|psi>
    psi = target_state.reshape(-1, 1)
    F = (psi.conj().T @ rho @ psi).real
    return max(0.0, min(1.0, F))

def compute_score(I_arr, Q_arr):
    """Compute the score for a given pulse."""
    # Target states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    dim = 3
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # 9 cases: detuning in (-2*pi*0.002, 0, 2*pi*0.002), scale in (0.97, 1, 1.03)
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            rho_final = evolve_pulse(I_arr, Q_arr, det, scale)
            # Mean fidelity over the 6 states
            fidelities = [compute_fidelity(rho_final, s) for s in states]
            mean_fid = np.mean(fidelities)
            infidelity = 1.0 - mean_fid
            infidelities.append(infidelity)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        # Return a default valid pulse for other gates
        return np.zeros((N_STEPS, 2))
    
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second limit with margin
    
    # Initial DRAG-style Gaussian seed
    t = np.linspace(0, T_TOTAL, N_STEPS)
    sigma = 40.0 / 6.0
    # Gaussian with offset to 0 at edges
    gauss = np.exp(-0.5 * ((t - T_TOTAL / 2) / sigma) ** 2)
    # Offset to make edges zero
    edge_val = gauss[0]
    gauss = gauss - edge_val
    # Scale to have area pi
    area = np.sum(gauss) * DT
    if area != 0:
        gauss = gauss * (np.pi / area)
    else:
        gauss = np.zeros(N_STEPS)
    
    I_seed = gauss
    Q_seed = np.zeros(N_STEPS)
    
    # Check validity of seed
    def check_validity(I_arr, Q_arr):
        # Amplitude bound
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        if np.max(amps) > AMPLITUDE_BOUND + 1e-9:
            return False
        # Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
        # Including from zero into step 1 and from last step to zero
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            dI = I_arr[i] - prev_I
            dQ = Q_arr[i] - prev_Q
            if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-9:
                return False
            prev_I, prev_Q = I_arr[i], Q_arr[i]
        # From last step to zero
        dI = 0.0 - prev_I
        dQ = 0.0 - prev_Q
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-9:
            return False
        return True
    
    if not check_validity(I_seed, Q_seed):
        # Scale down to satisfy constraints
        # Find max amplitude violation
        amps = np.sqrt(I_seed**2 + Q_seed**2)
        max_amp = np.max(amps)
        if max_amp > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / max_amp * 0.99
            I_seed = I_seed * scale_factor
            Q_seed = Q_seed * scale_factor
        
        # Check slew
        prev_I, prev_Q = 0.0, 0.0
        max_slew = 0.0
        for i in range(N_STEPS):
            dI = I_seed[i] - prev_I
            dQ = Q_seed[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            max_slew = max(max_slew, slew)
            prev_I, prev_Q = I_seed[i], Q_seed[i]
        dI = 0.0 - prev_I
        dQ = 0.0 - prev_Q
        max_slew = max(max_slew, np.sqrt(dI**2 + dQ**2) / DT)
        
        if max_slew > SLEW_BOUND:
            scale_factor = SLEW_BOUND / max_slew * 0.99
            I_seed = I_seed * scale_factor
            Q_seed = Q_seed * scale_factor
    
    # Optimize using Legendre polynomial basis
    # p(t) = sum_{k=0}^{7} c_k * L_k(2t/T - 1)
    # We optimize I and Q separately using Legendre coefficients
    
    def legendre_basis(t, order):
        """Return Legendre polynomial values of given order at points t (scaled to [-1, 1])."""
        x = 2 * t / T_TOTAL - 1
        from numpy.polynomial.legendre import legval
        # We'll use the first (order+1) coefficients
        return legval(x, np.eye(order + 1))
    
    # Precompute Legendre basis values
    L_basis = np.zeros((N_STEPS, 8))
    for k in range(8):
        x = 2 * t / T_TOTAL - 1
        from numpy.polynomial.legendre import legval
        L_basis[:, k] = legval(x, np.zeros(8) + 1.0 if k == 0 else np.zeros(8))
        # Actually, let's compute each Legendre polynomial separately
        coeffs = np.zeros(8)
        coeffs[k] = 1.0
        L_basis[:, k] = legval(x, coeffs)
    
    def legendre_to_pulse(cI, cQ):
        """Convert Legendre coefficients to I and Q arrays."""
        I_arr = L_basis @ cI
        Q_arr = L_basis @ cQ
        return I_arr, Q_arr
    
    def objective(params):
        """Objective function for optimization."""
        cI = params[:8]
        cQ = params[8:]
        I_arr, Q_arr = legendre_to_pulse(cI, cQ)
        
        # Check amplitude constraint
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        if np.max(amps) > AMPLITUDE_BOUND:
            return 1e6
        
        # Check slew constraint
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            dI = I_arr[i] - prev_I
            dQ = Q_arr[i] - prev_Q
            if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND:
                return 1e6
            prev_I, prev_Q = I_arr[i], Q_arr[i]
        dI = 0.0 - prev_I
        dQ = 0.0 - prev_Q
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND:
            return 1e6
        
        return compute_score(I_arr, Q_arr)
    
    # Initial guess in Legendre coefficients
    # Project seed onto Legendre basis
    from numpy.linalg import lstsq
    cI_init, _, _, _ = lstsq(L_basis, I_seed)
    cQ_init, _, _, _ = lstsq(L_basis, Q_seed)
    
    x0 = np.concatenate([cI_init, cQ_init])
    
    # Bounds for Legendre coefficients (loose, will be enforced by objective)
    bounds = [(-1.0, 1.0)] * 16
    
    best_result = None
    best_score = compute_score(I_seed, Q_seed)
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6,
                                  'maxfun': 500})
        
        if time.time() < deadline - 60:  # Check if we have time
            if result.fun < best_score:
                cI_opt = result.x[:8]
                cQ_opt = result.x[8:]
                I_opt, Q_opt = legendre_to_pulse(cI_opt, cQ_opt)
                
                if check_validity(I_opt, Q_opt):
                    best_result = np.column_stack([I_opt, Q_opt])
                    best_score = result.fun
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    else:
        return np.column_stack([I_seed, Q_seed])

# EVOLVE-BLOCK-END
