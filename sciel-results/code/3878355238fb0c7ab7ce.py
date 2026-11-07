import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T_PHI = 40000.0

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Pre-compute operators
def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()
I3 = np.eye(3)

# Pre-compute Lindblad terms
# L1 = a / sqrt(T1), L2 = n * sqrt(2/Tphi)
L1 = a_op / np.sqrt(T1)
L2 = n_op * np.sqrt(2.0 / T_PHI)

# Superoperator for Lindblad part: S_l = sum (L^T \otimes L - 0.5 (I \otimes L^dag L + (L^dag L)^T \otimes I))
# Note: vec(rho) -> (U^T \otimes U) vec(rho)
# The Liouvillian step is:
# rho_vec_new = (U^T \otimes U) rho_vec + DT * S_l rho_vec
# We can pre-compute S_l once.

def build_lindblad_superoperator():
    dim = 3
    dim2 = dim * dim
    S_l = np.zeros((dim2, dim2), dtype=complex)
    for L in [L1, L2]:
        Ldag = L.conj().T
        # L rho L^dag -> (L^T \otimes L) vec(rho)
        S_l += np.kron(L.T, L)
        # -0.5 {L^dag L, rho}
        LdagL = Ldag @ L
        S_l -= 0.5 * (np.kron(I3, LdagL) + np.kron(LdagL.T, I3))
    return S_l

S_LINDBLAD = build_lindblad_superoperator()

def hamiltonian(detuning, scale, I, Q):
    # H = detuning * n + 0.5 * alpha * n(n-1) + 0.5 * scale * (I(a+a^dag) + iQ(a-a^dag))
    H = detuning * n_op + 0.5 * ANHARMONICITY * (n_op @ n_op - n_op)
    drive_term = I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op)
    H += 0.5 * scale * drive_term
    return H

def evolve_pulse(pulse, detuning, scale):
    """
    Evolve the state under the given pulse.
    pulse: (80, 2) array of (I, Q)
    Returns final density matrix (3x3).
    """
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    rho_vec = rho.flatten()
    
    for i in range(N_STEPS):
        I = pulse[i, 0]
        Q = pulse[i, 1]
        H = hamiltonian(detuning, scale, I, Q)
        U = expm(-1j * H * DT)
        
        # Unitary part superoperator
        S_u = np.kron(U.T, U)
        
        # Total superoperator for this step
        # Note: S_LINDBLAD is already scaled by rates, so we multiply by DT
        S_step = S_u + DT * S_LINDBLAD
        
        rho_vec = S_step @ rho_vec
    
    rho = rho_vec.reshape(3, 3)
    return rho

def compute_infidelities(pulse):
    """
    Compute infidelities for the 9 cases.
    Returns array of 9 infidelities.
    """
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target state for X gate on |0> is |1>
    # But the fidelity is measured against X applied to the *input* state.
    # The problem statement says: "infidelity = 1 - mean fidelity to X over |0>, |1>, ..."
    # This implies we simulate the evolution of the state |psi> and compare to X|psi>.
    # For |0>, X|0> = |1>.
    # For |1>, X|1> = |0>.
    # For (|0>+|1>)/sqrt(2), X(...) = (|1>+|0>)/sqrt(2).
    
    # However, the seed program only simulated starting from |0>.
    # "fidelity = 1 - mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # This means we need to compute the fidelity for all these input states.
    # Fidelity F = <psi_final| U_X^dag U U_X |psi_final>? No.
    # Fidelity between state rho and target state |phi> is <phi|rho|phi>.
    # Target state is X|psi_initial>.
    # So we need to evolve |psi_initial> to |psi_final> and compute <X psi_initial | psi_final>.
    
    infidelities = []
    
    # Define initial states
    states = [
        np.array([1, 0, 0]), # |0>
        np.array([0, 1, 0]), # |1>
        np.array([1, 1, 0]) / np.sqrt(2), # (|0>+|1>)/sqrt(2)
        np.array([1, -1, 0]) / np.sqrt(2), # (|0>-|1>)/sqrt(2)
        np.array([1, 1j, 0]) / np.sqrt(2), # (|0>+i|1>)/sqrt(2)
        np.array([1, -1j, 0]) / np.sqrt(2) # (|0>-i|1>)/sqrt(2)
    ]
    
    # Pre-compute target states (X applied to initial states)
    # X is the Pauli X matrix in the 2-level subspace.
    # In 3-level, X swaps |0> and |1>.
    X_mat = np.zeros((3,3), dtype=complex)
    X_mat[0,1] = 1.0
    X_mat[1,0] = 1.0
    
    for psi_init in states:
        target_state = X_mat @ psi_init
        
        for det in detunings:
            for sc in scales:
                # Evolve the state
                # We can evolve the density matrix, but for pure states, we can just evolve the vector.
                # However, the Hamiltonian is the same.
                # Let's evolve the vector to save time (3x3 matrix vec vs 9x1 vector).
                # Actually, the Lindblad terms make it a mixed state evolution if we consider noise.
                # But for fidelity of a pure state evolution under Lindblad, we should evolve the density matrix.
                
                rho = np.outer(psi_init, psi_init.conj())
                rho_vec = rho.flatten()
                
                for i in range(N_STEPS):
                    I = pulse[i, 0]
                    Q = pulse[i, 1]
                    H = hamiltonian(det, sc, I, Q)
                    U = expm(-1j * H * DT)
                    S_u = np.kron(U.T, U)
                    S_step = S_u + DT * S_LINDBLAD
                    rho_vec = S_step @ rho_vec
                
                rho_final = rho_vec.reshape(3, 3)
                
                # Fidelity with target state
                # F = <target | rho_final | target>
                fid = np.real(np.vdot(target_state, rho_final @ target_state))
                # Clip to [0, 1] to handle numerical errors
                fid = np.clip(fid, 0.0, 1.0)
                infid = 1.0 - fid
                infidelities.append(infid)
                
    return np.array(infidelities)

def check_validity(pulse):
    """Check if pulse satisfies amplitude and slew constraints."""
    # Amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-9):
        return False
    
    # Slew bound
    # From zero to step 0
    if np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between steps
    diffs = np.diff(pulse, axis=0)
    slews = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) / DT
    if np.any(slews > SLEW_BOUND + 1e-9):
        return False
        
    # From last step to zero
    if np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2) / DT > SLEW_BOUND + 1e-9:
        return False
        
    return True

def generate_pulse(sigma, beta):
    """
    Generate a DRAG Gaussian pulse.
    sigma: width
    beta: DRAG coefficient
    """
    t = np.linspace(0, TOTAL_TIME, N_STEPS)
    t_center = TOTAL_TIME / 2.0
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Normalize area to pi
    area = np.sum(gauss) * DT
    A = np.pi / area
    I = A * gauss
    
    # DRAG term: Q = beta * dI/dt
    # Approximate derivative
    dI_dt = np.zeros_like(I)
    for i in range(1, N_STEPS):
        dI_dt[i] = (I[i] - I[i-1]) / DT
    for i in range(N_STEPS - 1):
        dI_dt[i] = (I[i+1] - I[i-1]) / (2 * DT)
    dI_dt[0] = (I[1] - I[0]) / DT
    dI_dt[-1] = (I[-1] - I[-2]) / DT
    
    Q = beta * dI_dt
    
    pulse = np.column_stack([I, Q])
    
    # Ensure validity by scaling if necessary
    if not check_validity(pulse):
        # Scale down to satisfy amplitude
        max_amp = np.max(np.sqrt(I**2 + Q**2))
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp
            pulse *= scale
        
        # Check slew again
        if not check_validity(pulse):
            # Scale down further for slew
            # This is a crude fix, but should work for small violations
            diffs = np.diff(pulse, axis=0)
            slews = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2) / DT
            max_slew = np.max(slews)
            if max_slew > SLEW_BOUND:
                scale = SLEW_BOUND / max_slew
                pulse *= scale
            
            # Check boundary slews
            if np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2) / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / np.sqrt(pulse[0, 0]**2 + pulse[0, 1]**2)
                pulse *= scale
            if np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2) / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / np.sqrt(pulse[-1, 0]**2 + pulse[-1, 1]**2)
                pulse *= scale
                
    return pulse

def objective(params):
    """
    Objective function for optimization.
    params: [sigma, beta]
    """
    sigma, beta = params
    
    # Bounds for sigma and beta to keep search reasonable
    if sigma < 1.0 or sigma > 10.0:
        return 1e6
    if abs(beta) > 1.0:
        return 1e6
        
    pulse = generate_pulse(sigma, beta)
    
    if not check_validity(pulse):
        return 1e6
        
    try:
        infids = compute_infidelities(pulse)
        score = np.mean(infids) + np.max(infids)
        return score
    except:
        return 1e6

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes for the X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    start_time = time.time()
    deadline = start_time + 300.0 # 300s limit, keep margin
    
    # Initial guess
    # sigma ~ 4.0 ns, beta ~ 0.1
    x0 = np.array([4.0, 0.1])
    
    # Run optimization
    best_pulse = generate_pulse(4.0, 0.1)
    best_score = objective(x0)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 500})
        
        if result.success or result.fun < best_score:
            best_sigma, best_beta = result.x
            best_pulse = generate_pulse(best_sigma, best_beta)
            best_score = result.fun
    except:
        pass
        
    # If time is running out, we might want to do a quick polish or just return
    # But Nelder-Mead is relatively fast.
    
    return best_pulse
