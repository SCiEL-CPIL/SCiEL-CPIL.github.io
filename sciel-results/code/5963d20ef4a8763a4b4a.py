"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
T2 = 40000.0

N_STEPS = 80
DT = 0.5
DIM = 3

# Build ladder operators for 3-level system
def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

# Precompute the dissipation superoperator (independent of detuning, scale, I, Q)
def build_dissipation_superop():
    dim = DIM
    L_diss = np.zeros((dim * dim, dim * dim), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    eye = np.eye(dim)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L_dag_L = L_dag @ L_op
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for m in range(dim):
                        L_diss[i * dim + j, k * dim + m] += L_op[i, k] * L_dag[m, j]
                        L_diss[i * dim + j, k * dim + m] -= 0.5 * L_dag_L[i, k] * (eye[j, m] + eye[m, j])
    return L_diss

L_DISSIPATION = build_dissipation_superop()

# Precompute the commutator superoperator for a general Hamiltonian H:
# L_coh(rho) = -i [H, rho], so in vectorized form:
# (L_coh)_{ij,kl} = -i * H_{ik} * delta_{jl} + i * H_{kl} * delta_{ij}
# We'll build a function that takes H and returns the 9x9 superoperator.

def hamiltonian_superop(H):
    """Build the -i[H, .] superoperator from a 3x3 Hamiltonian H."""
    dim = DIM
    S = np.zeros((dim * dim, dim * dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                S[i * dim + j, k * dim + j] -= 1j * H[i, k]
                S[i * dim + j, k * dim + i] += 1j * H[k, j]
    return S

# Precompute the drive superoperator basis:
# H_drive = 0.5 * scale * (I * (a + a†) + iQ * (a - a†))
# We can write H_drive = c_I * I * H_I + c_I * Q * H_Q where c_I = 0.5*scale
# Actually: H_drive = 0.5*scale*I*(a+a†) + 0.5*scale*iQ*(a-a†)
# Let A = a + a† (Hermitian) and B = i*(a - a†) (also Hermitian since a-a† is anti-Hermitian, i*anti-Hermitian is Hermitian)
# Then H_drive = 0.5*scale*(I*A + Q*B)
# So L_drive = 0.5*scale*(I * L_A + Q * L_B) where L_A = -i[A,.] and L_B = -i[B,.]

A_mat = a_op + a_dag_op
B_mat = 1j * (a_op - a_dag_op)

L_A = hamiltonian_superop(A_mat)
L_B = hamiltonian_superop(B_mat)

# Detuning superoperator: H_det = detuning * n_op
# L_det = -i * detuning * [n_op, .]
L_N = hamiltonian_superop(n_op)

# Anharmonicity superoperator: H_anh = 0.5 * ANHARMONICITY * n_op @ (n_op - I)
# n_op @ (n_op - I) is diagonal: diag(0, 0, 2)
# So H_anh is diagonal, and [diag, rho] is simple
n_minus_1 = n_op - np.eye(DIM)
anh_term = n_op @ n_minus_1  # diag(0, 0, 2)
H_anh = 0.5 * ANHARMONICITY * anh_term
L_ANH = hamiltonian_superop(H_anh)

# Total static (non-drive) part: L_static = L_ANH + detuning * L_N + L_DISSIPATION
# Drive part: L_drive = 0.5 * scale * (I * L_A + Q * L_B)

def build_liouvillian_fast(detuning, scale, I, Q):
    """Build full Liouvillian using precomputed components."""
    L = L_DISSIPATION + L_ANH + detuning * L_N
    L += 0.5 * scale * (I * L_A + Q * L_B)
    return L

def check_validity(pulse):
    """Check if the pulse satisfies all constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    diffs = np.diff(pulse, axis=0, prepend=np.zeros((1, 2)))
    slew = np.linalg.norm(diffs, axis=1) / DT
    if np.any(slew > SLEW_BOUND + 1e-10):
        return False
    if np.linalg.norm(pulse[-1]) / DT > SLEW_BOUND + 1e-10:
        return False
    return True

def seed_pulse():
    """DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, N_STEPS * DT, N_STEPS)
    sigma = 40.0 / 6.0
    center = N_STEPS * DT / 2.0
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_offset = env[0]
    env = env - edge_offset
    env = np.maximum(env, 0)
    area = np.sum(env) * DT
    if area > 0:
        env = env * (np.pi / area)
    denv = np.gradient(env, DT)
    drag_coeff = 0.1
    I = env
    Q = drag_coeff * denv
    pulse = np.column_stack([I, Q])
    if not check_validity(pulse):
        max_amp = np.max(np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            pulse *= AMPLITUDE_BOUND / max_amp
        for i in range(1, N_STEPS):
            delta = np.linalg.norm(pulse[i] - pulse[i-1]) / DT
            if delta > SLEW_BOUND:
                scale = SLEW_BOUND / delta
                pulse[i:] *= scale
    return pulse

# Precompute all 54 initial density matrices (6 states x 9 cases don't matter, just 6 states)
STATES = [
    np.array([1, 0, 0], dtype=complex),
    np.array([0, 1, 0], dtype=complex),
    (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
]
TARGET_RHOS = []
for st in STATES:
    ts = np.zeros(3, dtype=complex)
    ts[0] = st[1]
    ts[1] = st[0]
    ts[2] = st[2]
    TARGET_RHOS.append(np.outer(ts, ts.conj()))

INITIAL_RHOS = [np.outer(st, st.conj()) for st in STATES]
INITIAL_RHOS_VEC = np.array([r.flatten() for r in INITIAL_RHOS])  # shape (6, 9)

# Precompute trace coefficients for fidelity:
# fid = Tr(rho @ target_rho) = sum_{ij} rho_{ij} * target_rho_{ji}
# In vectorized form: fid = sum_{i,j} rho_vec[i*3+j] * target_rho.T.flatten()[i*3+j]
# Actually: Tr(rho @ T) = sum_{i,j} rho_{ij} T_{ji} = sum_{i,j} rho_vec[i*3+j] * T_vec[j*3+i]
# So fid = rho_vec @ T_vec.T.conj()? Let's just precompute.
FID_COEFFS = np.zeros((6, 9), dtype=complex)
for s in range(6):
    T = TARGET_RHOS[s]
    # fid = Tr(rho @ T) = sum_{i,j} rho_{ij} T_{ji}
    # rho_vec index: i*3+j, T_{ji} index: j*3+i
    for i in range(3):
        for j in range(3):
            FID_COEFFS[s, i*3+j] = T[j, i]

def compute_score_fast(pulse):
    """Fast vectorized score computation."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    case_infidelities = []
    
    for det in detunings:
        for scale in scales:
            # Evolve all 6 states simultaneously
            # rhos_vec: shape (6, 9) - 6 states, each as 9-element vector
            rhos_vec = INITIAL_RHOS_VEC.copy()
            
            for i in range(N_STEPS):
                I_val = pulse[i, 0]
                Q_val = pulse[i, 1]
                L = build_liouvillian_fast(det, scale, I_val, Q_val)
                U = expm(L * DT)
                # rhos_vec is (6, 9), U is (9, 9)
                # New rho_vec = U @ old_rho_vec for each state
                rhos_vec = rhos_vec @ U.T  # (6, 9) @ (9, 9) = (6, 9)
                # Wait: rho_new = U @ rho_old, so rho_new_vec = U @ rho_old_vec
                # If rhos_vec[s] = rho_old_vec for state s, then new = U @ rhos_vec[s]
                # So rhos_vec_new = rhos_vec @ U.T
            
            # Compute fidelities
            fids = np.real(rhos_vec @ FID_COEFFS.T)  # (6, 9) @ (9, 6) = (6, 6)? No.
            # Let me redo: fid_s = sum_k rhos_vec[s, k] * FID_COEFFS[s, k]
            fids = np.real(np.sum(rhos_vec * FID_COEFFS, axis=1))  # shape (6,)
            
            mean_fid = np.mean(fids)
            infid = 1.0 - mean_fid
            case_infidelities.append(infid)
    
    score = np.mean(case_infidelities) + np.max(case_infidelities)
    return score

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Start with seed
    best_pulse = seed_pulse()
    best_score = compute_score_fast(best_pulse)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 90:
        return best_pulse
    
    # Phase 1: Nelder-Mead on a reduced parameterization
    # Use a smooth parameterization: pulse is sum of Gaussian basis functions
    # p(t) = sum_m c_m * gauss_m(t) where gauss_m are shifted/scaled Gaussians
    # This ensures smoothness and helps with slew constraints
    
    # Actually, let's use a direct approach but with better initialization
    # Try several DRAG variants with different parameters
    
    np.random.seed(42)
    
    # Generate diverse DRAG seeds
    candidates = [best_pulse]
    for sigma_mult in [0.8, 1.0, 1.2, 1.5]:
        for drag_coeff in [0.0, 0.05, 0.1, 0.15, 0.2]:
            t = np.linspace(0, N_STEPS * DT, N_STEPS)
            sigma = 40.0 / 6.0 * sigma_mult
            center = N_STEPS * DT / 2.0
            env = np.exp(-0.5 * ((t - center) / sigma)**2)
            edge_offset = env[0]
            env = env - edge_offset
            env = np.maximum(env, 0)
            area = np.sum(env) * DT
            if area > 0:
                env = env * (np.pi / area)
            denv = np.gradient(env, DT)
            I = env
            Q = drag_coeff * denv
            cand = np.column_stack([I, Q])
            if check_validity(cand):
                candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        sc = compute_score_fast(cand)
        if sc < best_score:
            best_score = sc
            best_pulse = cand
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_pulse
    
    # Phase 2: Local optimization via coordinate descent / random search
    # Use the faster score function
    
    def objective(params):
        pulse_arr = params.reshape(N_STEPS, 2)
        if not check_validity(pulse_arr):
            return 1e10
        return compute_score_fast(pulse_arr)
    
    x0 = best_pulse.flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        if result.fun < best_score:
            candidate = result.x.reshape(N_STEPS, 2)
            if check_validity(candidate):
                best_pulse = candidate
                best_score = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_pulse
    
    # Phase 3: Random local search with adaptive step size
    step_size = 0.015
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(500):
        if time.time() > deadline - 20:
            break
        if no_improve_count > max_no_improve:
            break
        
        # Adaptive step size
        if iteration % 30 == 0 and iteration > 0:
            step_size *= 0.85
        
        perturbation = np.random.randn(N_STEPS, 2) * step_size
        candidate = best_pulse + perturbation
        
        # Clip to amplitude bound
        amps = np.sqrt(candidate[:, 0]**2 + candidate[:, 1]**2)
        mask = amps > AMPLITUDE_BOUND
        if np.any(mask):
            candidate[mask] *= AMPLITUDE_BOUND / amps[mask]
        
        if check_validity(candidate):
            score = compute_score_fast(candidate)
            if score < best_score:
                best_pulse = candidate
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    return best_pulse
# EVOLVE-BLOCK-END
