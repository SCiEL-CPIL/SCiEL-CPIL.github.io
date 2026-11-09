"""Hermite-Gaussian basis expansion + Differential Evolution + L-BFGS-B polish for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import minimize, differential_evolution

# --- Constants from task ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
SCALE_VALUES = np.array([0.97, 1.0, 1.03])
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
MAX_DIFF = SLEW_BOUND * DT  # 0.06

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)
NN1 = N_MAT @ (N_MAT - I3)

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

# Precompute DISS_L (independent of H)
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

def build_liouvillian_full(H):
    coh_L = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh_L + DISS_L

def build_hamiltonian(detuning, scale, I, Q):
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT)))
    return H

# Precompute time grid
T_GRID = np.arange(N_STEPS) * DT

# Precompute Hermite-Gaussian basis functions (7 terms)
# We use a fixed center=20 ns, sigma=6 ns (close to seed)
# Basis: for k=0..3:
#   I_k(t) = H_k(x) * exp(-x^2/2)  where x = (t - center)/sigma
#   Q_k(t) = H_k(x) * exp(-x^2/2)  (same shape, but for Q channel)
# Plus one linear term for detuning compensation: L(t) = (t - center)/GATE_TIME
# Total 7+7+1 = 15 params? Let's reduce: use 4 I-terms (k=0,1,2,3), 3 Q-terms (k=0,1,2), 1 linear = 8 params
# Actually, let's use a symmetric set: 
# I: c0*H0*g + c1*H1*g + c2*H2*g + c3*H3*g  (4)
# Q: d0*H0*g + d1*H1*g + d2*H2*g  (3)
# Linear: e0*(t-center)  (1)
# Total: 8 parameters

CENTER = 20.0
SIGMA = 6.0
X_GRID = (T_GRID - CENTER) / SIGMA

def hermite(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    else:
        raise ValueError("n too large")

# Precompute basis matrices (80 x 8)
GAUSS = np.exp(-0.5 * X_GRID**2)
BASIS_I = np.column_stack([
    hermite(0, X_GRID) * GAUSS,
    hermite(1, X_GRID) * GAUSS,
    hermite(2, X_GRID) * GAUSS,
    hermite(3, X_GRID) * GAUSS,
])
BASIS_Q = np.column_stack([
    hermite(0, X_GRID) * GAUSS,
    hermite(1, X_GRID) * GAUSS,
    hermite(2, X_GRID) * GAUSS,
])
BASIS_LIN = (T_GRID - CENTER) / GATE_TIME  # 80,

# Full basis matrix: 80 x 8
BASIS = np.column_stack([BASIS_I, BASIS_Q, BASIS_LIN[:, None]])
# But we need separate I and Q. Let's restructure:
# I = c0*B0 + c1*B1 + c2*B2 + c3*B3 + e0*BL
# Q = d0*B0 + d1*B1 + d2*B2
# So 8 params: [c0,c1,c2,c3, d0,d1,d2, e0]

BASIS_I_FULL = np.column_stack([BASIS_I, BASIS_LIN[:, None]])  # 80 x 5
BASIS_Q_FULL = BASIS_Q  # 80 x 3

def generate_pulse_from_params(params):
    """
    params: [c0,c1,c2,c3, d0,d1,d2, e0]
    I = c0*B0 + c1*B1 + c2*B2 + c3*B3 + e0*BL
    Q = d0*B0 + d1*B1 + d2*B2
    """
    c0, c1, c2, c3, d0, d1, d2, e0 = params
    I = c0 * BASIS_I[:, 0] + c1 * BASIS_I[:, 1] + c2 * BASIS_I[:, 2] + c3 * BASIS_I[:, 3] + e0 * BASIS_LIN
    Q = d0 * BASIS_Q[:, 0] + d1 * BASIS_Q[:, 1] + d2 * BASIS_Q[:, 2]
    pulse = np.column_stack((I, Q))
    pulse = enforce_constraints(pulse)
    return pulse

def enforce_constraints(pulse):
    """Project pulse onto feasible set (amplitude + slew)."""
    pulse = pulse.copy()
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    # Enforce slew forward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        if diff_norm > MAX_DIFF:
            curr = prev + diff * (MAX_DIFF / diff_norm)
        pulse[i] = curr
        prev = curr
    # Enforce slew backward
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS - 1, -1, -1):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        if diff_norm > MAX_DIFF:
            curr = prev + diff * (MAX_DIFF / diff_norm)
        pulse[i] = curr
        prev = curr
    # Re-enforce amplitude
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    return pulse

def evolve_all_states(pulse, detuning, scale):
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    X3 = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=complex)
    
    exp_mats = []
    for i in range(N_STEPS):
        I_val, Q_val = pulse[i]
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        L = build_liouvillian_full(H)
        exp_mats.append(sla.expm(L * DT))
    
    fidelities = []
    for psi0 in states:
        rho0 = np.outer(psi0, psi0.conj())
        rho_vec = rho0.reshape(-1).copy()
        for i in range(N_STEPS):
            rho_vec = exp_mats[i] @ rho_vec
        rho = rho_vec.reshape(3, 3)
        psi_target = X3 @ psi0
        f = np.vdot(psi_target, rho @ psi_target).real
        f = np.clip(f, 0, 1)
        fidelities.append(f)
    return np.mean(fidelities)

def score_only(pulse):
    case_fidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            f = evolve_all_states(pulse, detuning, scale)
            case_fidelities.append(f)
    case_fidelities = np.array(case_fidelities)
    case_infidelities = 1.0 - case_fidelities
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    return mean_inf + max_inf

def check_validity(pulse):
    for i in range(len(pulse)):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse)):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    diff = np.linalg.norm(prev) / DT
    if diff > SLEW_BOUND + 1e-9:
        return False
    return True

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # 50s margin
    
    # 8 parameters: [c0,c1,c2,c3, d0,d1,d2, e0]
    # Initial guess: approximate a Gaussian DRAG pulse
    # Gaussian: c0 ~ 0.2, others small
    # DRAG: d1 (odd term) provides Q ~ derivative
    x0 = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    p0 = generate_pulse_from_params(x0)
    best_pulse = p0.copy()
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Objective
    def objective(x):
        p = generate_pulse_from_params(x)
        return score_only(p)
    
    # Bounds for 8 parameters
    # c0: [0, 0.4], c1-c3: [-0.2, 0.2], d0-d2: [-0.1, 0.1], e0: [-0.05, 0.05]
    bounds = [
        (0.0, 0.4),    # c0
        (-0.2, 0.2),   # c1
        (-0.2, 0.2),   # c2
        (-0.2, 0.2),   # c3
        (-0.1, 0.1),   # d0
        (-0.1, 0.1),   # d1
        (-0.1, 0.1),   # d2
        (-0.05, 0.05), # e0
    ]
    
    # Phase 1: Differential Evolution (global)
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=False,
            init='latinhypercube',
            updating='immediate',
            workers=1,
        )
        if de_result.fun < best_score:
            de_pulse = generate_pulse_from_params(de_result.x)
            de_score = score_only(de_pulse)
            if de_score < best_score:
                best_pulse = de_pulse
                best_score = de_score
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_pulse
    
    # Phase 2: L-BFGS-B local polish from DE result and from initial
    local_starts = [x0]
    if 'de_result' in dir() and de_result.success:
        local_starts.append(de_result.x)
    
    for ls in local_starts:
        if time.time() > deadline - 30:
            break
        try:
            lbfgs_result = minimize(
                objective,
                ls,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 500}
            )
            if lbfgs_result.fun < best_score:
                lbfgs_pulse = generate_pulse_from_params(lbfgs_result.x)
                lbfgs_score = score_only(lbfgs_pulse)
                if lbfgs_score < best_score:
                    best_pulse = lbfgs_pulse
                    best_score = lbfgs_score
        except Exception:
            pass
    
    return best_pulse
# EVOLVE-BLOCK-END
