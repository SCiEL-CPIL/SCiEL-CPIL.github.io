import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

# Constants
DT = 0.5
N_STEPS = 80
T_TOTAL = N_STEPS * DT
ALPHA_ANH = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
TPHI = 40000.0

# Hardware cases
DELTAS = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]

# Basis states
# |0> = [1, 0, 0]
# |1> = [0, 1, 0]
# |+> = [1, 1, 0] / sqrt(2)
# |-> = [1, -1, 0] / sqrt(2)
# |+i> = [1, i, 0] / sqrt(2)
# |-i> = [1, -i, 0] / sqrt(2)

def get_initial_states():
    s0 = np.array([1, 0, 0], dtype=complex)
    s1 = np.array([0, 1, 0], dtype=complex)
    sp = (s0 + s1) / np.sqrt(2)
    sm = (s0 - s1) / np.sqrt(2)
    spi = (s0 + 1j * s1) / np.sqrt(2)
    smi = (s0 - 1j * s1) / np.sqrt(2)
    return np.array([s0, s1, sp, sm, spi, smi])

def get_target_states():
    # X-pi gate: |0> -> -i|1>, |1> -> -i|0>
    # Wait, X-pi is usually defined as exp(-i pi/2 X) = -i X.
    # X |0> = |1>, X |1> = |0>.
    # So -i X |0> = -i |1>.
    # -i X |1> = -i |0>.
    # Let's check the fidelity definition. "fidelity to X".
    # Usually, this means the overlap with the state evolved by the ideal X-pi gate.
    t0 = -1j * np.array([0, 1, 0], dtype=complex)
    t1 = -1j * np.array([1, 0, 0], dtype=complex)
    tp = -1j * (np.array([0, 1, 0]) + np.array([1, 0, 0])) / np.sqrt(2)
    tm = -1j * (np.array([0, 1, 0]) - np.array([1, 0, 0])) / np.sqrt(2)
    tpi = -1j * (np.array([0, 1, 0]) + 1j * np.array([1, 0, 0])) / np.sqrt(2)
    tmi = -1j * (np.array([0, 1, 0]) - 1j * np.array([1, 0, 0])) / np.sqrt(2)
    return np.array([t0, t1, tp, tm, tpi, tmi])

# Pre-compute operators
def build_operators():
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = np.sqrt(1)
    a[1, 2] = np.sqrt(2)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

A, ADAG, N_OP = build_operators()

def build_liouvillian(I, Q, delta, scale):
    # H = delta * n + 0.5 * alpha * n * (n - 1) + 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    H = delta * N_OP + 0.5 * ALPHA_ANH * N_OP @ (N_OP - np.eye(3)) + 0.5 * scale * (I * (A + ADAG) + 1j * Q * (A - ADAG))
    
    # Lindblad operators
    L1 = A / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2 / TPHI)
    
    # Liouvillian superoperator
    # L(rho) = -i [H, rho] + sum (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # We can represent this as a 9x9 matrix acting on vec(rho)
    # vec(A rho B) = (B^T \otimes A) vec(rho)
    
    L_mat = np.zeros((9, 9), dtype=complex)
    
    # Coherent part
    H_kron = np.kron(np.eye(3), H) - np.kron(H.conj().T, np.eye(3))
    L_mat += -1j * H_kron
    
    # Dissipative part
    for L_k in [L1, L2]:
        L_k_dag = L_k.conj().T
        # L_k rho L_k^dag
        term1 = np.kron(L_k_dag.conj().T, L_k)
        # 0.5 {L_k^dag L_k, rho}
        L_k_dag_L_k = L_k_dag @ L_k
        term2 = 0.5 * (np.kron(np.eye(3), L_k_dag_L_k) + np.kron(L_k_dag_L_k.conj().T, np.eye(3)))
        L_mat += term1 - term2
        
    return L_mat

def simulate_pulse(I, Q):
    # I, Q are arrays of length N_STEPS
    # Returns a list of infidelities for the 9 cases
    infidelities = []
    init_states = get_initial_states()
    target_states = get_target_states()
    
    # Vectorize initial states: shape (6, 3, 3)
    rho_init = np.zeros((6, 3, 3), dtype=complex)
    for i, s in enumerate(init_states):
        rho_init[i] = np.outer(s, s.conj().T)
        
    # We need to simulate for each of the 9 cases
    for delta in DELTAS:
        for scale in SCALES:
            # Stack rho for all 6 states
            rho = rho_init.copy()
            
            for k in range(N_STEPS):
                L_k = build_liouvillian(I[k], Q[k], delta, scale)
                U_k = expm(L_k * DT)
                # Apply U_k to each state
                # rho is (6, 3, 3). We can reshape to (6, 9)
                rho_flat = rho.reshape(6, 9)
                rho_flat = U_k @ rho_flat.T
                rho = rho_flat.T.reshape(6, 3, 3)
                
            # Calculate fidelity for each state
            fidelities = []
            for i in range(6):
                rho_final = rho[i]
                t = target_states[i]
                # Fidelity F = <psi| rho |psi>
                f = np.real(np.vdot(t, rho_final @ t))
                fidelities.append(f)
                
            mean_fid = np.mean(fidelities)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
            
    return infidelities

def check_constraints(I, Q):
    # Amplitude bound
    amps = np.sqrt(I**2 + Q**2)
    if np.any(amps > AMP_BOUND):
        return False
        
    # Slew bound
    # ||p[i+1] - p[i]|| / dt <= 0.12
    # p = [I, Q]
    dI = np.diff(I)
    dQ = np.diff(Q)
    d_norms = np.sqrt(dI**2 + dQ**2)
    if np.any(d_norms / DT > SLEW_BOUND):
        return False
        
    # From zero to step 1
    if np.sqrt(I[0]**2 + Q[0]**2) / DT > SLEW_BOUND:
        return False
    # From last step to zero
    if np.sqrt(I[-1]**2 + Q[-1]**2) / DT > SLEW_BOUND:
        return False
        
    return True

def generate_pulse(sigma, beta):
    t = np.linspace(0, T_TOTAL, N_STEPS)
    t_center = T_TOTAL / 2
    t = t - t_center
    
    A_amp = np.sqrt(np.pi / 2) / sigma
    I = A_amp * np.exp(-t**2 / (2 * sigma**2))
    Q = -beta * t / (sigma**2) * I
    
    return I, Q

def objective(params):
    sigma, beta = params
    if sigma < 1.0 or sigma > 10.0:
        return 1e6
    if beta < -50.0 or beta > 50.0:
        return 1e6
        
    I, Q = generate_pulse(sigma, beta)
    
    if not check_constraints(I, Q):
        return 1e6
        
    infids = simulate_pulse(I, Q)
    score = np.mean(infids) + np.max(infids)
    return score

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Initial guess
    # Typical DRAG sigma is around 5-10 ns
    # Beta is around 10-20 ns
    
    # Use differential evolution
    bounds = [(2.0, 10.0), (-20.0, 20.0)]
    
    # To save time, we can use a smaller population and maxiter
    # But we need to be robust.
    # Let's try a local optimization first with a few starting points.
    
    best_score = 1e6
    best_params = (5.0, 10.0)
    
    # Try a few random starts
    starts = [
        (4.0, 5.0),
        (5.0, 10.0),
        (6.0, 15.0),
        (5.0, 0.0),
        (5.0, -10.0)
    ]
    
    from scipy.optimize import minimize
    
    for s in starts:
        res = minimize(objective, s, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
        if time.time() > deadline - 30.0:
            break
            
    # If we have time, try a wider search
    if time.time() < deadline - 60.0:
        try:
            res_de = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=10, tol=1e-4, polish=True)
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_params = res_de.x
        except Exception:
            pass
            
    sigma, beta = best_params
    I, Q = generate_pulse(sigma, beta)
    
    # Final check
    if not check_constraints(I, Q):
        # Fallback to a simple safe pulse
        I = np.zeros(N_STEPS)
        Q = np.zeros(N_STEPS)
        # Make it a valid pulse with area pi
        # Use a rectangular pulse? No, slew rate.
        # Use a very wide Gaussian
        sigma_safe = 8.0
        beta_safe = 0.0
        I, Q = generate_pulse(sigma_safe, beta_safe)
        
    return np.column_stack((I, Q))
