"""Direct 160-dim raw pulse optimization with fast Liouvillian for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _precompute_ops():
    """Precompute all constant operators."""
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    n2 = n_op @ n_op
    n_n1 = n2 - n_op  # n(n-1)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    # Precompute Liouvillian superoperator basis for identity and n_op
    I3 = np.eye(3, dtype=complex)
    # Precompute Lindblad terms
    T1 = 25000.0
    Tphi = 40000.0
    L1 = a / np.sqrt(T1)
    L1_dag = L1.conj().T
    L1_dag_L1 = L1_dag @ L1
    L2 = n_op * np.sqrt(2.0 / Tphi)
    L2_dag = L2.conj().T
    L2_dag_L2 = L2_dag @ L2
    return a, a_dag, n_op, n_n1, a_plus_adag, a_minus_adag, I3, L1, L1_dag, L1_dag_L1, L2, L2_dag, L2_dag_L2

def _build_liouvillian_fast(I, Q, detuning, anharmonicity, scale, ops):
    """Build 9x9 Liouvillian superoperator efficiently."""
    a, a_dag, n_op, n_n1, a_plus_adag, a_minus_adag, I3, L1, L1_dag, L1_dag_L1, L2, L2_dag, L2_dag_L2 = ops
    H = (detuning * n_op 
         + 0.5 * anharmonicity * n_n1
         + 0.5 * scale * (I * a_plus_adag + 1j * Q * a_minus_adag))
    
    # Build superoperator: for each basis element rho_ij, compute L(rho_ij)
    dim = 9
    L_super = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        e = np.zeros(9, dtype=complex)
        e[col] = 1.0
        rho = e.reshape(3, 3)
        # Coherent: -i[H, rho]
        coherent = (-1j * (H @ rho - rho @ H)).flatten()
        # Dissipative: L1 rho L1^dag - 0.5{L1^dag L1, rho} + L2 rho L2^dag - 0.5{L2^dag L2, rho}
        diss = (L1 @ rho @ L1_dag - 0.5 * (L1_dag_L1 @ rho + rho @ L1_dag_L1)
                + L2 @ rho @ L2_dag - 0.5 * (L2_dag_L2 @ rho + rho @ L2_dag_L2)).flatten()
        L_super[:, col] = coherent + diss
    return L_super

def _evolve_and_score(pulse_flat, detuning, anharmonicity, scale, T1, Tphi, dt, ops):
    """Evolve all 5 states and compute mean infidelity. pulse_flat is (160,) = 80 I + 80 Q."""
    n_steps = 80
    I_arr = pulse_flat[:80]
    Q_arr = pulse_flat[80:]
    
    # Initial states as density matrices (flattened)
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_p = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_m = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_ip = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    
    states = np.array([
        np.outer(p, p.conj()).flatten() for p in [psi0, psi1, psi_p, psi_m, psi_ip]
    ], dtype=complex)  # (5, 9)
    
    for step in range(n_steps):
        L_super = _build_liouvillian_fast(I_arr[step], Q_arr[step], detuning, anharmonicity, scale, ops)
        U_step = expm(L_super * dt)
        states = (U_step @ states.T).T  # (5, 9)
    
    # Compute fidelities
    X_op = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    psi_list = [psi0, psi1, psi_p, psi_m, psi_ip]
    fidelities = []
    for i in range(5):
        rho_final = states[i].reshape(3, 3)
        target = X_op @ psi_list[i]
        rho_target = np.outer(target, target.conj())
        fid = np.real(np.trace(rho_final @ rho_target))
        fidelities.append(fid)
    
    return np.mean(fidelities)

def _score_full(pulse_flat, ops):
    """Full score over 9 detuning/scale cases."""
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fid = _evolve_and_score(pulse_flat, det, anharmonicity, sc, T1, Tphi, dt, ops)
            infidelities.append(1.0 - fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _check_validity(pulse_IQ, dt=0.5):
    """Check amplitude and slew rate constraints."""
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amps > 0.4 + 1e-10):
        return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse_IQ)):
        curr = pulse_IQ[i]
        if np.linalg.norm(curr - prev) / dt > 0.12 + 1e-10:
            return False
        prev = curr
    if np.linalg.norm(prev) / dt > 0.12 + 1e-10:
        return False
    return True

def _enforce_constraints(pulse_IQ):
    """Project pulse onto feasible set (simple clipping)."""
    pulse_IQ = pulse_IQ.copy()
    # Amplitude constraint
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    mask = amps > 0.4
    if np.any(mask):
        scale = 0.4 / amps[mask]
        pulse_IQ[mask, 0] *= scale
        pulse_IQ[mask, 1] *= scale
    # Slew rate constraint: iteratively reduce
    for _ in range(50):
        if _check_validity(pulse_IQ, 0.5):
            break
        # Find worst violation and reduce
        prev = np.array([0.0, 0.0])
        for i in range(len(pulse_IQ)):
            curr = pulse_IQ[i]
            diff = curr - prev
            norm_diff = np.linalg.norm(diff)
            if norm_diff / 0.5 > 0.12:
                factor = 0.12 * 0.5 / norm_diff
                pulse_IQ[i] = prev + diff * factor
            prev = pulse_IQ[i]
        # Check last to zero
        if np.linalg.norm(pulse_IQ[-1]) / 0.5 > 0.12:
            factor = 0.12 * 0.5 / np.linalg.norm(pulse_IQ[-1])
            pulse_IQ[-1] *= factor
    return pulse_IQ

def _seed_pulse():
    """Generate seed DRAG pulse: Gaussian with sigma=40/6, area=pi."""
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt - 20.0
    sigma = 40.0 / 6.0
    A = np.pi / (sigma * np.sqrt(2 * np.pi))
    gauss = A * np.exp(-t**2 / (2 * sigma**2))
    I = gauss
    Q = np.zeros(n_steps)
    pulse_IQ = np.column_stack([I, Q])
    return _enforce_constraints(pulse_IQ)

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    ops = _precompute_ops()
    
    # Start with seed
    seed_IQ = _seed_pulse()
    seed_flat = seed_IQ.flatten()
    
    if not _check_validity(seed_IQ, 0.5):
        # Fallback
        seed_IQ = np.zeros((80, 2))
        seed_IQ[10:70, 0] = np.pi / 30.0
        seed_IQ = _enforce_constraints(seed_IQ)
        seed_flat = seed_IQ.flatten()
    
    best_flat = seed_flat.copy()
    best_IQ = seed_IQ.copy()
    best_score = _score_full(seed_flat, ops)
    
    # Powell optimization on 160-dim raw pulse
    # Use multiple restarts with small perturbations
    np.random.seed(123)
    
    n_restarts = 5
    max_iter_per_restart = 80
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x0 = best_flat.copy()
        else:
            # Perturb from best
            x0 = best_flat + np.random.normal(0, 0.01, size=160)
            x0_IQ = x0.reshape(80, 2)
            x0_IQ = _enforce_constraints(x0_IQ)
            x0 = x0_IQ.flatten()
        
        # Objective function
        def obj(x):
            x_IQ = x.reshape(80, 2)
            x_IQ = _enforce_constraints(x_IQ)
            x_valid = x_IQ.flatten()
            if not _check_validity(x_IQ, 0.5):
                return 10.0
            return _score_full(x_valid, ops)
        
        try:
            result = minimize(
                obj,
                x0,
                method='Powell',
                options={'maxiter': max_iter_per_restart, 'xatol': 1e-6, 'fatol': 1e-8,
                         'maxfev': 300}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_flat = result.x
                best_IQ = _enforce_constraints(result.x.reshape(80, 2))
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final refinement with tighter tolerance
    if time.time() < deadline - 15:
        x0 = best_flat.copy()
        x0_IQ = x0.reshape(80, 2)
        x0_IQ = _enforce_constraints(x0_IQ)
        x0 = x0_IQ.flatten()
        
        def obj2(x):
            x_IQ = x.reshape(80, 2)
            x_IQ = _enforce_constraints(x_IQ)
            x_valid = x_IQ.flatten()
            if not _check_validity(x_IQ, 0.5):
                return 10.0
            return _score_full(x_valid, ops)
        
        try:
            result2 = minimize(
                obj2,
                x0,
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9,
                         'maxfev': 200}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_flat = result2.x
                best_IQ = _enforce_constraints(result2.x.reshape(80, 2))
        except Exception:
            pass
    
    # Final validity check
    final_IQ = best_IQ
    if not _check_validity(final_IQ, 0.5):
        final_IQ = _enforce_constraints(final_IQ)
        if not _check_validity(final_IQ, 0.5):
            final_IQ = seed_IQ
            if not _check_validity(final_IQ, 0.5):
                final_IQ = np.zeros((80, 2))
                final_IQ[10:70, 0] = np.pi / 30.0
                final_IQ = _enforce_constraints(final_IQ)
    
    return final_IQ

# EVOLVE-BLOCK-END
