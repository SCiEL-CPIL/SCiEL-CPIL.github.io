"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = 320.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute static parts of Liouvillian
_I3 = np.eye(3, dtype=complex)

# Dissipative part is constant (Lindblad operators don't depend on I, Q)
_L_DISS = (np.kron(_L1, _L1.conj()) - 0.5*(np.kron(_I3, _L1ADAG @ _L1) + np.kron(_L1ADAG @ _L1, _I3))
          + np.kron(_L2, _L2.conj()) - 0.5*(np.kron(_I3, _L2ADAG @ _L2) + np.kron(_L2ADAG @ _L2, _I3)))

# Precompute the static (detuning + anharmonicity) part for each detuning value
_STATIC_LIOUV = {}
for det in _DET_GRID:
    H_static = det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - _I3))
    L_static = -1j * (np.kron(H_static, _I3) - np.kron(_I3, H_static.T))
    _STATIC_LIOUV[det] = L_static + _L_DISS

# Precompute drive-related kron products
# H_drive = 0.5 * scale * (I*(a + a_dag) + i*Q*(a - a_dag))
_A_PLUS_ADAG = _A + _ADAG
_A_MINUS_ADAG = _A - _ADAG

# For the drive term: L_drive = -1j * (kron(H_drive, I) - kron(I, H_drive.T))
# = -1j * (I * kron(A_plus, I) - I * kron(I, A_plus.T) + i*Q * kron(A_minus, I) - i*Q * kron(I, A_minus.T))
# = -1j*I*kron(A_plus, I) + 1j*I*kron(I, A_plus.T) + Q*kron(A_minus, I) - Q*kron(I, A_minus.T)

_KRON_AP_I = np.kron(_A_PLUS_ADAG, _I3)
_KRON_I_APT = np.kron(_I3, _A_PLUS_ADAG.T)
_KRON_AM_I = np.kron(_A_MINUS_ADAG, _I3)
_KRON_I_AMT = np.kron(_I3, _A_MINUS_ADAG.T)

def _make_seed_pulse(N, dt, sigma_ns, area):
    """Create a DRAG-style Gaussian seed pulse."""
    t = np.arange(N) * dt
    center = N * dt / 2.0
    sigma = sigma_ns
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset so edges are zero
    edge_val = np.exp(-0.5 * ((t[0] - center) / sigma)**2)
    env = env - edge_val
    
    # Scale to get area = pi
    current_area = np.sum(env) * dt
    if current_area > 1e-10:
        env = env * (area / current_area)
    
    I = env
    Q = np.zeros(N)
    
    # Enforce constraints
    I, Q = _enforce_constraints(I, Q, _AMP, _SLEW, dt)
    
    return I, Q

def _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
    """Enforce amplitude and slew rate constraints by clipping."""
    I_arr = I_arr.copy()
    Q_arr = Q_arr.copy()
    N = len(I_arr)
    
    for _ in range(200):
        # Enforce amplitude
        for i in range(N):
            amp = np.sqrt(I_arr[i]**2 + Q_arr[i]**2)
            if amp > amp_bound:
                scale = amp_bound / amp
                I_arr[i] *= scale
                Q_arr[i] *= scale
        
        # Enforce slew rate - from zero to first
        amp0 = np.sqrt(I_arr[0]**2 + Q_arr[0]**2)
        if amp0 / dt > slew_bound:
            scale = slew_bound * dt / amp0
            I_arr[0] *= scale
            Q_arr[0] *= scale
        
        # Enforce slew rate - between steps
        for i in range(N - 1):
            dI = I_arr[i+1] - I_arr[i]
            dQ = Q_arr[i+1] - Q_arr[i]
            amp = np.sqrt(dI**2 + dQ**2)
            if amp / dt > slew_bound:
                scale = slew_bound * dt / amp
                I_arr[i+1] = I_arr[i] + dI * scale
                Q_arr[i+1] = Q_arr[i] + dQ * scale
        
        # Enforce slew rate - from last to zero
        amp_last = np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2)
        if amp_last / dt > slew_bound:
            scale = slew_bound * dt / amp_last
            I_arr[-1] *= scale
            Q_arr[-1] *= scale
        
        # Check if all constraints satisfied
        valid = True
        for i in range(N):
            if np.sqrt(I_arr[i]**2 + Q_arr[i]**2) > amp_bound + 1e-10:
                valid = False
                break
        if valid:
            for i in range(N - 1):
                dI = I_arr[i+1] - I_arr[i]
                dQ = Q_arr[i+1] - Q_arr[i]
                if np.sqrt(dI**2 + dQ**2) / dt > slew_bound + 1e-10:
                    valid = False
                    break
        if valid:
            if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / dt > slew_bound + 1e-10:
                valid = False
            if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / dt > slew_bound + 1e-10:
                valid = False
        if valid:
            break
    
    return I_arr, Q_arr

def _liouvillian_step_fast(dt, I, Q, detuning, scale):
    """Compute the Liouvillian superoperator for one step using precomputed parts."""
    # L_total = L_static[det] + L_drive
    # L_drive = -1j*I*scale/2 * KRON_AP_I + 1j*I*scale/2 * KRON_I_APT + Q*scale/2 * KRON_AM_I - Q*scale/2 * KRON_I_AMT
    
    L_total = _STATIC_LIOUV[detuning]
    
    if abs(I) > 1e-15 or abs(Q) > 1e-15:
        coeff_I = 0.5 * scale * I
        coeff_Q = 0.5 * scale * Q
        L_total = L_total + (-1j * coeff_I * _KRON_AP_I + 1j * coeff_I * _KRON_I_APT 
                            + coeff_Q * _KRON_AM_I - coeff_Q * _KRON_I_AMT)
    
    # Propagator: expm(L_total * dt)
    return expm(L_total * dt)

def _simulate_pulse(I_arr, Q_arr, detuning, scale):
    """Simulate the pulse and return the final state for each initial basis state."""
    N = len(I_arr)
    
    # Precompute all propagators
    props = []
    for i in range(N):
        prop = _liouvillian_step_fast(_DT, I_arr[i], Q_arr[i], detuning, scale)
        props.append(prop)
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    
    results = []
    for state in states:
        rho = np.outer(state, state.conj())
        vec_rho = rho.flatten()
        
        for prop in props:
            vec_rho = prop @ vec_rho
        
        rho_final = vec_rho.reshape(3, 3)
        results.append(rho_final)
    
    return results

def _compute_fidelity(rho_final, target_state):
    """Compute fidelity between density matrix and target pure state."""
    target_rho = np.outer(target_state, target_state.conj())
    return np.real(np.trace(target_rho @ rho_final))

def _score_pulse(I_arr, Q_arr):
    """Compute the score for a given pulse."""
    # Target states for X gate
    target_0 = np.array([0, 1, 0], dtype=complex)  # X|0> = |1>
    target_1 = np.array([1, 0, 0], dtype=complex)  # X|1> = |0>
    
    all_fidelities = []
    
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            results = _simulate_pulse(I_arr, Q_arr, det, scale)
            
            targets = [
                target_1,  # for |0>
                target_0,  # for |1>
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+|1>)/sqrt(2)
                (np.array([-1, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2)
                (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>+i|1>)/sqrt(2) -> (i|0>+|1>)/sqrt(2)
                (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),  # for (|0>-i|1>)/sqrt(2) -> (-i|0>+|1>)/sqrt(2)
            ]
            
            for rho_final, target in zip(results, targets):
                fid = _compute_fidelity(rho_final, target)
                all_fidelities.append(fid)
    
    all_fidelities = np.array(all_fidelities)
    mean_fidelity = np.mean(all_fidelities)
    max_fidelity = np.max(all_fidelities)
    
    infidelity_mean = 1.0 - mean_fidelity
    infidelity_max = 1.0 - max_fidelity
    
    score = infidelity_mean + infidelity_max
    return score

def _check_constraints(I_arr, Q_arr):
    """Check if the pulse satisfies amplitude and slew rate constraints."""
    N = len(I_arr)
    
    # Check amplitude
    for i in range(N):
        if np.sqrt(I_arr[i]**2 + Q_arr[i]**2) > _AMP + 1e-10:
            return False
    
    # Check slew rate including from zero
    for i in range(N - 1):
        dI = I_arr[i+1] - I_arr[i]
        dQ = Q_arr[i+1] - Q_arr[i]
        if np.sqrt(dI**2 + dQ**2) / _DT > _SLEW + 1e-10:
            return False
    
    # From zero to first step
    if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / _DT > _SLEW + 1e-10:
        return False
    
    # From last step to zero
    if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / _DT > _SLEW + 1e-10:
        return False
    
    return True

def _drag_pulse(N, dt, sigma_ns, area, alpha):
    """Create a DRAG pulse with Gaussian envelope and alpha derivative term."""
    t = np.arange(N) * dt
    center = N * dt / 2.0
    sigma = sigma_ns
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset so edges are zero
    edge_val = np.exp(-0.5 * ((t[0] - center) / sigma)**2)
    env = env - edge_val
    
    # Derivative of Gaussian (for DRAG Q component)
    denv_dt = -((t - center) / sigma**2) * np.exp(-0.5 * ((t - center) / sigma)**2)
    denv_dt = denv_dt - denv_dt[0]  # offset to zero at edges
    
    # Scale I to get area = pi
    current_area = np.sum(env) * dt
    if current_area > 1e-10:
        env = env * (area / current_area)
    
    I = env
    Q = alpha * denv_dt
    
    # Enforce constraints
    I, Q = _enforce_constraints(I, Q, _AMP, _SLEW, dt)
    
    return I, Q

def pulse(target_gate, hardware_model):
    """Main entry point. Returns an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    # Start with seed pulse
    I, Q = _make_seed_pulse(_N, _DT, 40/6, np.pi)
    
    best_I, best_Q = I.copy(), Q.copy()
    best_score = _score_pulse(best_I, best_Q)
    
    # Try DRAG pulses with different alpha values
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3]
    sigmas = [40/6, 40/5, 40/4, 40/7, 40/8]
    
    for sigma in sigmas:
        for alpha in alphas:
            if _time_remaining() < 60:
                break
            I_test, Q_test = _drag_pulse(_N, _DT, sigma, np.pi, alpha)
            if not _check_constraints(I_test, Q_test):
                continue
            score = _score_pulse(I_test, Q_test)
            if score < best_score:
                best_score = score
                best_I, best_Q = I_test.copy(), Q_test.copy()
    
    # Local optimization around best pulse
    n_iter = 0
    max_iter = 100
    step_size = 0.02
    
    while _time_remaining() > 30 and n_iter < max_iter:
        n_iter += 1
        
        # Try random perturbations
        improved = False
        for _ in range(10):
            I_test = best_I + np.random.randn(_N) * step_size
            Q_test = best_Q + np.random.randn(_N) * step_size
            
            I_test, Q_test = _enforce_constraints(I_test, Q_test, _AMP, _SLEW, _DT)
            
            if not _check_constraints(I_test, Q_test):
                continue
            
            score = _score_pulse(I_test, Q_test)
            
            if score < best_score:
                best_score = score
                best_I, best_Q = I_test.copy(), Q_test.copy()
                improved = True
        
        if improved:
            step_size = max(0.001, step_size * 0.8)
        else:
            step_size = min(0.05, step_size * 1.3)
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
