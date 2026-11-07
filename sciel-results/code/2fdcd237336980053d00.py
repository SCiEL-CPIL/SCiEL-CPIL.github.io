"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = 310.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# 3-level operators
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute superoperator pieces
# Liouvillian L = -i(H⊗I - I⊗H†) + sum_k (L_k⊗L_k† ρ L_k⊗L_k†† - 0.5{L_k†L_k, ρ})
# In superoperator form: L = -i(H⊗I - I⊗H†) + sum_k (L_k⊗L_k† - 0.5 L_k†L_k⊗I - 0.5 I⊗L_k†L_k)
# Actually for density matrix evolution: dρ/dt = -i[H,ρ] + sum_k (L_k ρ L_k† - 0.5{L_k†L_k, ρ})
# Superoperator: S = -i(H⊗I - I⊗H†) + sum_k (L_k⊗L_k† - 0.5 L_k†L_k⊗I - 0.5 I⊗L_k†L_k)

# Precompute constant parts
_H0_ID = np.eye(9)
_ID_H0 = np.eye(9)

# For each case, H depends on detuning and scale, and I,Q at each step
# H(t) = detuning * n + 0.5*anh*n*(n-1) + 0.5*scale*(I*(a+a†) + i*Q*(a-a†))

# Precompute n*(n-1)
_NNM1 = _NOP @ (_NOP - np.eye(3))

# Precompute a+a† and a-a†
_A_ADAG = _A + _ADAG
_A_MINUS_ADAG = _A - _ADAG

# Precompute Lindblad superoperator constant parts
_L1T_L1DAG = np.kron(_L1, _L1ADAG)
_L2T_L2DAG = np.kron(_L2, _L2ADAG)
_L1DAG_L1 = _L1ADAG @ _L1
_L2DAG_L2 = _L2ADAG @ _L2
_LIND_CONST = (_L1T_L1DAG + _L2T_L2DAG - 
               0.5 * np.kron(_L1DAG_L1, np.eye(3)) - 0.5 * np.kron(np.eye(3), _L1DAG_L1) -
               0.5 * np.kron(_L2DAG_L2, np.eye(3)) - 0.5 * np.kron(np.eye(3), _L2DAG_L2))


def _make_seed_pulse():
    """DRAG-style Gaussian seed with area pi."""
    t = np.arange(_N) * _DT
    center = _T / 2.0
    sigma = 40.0 / 6.0
    # Gaussian with area pi
    gauss = np.exp(-0.5 * ((t - center) / sigma)**2)
    # Normalize area to pi
    area = np.sum(gauss) * _DT
    gauss = gauss * (np.pi / area)
    # Offset to 0 at edges (Gaussian already ~0 at edges for sigma=6.67, T=40)
    # Clip to amplitude bound
    gauss = np.clip(gauss, 0, _AMP)
    I = gauss
    Q = np.zeros(_N)
    return np.column_stack([I, Q])


def _enforce_constraints(pulse):
    """Enforce amplitude and slew rate constraints."""
    I, Q = pulse[:, 0], pulse[:, 1]
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > _AMP
    if np.any(mask):
        scale = _AMP / amp[mask]
        I[mask] *= scale
        Q[mask] *= scale
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= SLEW
    # Include from zero into step 1 and last step to zero
    # We need to project onto the feasible set
    # Simple iterative projection
    for _ in range(50):
        # Check slew from 0 to step 0
        d0 = np.sqrt(I[0]**2 + Q[0]**2) / _DT
        if d0 > _SLEW:
            scale = _SLEW * _DT / np.sqrt(I[0]**2 + Q[0]**2)
            I[0] *= scale
            Q[0] *= scale
        
        # Check slew between steps
        for i in range(_N - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            d = np.sqrt(dI**2 + dQ**2) / _DT
            if d > _SLEW:
                scale = _SLEW * _DT / np.sqrt(dI**2 + dQ**2)
                I[i+1] = I[i] + dI * scale
                Q[i+1] = Q[i] + dQ * scale
        
        # Check slew from last step to 0
        dlast = np.sqrt(I[-1]**2 + Q[-1]**2) / _DT
        if dlast > _SLEW:
            scale = _SLEW * _DT / np.sqrt(I[-1]**2 + Q[-1]**2)
            I[-1] *= scale
            Q[-1] *= scale
        
        # Re-check amplitude
        amp = np.sqrt(I**2 + Q**2)
        mask = amp > _AMP
        if np.any(mask):
            scale = _AMP / amp[mask]
            I[mask] *= scale
            Q[mask] *= scale
    
    return np.column_stack([I, Q])


def _simulate_pulse_vectorized(pulse, detunings, scales):
    """
    Simulate pulse for all 9 (detuning, scale) cases in parallel.
    Returns fidelity array of shape (9, 5) for 5 input states.
    """
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Input states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # That's 6 states, but task says 5? Let me re-read.
    # "over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # That's |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2) = 6 states
    # But it says mean fidelity to X over these. Let me count: |0>, |1>, then two real superpositions, two complex = 6.
    
    # X gate: |0> -> |1>, |1> -> |0>
    # Fidelity = |<psi_final|X|psi_initial>|^2 for pure states
    
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (psi0 + psi1) / np.sqrt(2)
    psi_m = (psi0 - psi1) / np.sqrt(2)
    psi_ip = (psi0 + 1j*psi1) / np.sqrt(2)
    psi_im = (psi0 - 1j*psi1) / np.sqrt(2)
    
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    states = [psi0, psi1, psi_p, psi_m, psi_ip, psi_im]
    n_states = len(states)
    
    # For each state, evolve density matrix
    # rho = |psi><psi|
    # Evolve: rho_{k+1} = expm(L_k * dt) @ rho_k
    # where L_k is the Liouvillian superoperator
    
    # We'll simulate all 9 cases x 6 states in parallel
    # Shape: (9, 6, 9, 9) for density matrices
    
    n_cases = 9
    case_det = []
    case_scale = []
    for d in detunings:
        for s in scales:
            case_det.append(d)
            case_scale.append(s)
    case_det = np.array(case_det)
    case_scale = np.array(case_scale)
    
    # Initialize density matrices
    rhos = np.zeros((n_cases, n_states, 3, 3), dtype=complex)
    for si, psi in enumerate(states):
        rho = np.outer(psi, psi.conj())
        for ci in range(n_cases):
            rhos[ci, si] = rho
    
    # Simulate step by step
    for step in range(_N):
        I_val = I[step]
        Q_val = Q[step]
        
        # Build Hamiltonian for each case
        # H = det * n + 0.5*anh*n*(n-1) + 0.5*scale*(I*(a+a†) + i*Q*(a-a†))
        # The det and scale parts vary per case, I,Q are same for all cases at this step
        
        # Precompute the drive part (same for all cases at this step, scaled by scale)
        drive = I_val * _A_ADAG + 1j * Q_val * _A_MINUS_ADAG
        
        for ci in range(n_cases):
            d = case_det[ci]
            s = case_scale[ci]
            
            H = d * _NOP + 0.5 * _ANH * _NNM1 + 0.5 * s * drive
            
            # Liouvillian superoperator
            # -i(H⊗I - I⊗H†) + _LIND_CONST
            # H is Hermitian, so H† = H
            # H⊗I - I⊗H = commutator structure
            # Actually: -i[H,ρ] corresponds to superoperator -i(H⊗I - I⊗H†)
            # Since H is Hermitian: H⊗I - I⊗H
            
            # Use the fact that H⊗I - I⊗H can be built efficiently
            # For 3x3 matrices, just do it
            H_super = -1j * (np.kron(H, np.eye(3)) - np.kron(np.eye(3), H.conj().T))
            
            L = H_super + _LIND_CONST
            
            # Evolve all 6 states for this case
            # rho_new = expm(L*dt) @ rho
            # For efficiency, compute expm once per case per step
            U_step = expm(L * _DT)
            
            for si in range(n_states):
                rhos[ci, si] = U_step @ rhos[ci, si]
        
        if _time_remaining() < 30.0:
            break
    
    # Compute fidelities
    # Fidelity to X: |<psi_final| X |psi_initial>|^2
    # For pure states: F = |<psi_f| X |psi_i>|^2
    # But we have density matrices. For pure states, F = Tr(X† rho_f X rho_i) ... 
    # Actually simpler: F = |<psi_f| X |psi_i>|^2 where psi_f is the final state
    # From density matrix: <psi_f| X |psi_i> = Tr(psi_f† X psi_i psi_f†) ... 
    # Easier: reconstruct state from density matrix (pure state assumption)
    
    fidelities = np.zeros((n_cases, n_states))
    for ci in range(n_cases):
        for si in range(n_states):
            rho_f = rhos[ci, si]
            psi_i = states[si]
            # X |psi_i>
            psi_x = X_gate @ psi_i
            # <psi_f| psi_x> = Tr(rho_f† |psi_x><psi_x|) ... no
            # <psi_f|psi_x> where psi_f is from rho_f
            # For pure state rho_f = |phi><phi|, <phi|psi_x> = Tr(|phi><phi|psi_x><x|) where x is any... 
            # Simpler: <phi|psi_x> = Tr(rho_f @ |psi_x><psi_x|) ... no that gives <psi_x|rho_f|psi_x>
            # Actually: if rho_f = |phi><phi|, then <phi|psi_x> = Tr(|phi><phi|psi_x><psi_x|) only if... no.
            # <phi|psi_x> = sum_i phi_i^* psi_x_i
            # rho_f @ |psi_x><psi_x| has trace = sum_i (rho_f @ |psi_x><psi_x|)_{ii} = <psi_x|rho_f|psi_x>
            # That's not what we want.
            # We want <phi|psi_x> = Tr(|phi><phi|psi_x><x|) where |x> is arbitrary... 
            # Just extract the state vector.
            # For a pure state density matrix, the state vector can be extracted as:
            # rho_f = |phi><phi|, so phi_i = rho_f[i,j] / phi_j for any j where phi_j != 0
            # Or: phi = eigenvector with eigenvalue 1
            
            # Simple approach: phi_i = rho_f[i,0] / phi_0 ... circular
            # Use: phi = rho_f @ v / ||rho_f @ v|| for some v, but this is approximate
            # Better: SVD
            U, s, Vh = np.linalg.svd(rho_f)
            phi = U[:, 0]
            # Ensure correct phase
            # <phi|psi_x>
            amp = np.vdot(phi, psi_x)
            fidelities[ci, si] = np.abs(amp)**2
    
    return fidelities


def _compute_score(fidelities):
    """Compute mean + max infidelity over 9 cases."""
    infidelities = 1.0 - fidelities  # shape (9, 6)
    # Mean over states, then over cases? Or mean over all?
    # "infidelity = 1 - mean fidelity to X over |0>, |1>, ..."
    # So for each case, mean fidelity over the 6 states, then infidelity = 1 - that
    # Then score = mean of 9 infidelities + max of 9 infidelities
    
    mean_fid_per_case = np.mean(fidelities, axis=1)  # shape (9,)
    inf_per_case = 1.0 - mean_fid_per_case
    
    score = np.mean(inf_per_case) + np.max(inf_per_case)
    return score


def _soft_max_score(fidelities, temperature=0.01):
    """Soft-max surrogate for the max term."""
    inf_per_case = 1.0 - np.mean(fidelities, axis=1)
    # Soft max: temperature * log(sum(exp(inf/temperature)))
    # This approximates max when temperature -> 0
    max_term = temperature * np.log(np.sum(np.exp(inf_per_case / temperature)))
    mean_term = np.mean(inf_per_case)
    return mean_term + max_term


def _objective(params):
    """Objective function for optimization."""
    I = params[:_N]
    Q = params[_N:2*_N]
    pulse = np.column_stack([I, Q])
    pulse = _enforce_constraints(pulse)
    
    fidelities = _simulate_pulse_vectorized(pulse, _DET_GRID, _SCALE_GRID)
    score = _soft_max_score(fidelities)
    return score


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return _make_seed_pulse()
    
    # Start with seed
    best_pulse = _make_seed_pulse()
    best_score = _compute_score(_simulate_pulse_vectorized(best_pulse, _DET_GRID, _SCALE_GRID))
    
    # Optimization
    x0 = best_pulse.flatten()
    
    # Try L-BFGS-B
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_score:
            best_pulse = _enforce_constraints(result.x.reshape(_N, 2))
            best_score = _compute_score(_simulate_pulse_vectorized(best_pulse, _DET_GRID, _SCALE_GRID))
    except Exception:
        pass
    
    # Deadline check
    if _time_remaining() < 10.0:
        return best_pulse
    
    # Final polish with tighter tolerance
    try:
        x0 = best_pulse.flatten()
        result = minimize(_objective, x0, method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < best_score:
            best_pulse = _enforce_constraints(result.x.reshape(_N, 2))
            best_score = _compute_score(_simulate_pulse_vectorized(best_pulse, _DET_GRID, _SCALE_GRID))
    except Exception:
        pass
    
    return best_pulse

# EVOLVE-BLOCK-END
