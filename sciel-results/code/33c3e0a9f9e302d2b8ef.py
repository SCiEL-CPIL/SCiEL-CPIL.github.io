"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, Bounds
import warnings

_START_TIME = time.time()
_DEADLINE = 310.0

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

# Precompute Liouvillian components (detuning=0, scale=1 baseline)
# H = detuning * n + 0.5 * anh * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
# L = H ⊗ I - I ⊗ H† + sum_k (L_k ⊗ L_k* - 0.5 {L_k†L_k, I} ⊗ I)

def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian matrix for given parameters."""
    # Hamiltonian
    n = _NOP
    n_n_minus_1 = n @ n - n  # n(n-1)
    
    H = (detuning * n 
         + 0.5 * _ANH * n_n_minus_1 
         + 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG)))
    
    # Liouvillian: L = H ⊗ I - I ⊗ H† + dissipative terms
    I3 = np.eye(3)
    L = np.kron(H, I3) - np.kron(I3, H.conj().T)
    
    # Dissipative terms
    L1dagL1 = _L1ADAG @ _L1
    L2dagL2 = _L2ADAG @ _L2
    L += np.kron(_L1, _L1.conj()) - 0.5 * np.kron(L1dagL1 + L1dagL1.conj().T, I3)
    L += np.kron(_L2, _L2.conj()) - 0.5 * np.kron(L2dagL2 + L2dagL2.conj().T, I3)
    
    return L

def _simulate_pulse(IQ, detuning, scale):
    """Simulate the pulse and return the final density matrix (9x9)."""
    # Initial state: |0><0|
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(_N):
        I_val = IQ[i, 0]
        Q_val = IQ[i, 1]
        L = _build_liouvillian(detuning, scale, I_val, Q_val)
        U = expm(L * _DT)
        rho = U @ rho
        # Renormalize trace to prevent drift
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho /= tr
    
    return rho

def _fidelity_to_X(rho, psi):
    """Compute fidelity F = <psi|rho|psi> for target X gate."""
    # X gate maps |0>->|1>, |1>->|0>
    # So X|psi> is the target state
    psi_vec = psi.flatten()
    # Apply X: swap |0> and |1> components, |2> stays
    psi_X = np.zeros_like(psi_vec)
    psi_X[1] = psi_vec[0]  # |0> -> |1>
    psi_X[0] = psi_vec[1]  # |1> -> |0>
    psi_X[2] = psi_vec[2]  # |2> -> |2>
    
    F = np.real(np.vdot(psi_X, rho @ psi_X))
    return F

def _compute_score(IQ):
    """Compute the full score (mean + max of 9-case infidelities)."""
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    infidelities = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            rho = _simulate_pulse(IQ, det, scale)
            case_inf = 0.0
            for psi in states:
                F = _fidelity_to_X(rho, psi)
                case_inf += (1.0 - F) / len(states)
            infidelities.append(case_inf)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _enforce_constraints(IQ):
    """Enforce amplitude and slew constraints by clipping."""
    IQ = IQ.copy()
    
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    for i in range(_N):
        amp = np.sqrt(IQ[i, 0]**2 + IQ[i, 1]**2)
        if amp > _AMP:
            scale = _AMP / amp
            IQ[i, 0] *= scale
            IQ[i, 1] *= scale
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    prev = np.array([0.0, 0.0])
    for i in range(_N):
        diff = IQ[i] - prev
        norm = np.linalg.norm(diff)
        max_step = _SLEW * _DT
        if norm > max_step:
            scale = max_step / norm
            IQ[i] = prev + scale * diff
        prev = IQ[i].copy()
    
    # Check last step to zero
    diff = np.array([0.0, 0.0]) - prev
    norm = np.linalg.norm(diff)
    max_step = _SLEW * _DT
    if norm > max_step:
        # This means the last step is too far from zero; we need to adjust
        # For now, just clip the last step
        scale = max_step / norm
        IQ[_N-1] = scale * prev
    
    return IQ

def _make_seed_pulse():
    """Create the DRAG-style Gaussian seed pulse."""
    t = np.arange(_N) * _DT
    sigma = 40.0 / 6.0
    
    # Gaussian with area pi
    # Integral of A * exp(-(t-tc)^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi)
    # We want area = pi, so A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    A = np.sqrt(np.pi / 2.0) / sigma
    
    # Offset to 0 at edges: the Gaussian naturally goes to 0 at edges for large enough sigma
    # But let's ensure it's zero at t=0 and t=T
    I = A * np.exp(-t**2 / (2 * sigma**2))
    # Offset: subtract value at t=0 to make it zero there
    I = I - I[0]
    # Actually, let's just use the Gaussian as is; it's small at edges
    
    IQ = np.column_stack([I, np.zeros(_N)])
    return _enforce_constraints(IQ)

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return _make_seed_pulse()
    
    best_IQ = _make_seed_pulse()
    best_score = _compute_score(best_IQ)
    
    # Try L-BFGS-B optimization with constraints
    if _time_remaining() > 60:
        try:
            # Flatten IQ
            x0 = best_IQ.flatten()
            
            # Bounds for each element
            bounds = [(-_AMP, _AMP)] * (2 * _N)
            
            def objective(x):
                IQ = x.reshape(_N, 2)
                IQ = _enforce_constraints(IQ)
                return _compute_score(IQ)
            
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_score:
                candidate = _enforce_constraints(result.x.reshape(_N, 2))
                cand_score = _compute_score(candidate)
                if cand_score < best_score:
                    best_IQ = candidate
                    best_score = cand_score
        
        except Exception as e:
            pass
    
    # Try Nelder-Mead with random perturbations
    if _time_remaining() > 30:
        try:
            for restart in range(5):
                if _time_remaining() < 30:
                    break
                
                # Random perturbation
                noise = np.random.randn(_N, 2) * 0.05
                x0 = best_IQ + noise
                x0 = _enforce_constraints(x0)
                x0_flat = x0.flatten()
                
                def objective(x):
                    IQ = x.reshape(_N, 2)
                    IQ = _enforce_constraints(IQ)
                    return _compute_score(IQ)
                
                result = minimize(objective, x0_flat, method='Nelder-Mead',
                                 options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if result.fun < best_score:
                    candidate = _enforce_constraints(result.x.reshape(_N, 2))
                    cand_score = _compute_score(candidate)
                    if cand_score < best_score:
                        best_IQ = candidate
                        best_score = cand_score
        
        except Exception as e:
            pass
    
    return best_IQ

# EVOLVE-BLOCK-END
