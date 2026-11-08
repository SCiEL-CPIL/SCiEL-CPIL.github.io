"""Robust control pulses for a three-level transmon."""
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 300.0
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
_I3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute static parts of Liouvillian for speed
_L1DAG_L1 = _L1ADAG @ _L1
_L2DAG_L2 = _L2ADAG @ _L2

# Dissipative part (constant)
_L_DISS = (np.kron(_L1, _L1.conj()) - 0.5*(np.kron(_I3, _L1DAG_L1) + np.kron(_L1DAG_L1.T, _I3))
          + np.kron(_L2, _L2.conj()) - 0.5*(np.kron(_I3, _L2DAG_L2) + np.kron(_L2DAG_L2.T, _I3)))

# Precompute static Hamiltonian Liouvillian for each detuning
_STATIC_LIOUV = {}
for det in _DET_GRID:
    H_static = det * _NOP + 0.5 * _ANH * _NOP @ (_NOP - 3*_I3)
    L_static = -1j * (np.kron(_I3, H_static) - np.kron(H_static.T, _I3))
    _STATIC_LIOUV[det] = L_static + _L_DISS

# X gate in the 3-level system
_X_gate = np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]], dtype=complex)

# Initial states for fidelity computation
_PSI0 = np.array([1.0, 0.0, 0.0], dtype=complex)
_PSI1 = np.array([0.0, 1.0, 0.0], dtype=complex)
_PSI_PLUS = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
_PSI_MINUS = np.array([1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2)
_PSI_IP = np.array([1.0, 1j, 0.0], dtype=complex) / np.sqrt(2)
_PSI_IM = np.array([1.0, -1j, 0.0], dtype=complex) / np.sqrt(2)
_STATES = [_PSI0, _PSI1, _PSI_PLUS, _PSI_MINUS, _PSI_IP, _PSI_IM]

def _liouvillian_step(I_val, Q_val, scale, det_idx):
    """Build the Liouvillian superoperator for a given step with drive."""
    # Drive part of Hamiltonian
    H_drive = 0.5 * scale * (I_val * (_A + _ADAG) + 1j * Q_val * (_A - _ADAG))
    L_drive = -1j * (np.kron(_I3, H_drive) - np.kron(H_drive.T, _I3))
    return _STATIC_LIOUV[det_idx] + L_drive

def _simulate_pulse(I, Q, det_idx, scale):
    """Simulate the pulse and return the final evolution superoperator (9x9)."""
    # Start with identity superoperator (identity on density matrix space)
    U = np.eye(9, dtype=complex)
    for i in range(_N):
        L = _liouvillian_step(I[i], Q[i], scale, det_idx)
        # expm(L * dt) is the evolution for one step
        U_step = expm(L * _DT)
        U = U_step @ U
    return U

def _compute_fidelity(U_super, psi):
    """Compute fidelity F = <psi|U(rho0)|psi_target> where rho0 = |psi><psi|."""
    # Vectorize rho0 = |psi><psi|
    rho0_vec = np.outer(psi, psi.conj()).flatten()
    # Apply superoperator
    rho_final_vec = U_super @ rho0_vec
    rho_final = rho_final_vec.reshape(3, 3)
    # Target state
    psi_target = _X_gate @ psi
    # Fidelity = <psi_target|rho_final|psi_target>
    fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
    return fid

def _score_pulse(I, Q):
    """Compute the score for a given pulse."""
    # Check constraints
    for i in range(_N):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > _AMP + 1e-9:
            return 1e10
        if i > 0:
            slew = np.sqrt((I[i]-I[i-1])**2 + (Q[i]-Q[i-1])**2) / _DT
            if slew > _SLEW + 1e-9:
                return 1e10
    # Check slew from zero to first and last to zero
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / _DT
    if slew_start > _SLEW + 1e-9:
        return 1e10
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / _DT
    if slew_end > _SLEW + 1e-9:
        return 1e10
    
    infidelities = []
    for di, det in enumerate(_DET_GRID):
        for scale in _SCALE_GRID:
            U_super = _simulate_pulse(I, Q, di, scale)
            fid_sum = 0.0
            for psi in _STATES:
                fid = _compute_fidelity(U_super, psi)
                fid_sum += fid
            mean_fid = fid_sum / len(_STATES)
            infidelities.append(1.0 - mean_fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _make_seed_pulse():
    """Create the DRAG-style Gaussian seed pulse."""
    t = np.linspace(0, _T, _N)
    sigma = 40.0 / 6.0
    # Gaussian envelope centered at T/2
    envelope = np.exp(-0.5 * ((t - _T/2) / sigma)**2)
    # Normalize area to pi
    area = np.sum(envelope) * _DT
    envelope = envelope * (np.pi / area)
    I = envelope
    Q = np.zeros(_N)
    # Ensure constraints
    max_amp = np.max(np.sqrt(I**2 + Q**2))
    if max_amp > _AMP:
        scale_factor = _AMP / max_amp
        I *= scale_factor
        Q *= scale_factor
    return I, Q

def _enforce_constraints(I, Q):
    """Project the pulse to satisfy amplitude and slew constraints."""
    I = np.array(I, dtype=float)
    Q = np.array(Q, dtype=float)
    # Amplitude constraint: sqrt(I^2 + Q^2) <= AMP
    for i in range(_N):
        amp = np.sqrt(I[i]**2 + Q[i]**2)
        if amp > _AMP:
            scale = _AMP / amp
            I[i] *= scale
            Q[i] *= scale
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= SLEW
    max_d = _SLEW * _DT
    for _ in range(15):  # Iterative projection
        # Forward pass
        prev_I, prev_Q = 0.0, 0.0
        for i in range(_N):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            norm = np.sqrt(dI**2 + dQ**2)
            if norm > max_d:
                scale = max_d / norm
                I[i] = prev_I + dI * scale
                Q[i] = prev_Q + dQ * scale
            prev_I, prev_Q = I[i], Q[i]
        
        # Backward pass
        next_I, next_Q = 0.0, 0.0
        for i in range(_N - 1, -1, -1):
            dI = I[i] - next_I
            dQ = Q[i] - next_Q
            norm = np.sqrt(dI**2 + dQ**2)
            if norm > max_d:
                scale = max_d / norm
                I[i] = next_I + dI * scale
                Q[i] = next_Q + dQ * scale
            next_I, next_Q = I[i], Q[i]
        
        # Re-apply amplitude constraint
        for i in range(_N):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > _AMP:
                scale = _AMP / amp
                I[i] *= scale
                Q[i] *= scale
    
    return I, Q

def _drag_pulse_from_params(params):
    """Generate a DRAG-style pulse from low-dimensional parameters.
    
    params = [amp, sigma, center, q_amp, q_sigma, q_center, i_offset, q_offset]
    """
    amp, sigma, center, q_amp, q_sigma, q_center, i_offset, q_offset = params
    t = np.linspace(0, _T, _N)
    # I component: Gaussian
    I = amp * np.exp(-0.5 * ((t - center) / sigma)**2) + i_offset
    # Q component: DRAG-like (derivative of Gaussian or separate Gaussian)
    Q = q_amp * np.exp(-0.5 * ((t - q_center) / q_sigma)**2) + q_offset
    return I, Q

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((_N, 2))
    
    # Start with seed pulse
    I, Q = _make_seed_pulse()
    I, Q = _enforce_constraints(I, Q)
    best_score = _score_pulse(I, Q)
    best_I, best_Q = I.copy(), Q.copy()
    
    # Phase 1: Low-dimensional DRAG parameter search with DE
    if _time_remaining() > 120:
        try:
            def drag_objective(params):
                I_opt, Q_opt = _drag_pulse_from_params(params)
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                return _score_pulse(I_opt, Q_opt)
            
            # Bounds for DRAG parameters
            # amp: amplitude of I Gaussian, sigma: width, center: time center
            # q_amp: amplitude of Q, q_sigma: width, q_center: time center
            # i_offset, q_offset: DC offsets
            bounds = [
                (0.0, _AMP),      # amp
                (2.0, 15.0),      # sigma (ns)
                (5.0, 35.0),      # center (ns)
                (-0.1, 0.1),      # q_amp
                (2.0, 15.0),      # q_sigma
                (5.0, 35.0),      # q_center
                (-0.05, 0.05),    # i_offset
                (-0.05, 0.05),    # q_offset
            ]
            
            result = differential_evolution(
                drag_objective, bounds,
                maxiter=20, popsize=8,
                seed=42, tol=1e-5,
                polish=False,
                updating='deferred'
            )
            
            if result.fun < best_score:
                I_opt, Q_opt = _drag_pulse_from_params(result.x)
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                best_score = result.fun
                best_I, best_Q = I_opt.copy(), Q_opt.copy()
        
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead in low-dim space
    if _time_remaining() > 60:
        try:
            # Initialize from best DRAG params
            # Estimate params from best_I, best_Q
            t = np.linspace(0, _T, _N)
            # Rough estimate
            i_peak_idx = np.argmax(best_I)
            q_peak_idx = np.argmax(np.abs(best_Q))
            init_params = [
                np.max(best_I),                          # amp
                max(2.0, min(15.0, _T / 4.0)),           # sigma
                t[i_peak_idx],                            # center
                np.max(np.abs(best_Q)) if np.max(np.abs(best_Q)) > 0.01 else 0.03,  # q_amp
                max(2.0, min(15.0, _T / 4.0)),           # q_sigma
                t[q_peak_idx] if np.max(np.abs(best_Q)) > 0.01 else t[i_peak_idx],  # q_center
                np.mean(best_I) - np.max(best_I),        # i_offset
                np.mean(best_Q),                          # q_offset
            ]
            # Clamp to bounds
            bounds = [
                (0.0, _AMP),
                (2.0, 15.0),
                (5.0, 35.0),
                (-0.1, 0.1),
                (2.0, 15.0),
                (5.0, 35.0),
                (-0.05, 0.05),
                (-0.05, 0.05),
            ]
            for i in range(len(init_params)):
                init_params[i] = max(bounds[i][0], min(bounds[i][1], init_params[i]))
            
            def drag_objective(params):
                I_opt, Q_opt = _drag_pulse_from_params(params)
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                return _score_pulse(I_opt, Q_opt)
            
            result = minimize(
                drag_objective, init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < best_score:
                I_opt, Q_opt = _drag_pulse_from_params(result.x)
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                best_score = result.fun
                best_I, best_Q = I_opt.copy(), Q_opt.copy()
        
        except Exception:
            pass
    
    # Phase 3: High-dimensional local refinement
    if _time_remaining() > 45:
        try:
            def full_objective(params):
                I_opt = params[:_N]
                Q_opt = params[_N:]
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                return _score_pulse(I_opt, Q_opt)
            
            result = minimize(
                full_objective, np.concatenate([best_I, best_Q]),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < best_score:
                I_opt = result.x[:_N]
                Q_opt = result.x[_N:]
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                best_score = result.fun
                best_I, best_Q = I_opt.copy(), Q_opt.copy()
        
        except Exception:
            pass
    
    # Final constraint enforcement
    best_I, best_Q = _enforce_constraints(best_I, best_Q)
    
    return np.column_stack([best_I, best_Q])
