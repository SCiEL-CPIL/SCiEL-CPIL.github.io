"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = 310.0
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

# Precompute Lindblad terms
_L1_L1ADAG = _L1 @ _L1ADAG
_L2_L2ADAG = _L2 @ _L2ADAG

def _build_liouvillian(I, Q, detuning, scale):
    """Build the Liouvillian matrix for given pulse step parameters."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    H = detuning * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - 3*np.eye(3)))
    
    # Drive term: 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    drive = 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
    H += drive
    
    # Liouvillian: -i*H⊗I + 0.5*(L†L⊗I - 0.5*{L†L, I⊗I}) for each L
    # Actually, for the master equation: dρ/dt = -i[H,ρ] + Σ( LρL† - 0.5{L†L, ρ} )
    # In Liouvillian form: ℒ = -i(H⊗I - I⊗Hᵀ) + Σ( L⊗L* - 0.5(L†L⊗I + I⊗L†Lᵀ) )
    
    # Let's use the vectorization convention where vec(ρ) stacks columns
    # d vec(ρ)/dt = [ -i(H⊗I - I⊗Hᵀ) + Σ( L⊗L* - 0.5(L†L⊗I + I⊗L†Lᵀ) ) ] vec(ρ)
    
    liouv = -1j * (np.kron(H, np.eye(3)) - np.kron(np.eye(3), H.T))
    
    for L, LdagL in [(_L1, _L1_L1ADAG), (_L2, _L2_L2ADAG)]:
        liouv += np.kron(L, L.conj()) - 0.5 * (np.kron(LdagL, np.eye(3)) + np.kron(np.eye(3), LdagL.T))
    
    return liouv

def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return final state vector for each initial state."""
    # pulse shape (80, 2): I, Q
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    
    # We'll simulate all 6 initial states simultaneously by building a 9x18 matrix (3 levels x 6 states)
    # Actually, let's just simulate one at a time for clarity, or batch them.
    
    # For efficiency, let's simulate all 6 initial states at once.
    # State is a 9x6 matrix where each column is vec(ρ) for one initial state.
    
    # Initial states
    psi0 = np.array([1.0, 0.0, 0.0])
    psi1 = np.array([0.0, 1.0, 0.0])
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j*psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j*psi1) / np.sqrt(2)
    
    # Build density matrices and vectorize
    rho0 = np.outer(psi0, psi0.conj())
    rho1 = np.outer(psi1, psi1.conj())
    rho_plus = np.outer(psi_plus, psi_plus.conj())
    rho_minus = np.outer(psi_minus, psi_minus.conj())
    rho_i_plus = np.outer(psi_i_plus, psi_i_plus.conj())
    rho_i_minus = np.outer(psi_i_minus, psi_i_minus.conj())
    
    # Vectorize (column-major)
    vec_rho0 = rho0.flatten(order='F')
    vec_rho1 = rho1.flatten(order='F')
    vec_rho_plus = rho_plus.flatten(order='F')
    vec_rho_minus = rho_minus.flatten(order='F')
    vec_rho_i_plus = rho_i_plus.flatten(order='F')
    vec_rho_i_minus = rho_i_minus.flatten(order='F')
    
    # Stack into 9x6 matrix
    state = np.column_stack([vec_rho0, vec_rho1, vec_rho_plus, vec_rho_minus, vec_rho_i_plus, vec_rho_i_minus])
    
    # Simulate
    for i in range(_N):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        
        liouv = _build_liouvillian(I_val, Q_val, detuning, scale)
        U = expm(liouv * _DT)
        state = U @ state
    
    return state

def _fidelity_with_X(state, initial_state_idx):
    """Compute fidelity between final state and X gate applied to initial state."""
    # X gate: |0> -> |1>, |1> -> |0>
    # Fidelity = |<psi_target|psi_final>|^2 for pure states
    # But we have density matrices, so F = Tr(ρ_target ρ_final)
    
    # Target states after X gate:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2)
    
    psi_target = None
    if initial_state_idx == 0:  # |0> -> |1>
        psi_target = np.array([0.0, 1.0, 0.0])
    elif initial_state_idx == 1:  # |1> -> |0>
        psi_target = np.array([1.0, 0.0, 0.0])
    elif initial_state_idx == 2:  # (|0>+|1>)/sqrt(2) -> same
        psi_target = (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2)
    elif initial_state_idx == 3:  # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2)
        psi_target = (np.array([0.0, 1.0, 0.0]) - np.array([1.0, 0.0, 0.0])) / np.sqrt(2)
    elif initial_state_idx == 4:  # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2)
        psi_target = (np.array([0.0, 1.0, 0.0]) + 1j*np.array([1.0, 0.0, 0.0])) / np.sqrt(2)
    elif initial_state_idx == 5:  # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2)
        psi_target = (np.array([0.0, 1.0, 0.0]) - 1j*np.array([1.0, 0.0, 0.0])) / np.sqrt(2)
    
    rho_target = np.outer(psi_target, psi_target.conj())
    vec_rho_target = rho_target.flatten(order='F')
    
    # Final state for this initial condition
    vec_rho_final = state[:, initial_state_idx]
    
    # Fidelity = Tr(ρ_target ρ_final) = vec(ρ_target)† · vec(ρ_final) (for pure states, this is |<ψ_t|ψ_f>|^2)
    # Actually, for general density matrices: F = Tr(ρ_t ρ_f)
    # In vectorized form: F = vec(ρ_t)† @ vec(ρ_f) if both are pure... 
    # More generally: Tr(ρ_t ρ_f) = sum_ij (ρ_t)_{ij} (ρ_f)_{ji}
    # In vectorized form: = vec(ρ_t)† @ vec(ρ_f) when using column-major vectorization
    # Wait, let me verify: vec(AB) = (B⊗A)vec(C)... 
    # Tr(ρ_t ρ_f) = sum_{ij} (ρ_t)_{ij} (ρ_f)_{ji}
    # If we stack column by column, vec(ρ_t) has elements ρ_t[0,0], ρ_t[1,0], ρ_t[2,0], ρ_t[0,1], ...
    # vec(ρ_f) has elements ρ_f[0,0], ρ_f[1,0], ρ_f[2,0], ρ_f[0,1], ...
    # The dot product vec(ρ_t)† @ vec(ρ_f) = sum_k conj(vec(ρ_t)_k) * vec(ρ_f)_k
    # This equals sum_{i,j} conj(ρ_t[i,j]) * ρ_f[i,j] = Tr(ρ_t† ρ_f)
    # Since ρ_t is Hermitian, ρ_t† = ρ_t, so this is Tr(ρ_t ρ_f). Good.
    
    fidelity = np.real(np.vdot(vec_rho_target, vec_rho_final))
    return np.clip(fidelity, 0.0, 1.0)

def _score_pulse(pulse):
    """Compute the score: mean + max infidelity over 9 detuning/scale cases."""
    infidelities = []
    
    for detuning in _DET_GRID:
        for scale in _SCALE_GRID:
            state = _simulate_pulse(pulse, detuning, scale)
            
            case_infidelities = []
            for idx in range(6):
                fid = _fidelity_with_X(state, idx)
                infidelities.append(1.0 - fid)
            
            mean_inf = np.mean(case_infidelities)
            max_inf = np.max(case_infidelities)
            infidelities.append(mean_inf + max_inf)
    
    return np.mean(infidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > _AMP + 1e-9):
        return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero to step 1
    if np.linalg.norm(pulse[0]) / _DT > _SLEW + 1e-9:
        return False
    # Between steps
    for i in range(_N - 1):
        diff = pulse[i+1] - pulse[i]
        if np.linalg.norm(diff) / _DT > _SLEW + 1e-9:
            return False
    # From last step to zero
    if np.linalg.norm(pulse[-1]) / _DT > _SLEW + 1e-9:
        return False
    
    return True

def _make_seed_pulse():
    """Create DRAG-style Gaussian seed pulse."""
    t = np.arange(_N) * _DT
    t_center = _T / 2
    
    # Gaussian with sigma = 40/6 ns
    sigma = _T / 6.0
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to 0 at edges: subtract edge value
    edge_val = np.exp(-0.5 * ((t[0] - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Scale to area pi
    area = np.trapz(gauss, dx=_DT)
    if abs(area) > 1e-12:
        gauss = gauss * (np.pi / area)
    
    # DRAG: add derivative term
    # d(gauss)/dt approximated by central differences
    dgauss = np.zeros_like(gauss)
    dgauss[1:-1] = (gauss[2:] - gauss[:-2]) / (2*_DT)
    dgauss[0] = (gauss[1] - gauss[0]) / _DT
    dgauss[-1] = (gauss[-1] - gauss[-2]) / _DT
    
    # DRAG coefficient: for transmon, beta ≈ -anh/(4*omega_drive)
    # But we don't have omega_drive explicitly. Let's use a small empirical value.
    # Actually, for the X-pi gate on a transmon, the DRAG term is proportional to the anharmonicity.
    # Let's use beta = -_ANH / (4 * 2*np.pi*5e9) but we don't know the frequency.
    # Let's just use a small value for the Q component.
    drag_coeff = 0.0  # Start with no DRAG, pure I pulse
    
    I = gauss
    Q = drag_coeff * dgauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce constraints
    pulse = _enforce_constraints(pulse)
    
    return pulse

def _enforce_constraints(pulse):
    """Project pulse onto feasible set (amplitude and slew constraints)."""
    # Simple projection: clip amplitudes
    for i in range(_N):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > _AMP:
            scale = _AMP / amp
            pulse[i, 0] *= scale
            pulse[i, 1] *= scale
    
    # Slew constraint: iterative projection
    # This is complex, so let's just ensure it's valid by reducing if needed
    for _ in range(10):
        valid = True
        # Check slew from zero to step 1
        if np.linalg.norm(pulse[0]) / _DT > _SLEW:
            scale = _SLEW * _DT / np.linalg.norm(pulse[0])
            pulse[0] *= scale
            valid = False
        
        # Check between steps
        for i in range(_N - 1):
            diff = pulse[i+1] - pulse[i]
            norm_diff = np.linalg.norm(diff)
            if norm_diff / _DT > _SLEW:
                # Adjust pulse[i+1] to satisfy constraint
                max_step = _SLEW * _DT
                if norm_diff > max_step:
                    pulse[i+1] = pulse[i] + diff * (max_step / norm_diff)
                valid = False
        
        # Check from last step to zero
        if np.linalg.norm(pulse[-1]) / _DT > _SLEW:
            scale = _SLEW * _DT / np.linalg.norm(pulse[-1])
            pulse[-1] *= scale
            valid = False
        
        if valid:
            break
    
    return pulse

def _optimize_pulse(pulse):
    """Optimize the pulse using gradient-free methods."""
    best_pulse = pulse.copy()
    best_score = _score_pulse(pulse)
    
    # Use Nelder-Mead or similar
    # Given time constraints, let's try a simple coordinate descent
    step_size = 0.01
    patience = 50
    no_improve = 0
    
    for iteration in range(200):
        if _time_remaining() < 60.0:
            break
        
        improved = False
        for i in range(_N):
            for j in range(2):
                # Try positive step
                trial = best_pulse.copy()
                trial[i, j] += step_size
                if _check_validity(trial):
                    score = _score_pulse(trial)
                    if score < best_score:
                        best_pulse = trial
                        best_score = score
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1
                
                # Try negative step
                trial = best_pulse.copy()
                trial[i, j] -= step_size
                if _check_validity(trial):
                    score = _score_pulse(trial)
                    if score < best_score:
                        best_pulse = trial
                        best_score = score
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1
        
        if not improved:
            step_size *= 0.7
            no_improve += 10
        
        if no_improve > patience:
            break
    
    return best_pulse

def pulse(target_gate, hardware_model):
    """Main entry point: return (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gate
        return np.zeros((_N, 2))
    
    # Create seed pulse
    pulse_seed = _make_seed_pulse()
    
    # Optimize
    pulse_opt = _optimize_pulse(pulse_seed)
    
    # Ensure validity
    if not _check_validity(pulse_opt):
        pulse_opt = _enforce_constraints(pulse_opt)
    
    return pulse_opt
# EVOLVE-BLOCK-END
