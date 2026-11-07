import numpy as np
import time
from scipy.linalg import expm

_START_TIME = time.time()
_DEADLINE = 320.0 # Safety margin for 330s limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Hardware constants
_N = 80
_DT = 0.5
_T = _N * _DT
_ANH = -2.0 * np.pi * 0.25
_AMP_BOUND = 0.4
_SLEW_BOUND = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# Operators
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A
_NN1 = _NOP @ (_NOP - np.eye(3))

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)

# Precompute constant parts of Liouvillian
def _compute_dissipative_liouvillian():
    def diss_term(L):
        L_dag = L.conj().T
        # vec(L rho L^dag) = (L \otimes L^*) vec(rho)
        term1 = np.kron(L, L.conj())
        LL_dag = L_dag @ L
        # vec(0.5 {L^dag L, rho}) = 0.5 (I \otimes LL_dag + LL_dag^T \otimes I) vec(rho)
        term2 = 0.5 * (np.kron(np.eye(3), LL_dag) + np.kron(LL_dag.T, np.eye(3)))
        return term1 - term2
    
    return diss_term(_L1) + diss_term(_L2)

_L_DISS = _compute_dissipative_liouvillian()

def _liouvillian_matrix(delta, scale, I, Q):
    drive = 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
    H = delta * _NOP + 0.5 * _ANH * _NN1 + drive
    
    L_hamil = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    return L_hamil + _L_DISS

def _simulate_pulse(pulse, delta, scale):
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    U = np.eye(9)
    
    for i in range(_N):
        L_mat = _liouvillian_matrix(delta, scale, I[i], Q[i])
        U_step = expm(L_mat * _DT)
        U = U_step @ U
        
    return U

def _fidelity(U, psi_target, psi_init):
    rho_init = np.outer(psi_init, psi_init.conj())
    rho_init_vec = rho_init.flatten()
    
    rho_final_vec = U @ rho_init_vec
    rho_final = rho_final_vec.reshape((3, 3))
    
    fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
    return fid

def _infidelity_for_case(pulse, delta, scale):
    U = _simulate_pulse(pulse, delta, scale)
    
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    
    psi_p = (psi_0 + psi_1) / np.sqrt(2)
    psi_m = (psi_0 - psi_1) / np.sqrt(2)
    psi_p_i = (psi_0 + 1j*psi_1) / np.sqrt(2)
    psi_m_i = (psi_0 - 1j*psi_1) / np.sqrt(2)
    
    targets = {
        '0': psi_1,
        '1': psi_0,
        '+': psi_p,
        '-': psi_m,
        '+i': psi_m_i,
        '-i': psi_p_i
    }
    
    inits = {
        '0': psi_0,
        '1': psi_1,
        '+': psi_p,
        '-': psi_m,
        '+i': psi_p_i,
        '-i': psi_m_i
    }
    
    fids = []
    for key in inits:
        fid = _fidelity(U, targets[key], inits[key])
        fids.append(fid)
        
    mean_fid = np.mean(fids)
    infid = 1.0 - mean_fid
    return infid

def _score(pulse):
    infids = []
    for delta in _DET_GRID:
        for scale in _SCALE_GRID:
            infid = _infidelity_for_case(pulse, delta, scale)
            infids.append(infid)
            
    infids = np.array(infids)
    return np.mean(infids) + np.max(infids)

def _make_pulse(k_drag, sigma=40.0/6.0):
    t = np.arange(_N) * _DT + _DT/2.0
    t_center = _T / 2.0
    
    A = np.exp(-(t - t_center)**2 / (2.0 * sigma**2))
    A = A * (np.pi / (np.sum(A) * _DT))
    
    dA = np.gradient(A, _DT)
    
    I = A
    Q = k_drag * dA
    
    pulse = np.column_stack((I, Q))
    
    amps = np.sqrt(I**2 + Q**2)
    if np.max(amps) > _AMP_BOUND:
        scale_factor = _AMP_BOUND / np.max(amps)
        pulse *= scale_factor
        
    diffs = np.diff(pulse, axis=0)
    norms = np.sqrt(np.sum(diffs**2, axis=1))
    max_slew = np.max(norms) / _DT
    if max_slew > _SLEW_BOUND:
        scale_factor = _SLEW_BOUND / max_slew
        pulse *= scale_factor
        
    return pulse

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
        
    # Start with a reasonable DRAG parameter
    best_pulse = _make_pulse(15.0)
    best_score = _score(best_pulse)
    
    if _time_remaining() < 10.0:
        return best_pulse

    # Simple local search around a few k_drag values
    k_candidates = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    
    for k in k_candidates:
        if _time_remaining() < 5.0:
            break
        p = _make_pulse(k)
        s = _score(p)
        if s < best_score:
            best_score = s
            best_pulse = p
            
    return best_pulse
