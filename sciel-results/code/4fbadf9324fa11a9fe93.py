import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize

# ---- Constants from the task contract ----
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT
N_CASES = 9

_DEADLINE = 320.0  # Safety margin before 330s kill
_START_TIME = time.time()

# Time grid - midpoint sampling
_t = (np.arange(N_STEPS) + 0.5) * DT

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # Coherent part of Liouvillian
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    # H = delta * n + 0.5 * chi * n * (n - I) + 0.5 * S * (I * (a + a_dagger) + i * Q * (a - a_dagger))
    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    
    H_I = 0.5 * (a + a_dag)
    coh_I = coh(H_I)
    
    H_Q = 0.5j * (a - a_dag)
    coh_Q = coh(H_Q)
    
    H_det = n
    coh_det = coh(H_det)
    
    # Incoherent part
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
        
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # Test states
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
            
    # X-pi target: Swap |0> and |1>
    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)
        
    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_I, coh_Q, coh_det, base_inc, test, target

(_base_coh, _coh_I, _coh_Q, _coh_det, _base_inc,
 _test_states, _target_states) = _build()

def _valid(I, Q):
    """Check if pulse satisfies amplitude and slew constraints."""
    amp = np.sqrt(I * I + Q * Q)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-9:
        return False
    # Slew from zero to first step
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    # Slew between consecutive steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if np.max(slew) > SLEW_BOUND + 1e-9:
        return False
    # Slew from last step to zero
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    return True

def _score(I, Q):
    """Calculate robustness score."""
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _coh_det
                L += sc_val * (I[i] * _coh_I + Q[i] * _coh_Q)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # Basis: 4 Gaussians for I, 4 Gaussians for Q
    # Centers fixed at symmetric positions
    centers = np.array([10.0, 15.0, 25.0, 30.0])
    
    def get_pulse(params):
        """
        params: [AI1, AI2, AI3, AI4, SI1, SI2, SI3, SI4, AQ1, AQ2, AQ3, AQ4, SQ1, SQ2, SQ3, SQ4]
        """
        if len(params) != 16:
            return None
            
        AI = params[0:4]
        SI = params[4:8]
        AQ = params[8:12]
        SQ = params[12:16]
        
        # Ensure sigmas are positive
        SI = np.maximum(SI, 1.0)
        SQ = np.maximum(SQ, 1.0)
        
        I = np.zeros(N_STEPS)
        Q = np.zeros(N_STEPS)
        
        for k in range(4):
            g = np.exp(-0.5 * ((_t - centers[k]) / SI[k])**2)
            I += AI[k] * g
            g_q = np.exp(-0.5 * ((_t - centers[k]) / SQ[k])**2)
            Q += AQ[k] * g_q
            
        if not _valid(I, Q):
            return None
        return I, Q

    def objective(params):
        if elapsed() > _DEADLINE - 5:
            return 1000.0
        pulse_data = get_pulse(params)
        if pulse_data is None:
            return 10.0
        I, Q = pulse_data
        return _score(I, Q)

    # Initial guess for DE
    # Roughly: I is a single Gaussian centered at 20, Q is 0
    # AI: [0.2, 0.2, 0.2, 0.2], SI: [5, 5, 5, 5]
    # AQ: [0, 0, 0, 0], SQ: [5, 5, 5, 5]
    
    bounds = [
        (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), # AI
        (2.0, 10.0), (2.0, 10.0), (2.0, 10.0), (2.0, 10.0), # SI
        (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), (-0.4, 0.4), # AQ
        (2.0, 10.0), (2.0, 10.0), (2.0, 10.0), (2.0, 10.0)  # SQ
    ]
    
    best_score = 100.0
    best_pulse = None
    best_params = None

    # Phase 1: Differential Evolution
    if elapsed() < _DEADLINE - 100:
        try:
            res_de = scipy.optimize.differential_evolution(
                objective, bounds, maxiter=50, popsize=15, tol=0.01, 
                seed=42, polish=False, init='latinhypercube'
            )
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_params = res_de.x
                pulse_data = get_pulse(best_params)
                if pulse_data is not None:
                    best_pulse = pulse_data
        except Exception as e:
            pass

    # Phase 2: Local Refinement with Nelder-Mead
    if best_params is not None and elapsed() < _DEADLINE - 20:
        try:
            res_nm = scipy.optimize.minimize(objective, best_params, method='Nelder-Mead',
                                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x
                pulse_data = get_pulse(best_params)
                if pulse_data is not None:
                    best_pulse = pulse_data
        except Exception as e:
            pass

    # Fallback if DE failed or didn't find valid
    if best_pulse is None:
        # Simple DRAG Gaussian
        I = np.exp(-0.5 * ((_t - 20.0) / 6.0)**2)
        I *= 0.25
        dI = np.gradient(I, DT)
        Q = 0.05 * dI
        if _valid(I, Q):
            best_pulse = (I, Q)
        else:
            I = np.ones(N_STEPS) * 0.05
            Q = np.zeros(N_STEPS)
            best_pulse = (I, Q)

    I, Q = best_pulse
    return np.column_stack([I, Q])
