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

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]

    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)

    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# ---- Fast simulation using matrix exponential of Liouvillian ----
def _simulate_pulse(I, Q, detuning, scale):
    """
    Simulate the pulse and return the 9x9 superoperator S.
    """
    S = np.eye(9, dtype=complex)
    for i in range(N_STEPS):
        L = _base_coh + _base_inc
        L += detuning * _det_coh
        L += scale * (I[i] * _coh_x + Q[i] * _coh_y)
        S = expm(L * DT) @ S
    return S

def _calc_infidelities(I, Q):
    """
    Calculate infidelities for all 9 cases.
    """
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
            S = _simulate_pulse(I, Q, di_val, sc_val)
            
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return infidelities

def _score(I, Q):
    """
    Calculate the actual score: mean + max of infidelities.
    """
    infids = _calc_infidelities(I, Q)
    return np.mean(infids) + np.max(infids)

def _penalty(I, Q):
    """
    Calculate penalty for constraint violations.
    """
    pen = 0.0
    amp = np.sqrt(I**2 + Q**2)
    viol_amp = np.maximum(0, amp - AMPLITUDE_BOUND)
    pen += np.sum(viol_amp**2)
    
    dist0 = np.hypot(I[0], Q[0])
    if dist0 / DT > SLEW_BOUND:
        pen += ((dist0 / DT) - SLEW_BOUND)**2
        
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        dist = np.hypot(dI, dQ)
        rate = dist / DT
        if rate > SLEW_BOUND:
            pen += (rate - SLEW_BOUND)**2
            
    distN = np.hypot(I[-1], Q[-1])
    if distN / DT > SLEW_BOUND:
        pen += ((distN / DT) - SLEW_BOUND)**2
        
    return pen

def _is_valid(I, Q):
    """
    Strictly check validity.
    """
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    return True

def _enforce_constraints(I, Q):
    """
    Iterative projection for amplitude and slew rate.
    """
    I = I.copy()
    Q = Q.copy()
    
    for _ in range(20):
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / np.max(amp)
            I *= scale
            Q *= scale
            
        dist0 = np.hypot(I[0], Q[0])
        if dist0 / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / dist0
            I[0] *= scale
            Q[0] *= scale
            
        for i in range(N_STEPS - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = np.hypot(dI, dQ)
            if dist / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / dist
                I[i+1] = I[i] + dI * scale
                Q[i+1] = Q[i] + dQ * scale
                
        distN = np.hypot(I[-1], Q[-1])
        if distN / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / distN
            I[-1] *= scale
            Q[-1] *= scale

    return I, Q

def _generate_seed_pulse():
    """
    Generate a DRAG-style Gaussian seed.
    """
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    sigma = TOTAL_TIME / 6.0
    
    I = np.exp(-0.5 * ((t - center) / sigma)**2)
    area = np.sum(I) * DT
    I *= (np.pi / area)
    
    dI_dt = np.gradient(I, DT)
    Q = -0.1 * dI_dt
    
    return I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    best_score = float('inf')
    best_I = np.zeros(N_STEPS)
    best_Q = np.zeros(N_STEPS)
    
    I_seed, Q_seed = _generate_seed_pulse()
    
    # Try multiple optimization strategies
    strategies = [
        ('L-BFGS-B', {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}),
        ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}),
        ('Powell', {'maxiter': 200, 'ftol': 1e-10}),
    ]
    
    for method, options in strategies:
        if elapsed() > _DEADLINE - 60:
            break
            
        I_curr = I_seed.copy()
        Q_curr = Q_seed.copy()
        
        def objective(params):
            I_p = params[:N_STEPS]
            Q_p = params[N_STEPS:]
            
            # Enforce constraints softly in the objective
            I_c, Q_c = _enforce_constraints(I_p, Q_p)
            
            sc = _score(I_c, Q_c)
            pen = _penalty(I_p, Q_p)
            return sc + 5.0 * pen
        
        params0 = np.concatenate([I_seed, Q_seed])
        
        try:
            res = scipy.optimize.minimize(
                objective, 
                params0, 
                method=method,
                options=options
            )
            
            if elapsed() > _DEADLINE - 30:
                break
                
            I_opt = res.x[:N_STEPS]
            Q_opt = res.x[N_STEPS:]
            
            # Enforce constraints
            I_final, Q_final = _enforce_constraints(I_opt, Q_opt)
            
            if _is_valid(I_final, Q_final):
                sc = _score(I_final, Q_final)
                if sc < best_score:
                    best_score = sc
                    best_I = I_final.copy()
                    best_Q = Q_final.copy()
                    
                    # Update seed for next iteration
                    I_seed = I_final.copy()
                    Q_seed = Q_final.copy()
                    
        except Exception:
            pass

    # Final local refinement
    if elapsed() < _DEADLINE - 10 and best_score < float('inf'):
        def objective_final(params):
            I_p = params[:N_STEPS]
            Q_p = params[N_STEPS:]
            I_c, Q_c = _enforce_constraints(I_p, Q_p)
            sc = _score(I_c, Q_c)
            pen = _penalty(I_p, Q_p)
            return sc + 5.0 * pen
        
        params0 = np.concatenate([best_I, best_Q])
        
        try:
            res = scipy.optimize.minimize(
                objective_final, 
                params0, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            I_opt = res.x[:N_STEPS]
            Q_opt = res.x[N_STEPS:]
            I_final, Q_final = _enforce_constraints(I_opt, Q_opt)
            
            if _is_valid(I_final, Q_final):
                sc = _score(I_final, Q_final)
                if sc < best_score:
                    best_score = sc
                    best_I = I_final.copy()
                    best_Q = Q_final.copy()
                    
        except Exception:
            pass

    # Fallback if no valid pulse found
    if best_score == float('inf'):
        I_seed, Q_seed = _generate_seed_pulse()
        I_seed, Q_seed = _enforce_constraints(I_seed, Q_seed)
        best_I = I_seed
        best_Q = Q_seed

    return np.column_stack([best_I, best_Q])

# Helper to ensure we don't crash on import
if __name__ == "__main__":
    pass
