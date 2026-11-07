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

_DEADLINE = 310.0  # Safety margin before 330s kill
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

    # H = delta * n + 0.5 * chi * n * (n - I) + 0.5 * S * (I * (a + a_dag) + i * Q * (a - a_dag))
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

    # Analytic DRAG coefficient calculation
    # For a Gaussian envelope I(t) = A*exp(-t^2/(2*sigma^2)), the first-order DRAG
    # correction Q(t) = alpha * dI/dt cancels |0>->|2> leakage.
    # The optimal alpha for a given anharmonicity chi is approximately:
    # alpha = -chi / (4 * sigma * omega_r) where omega_r is the Rabi frequency
    # For our case, we'll compute this more carefully.
    
    def compute_drag_alpha(A, sigma, t_center):
        """
        Compute optimal DRAG alpha for a Gaussian pulse.
        The leakage to |2> from |0> via |1> is proportional to:
        integral of I(t)^2 * exp(i*delta_12*t) dt
        where delta_12 is the detuning between |1> and |2>.
        The DRAG correction Q(t) = alpha * dI/dt helps cancel this.
        
        For a Gaussian centered at t_center with width sigma:
        dI/dt = -A*(t-t_center)/sigma^2 * exp(-0.5*((t-t_center)/sigma)^2)
        
        The optimal alpha is approximately:
        alpha ≈ -chi / (2 * omega_r) * (1/sigma)
        where omega_r ≈ A (for resonant drive)
        """
        # For our specific case with chi = -2*pi*0.25:
        # The standard DRAG formula gives alpha = -chi/(2*omega) * (1/sigma)
        # But we need to be careful about the exact definition.
        # Let's use the standard result: for H = (A/2)(a+a†) + (Q/2)(a-a†)i
        # The DRAG coefficient is alpha = -chi/(2*A) approximately for small chi
        # More precisely, for a Gaussian pulse, the optimal beta (where Q = beta * dI/dt)
        # is beta = -chi * sigma / (2 * A) * (some factor)
        
        # Actually, the standard DRAG result for a Gaussian pulse is:
        # Q(t) = alpha * dI/dt where alpha = -chi/(2*omega_r)
        # and omega_r ≈ A for resonant drive
        # So alpha ≈ -chi/(2*A)
        
        # But wait, this doesn't depend on sigma? Let me reconsider.
        # The leakage amplitude is proportional to the Fourier transform of I(t)^2
        # at frequency delta_12. For a Gaussian, this is also Gaussian.
        # The DRAG correction needs to match this frequency content.
        
        # A more careful analysis: for a pulse I(t) = A*exp(-t^2/(2*sigma^2)),
        # the effective Rabi frequency is A, and the DRAG coefficient is:
        # alpha = -chi * sigma^2 / (2 * A) * (1/sigma) = -chi * sigma / (2 * A)
        
        # Let's use: alpha = -chi * sigma / (2 * A)
        # This is the standard DRAG formula.
        
        if A < 1e-6:
            return 0.0
        return -ANHARMONICITY * sigma / (2.0 * A)

    def get_pulse(params):
        """
        params: [A, sigma, t_center, alpha, A2, sigma2, t_center2, alpha2]
        I(t) = A*exp(-0.5*((t-t_center)/sigma)^2) + A2*exp(-0.5*((t-t_center2)/sigma2)^2)
        Q(t) = alpha * dI/dt + alpha2 * dI2/dt
        """
        A, sigma, t_center, alpha, A2, sigma2, t_center2, alpha2 = params
        
        # Clamp parameters to valid ranges
        sigma = max(2.0, min(12.0, sigma))
        sigma2 = max(2.0, min(12.0, sigma2))
        t_center = max(8.0, min(32.0, t_center))
        t_center2 = max(8.0, min(32.0, t_center2))
        
        # Compute I(t)
        g1 = np.exp(-0.5 * ((_t - t_center) / sigma)**2)
        g2 = np.exp(-0.5 * ((_t - t_center2) / sigma2)**2)
        
        I = A * g1 + A2 * g2
        
        # Compute dI/dt
        dI1 = -A * g1 * (_t - t_center) / sigma**2
        dI2 = -A2 * g2 * (_t - t_center2) / sigma2**2
        dI = dI1 + dI2
        
        Q = alpha * dI1 + alpha2 * dI2
        
        # Check validity
        if not _valid(I, Q):
            return None
        return I, Q

    def objective(params):
        if elapsed() > _DEADLINE - 5:
            return 1000.0  # Large penalty if we're running out of time
        pulse_data = get_pulse(params)
        if pulse_data is None:
            return 10.0  # Penalty for invalid pulse
        I, Q = pulse_data
        return _score(I, Q)

    # Generate initial guesses using analytic DRAG
    starts = []
    
    # Single Gaussian pulses with analytic DRAG
    for A in [0.20, 0.25, 0.30]:
        for sigma in [5.0, 6.0, 7.0]:
            t_center = 20.0
            alpha = compute_drag_alpha(A, sigma, t_center)
            starts.append([A, sigma, t_center, alpha, 0.0, sigma, t_center, 0.0])
    
    # Two-Gaussian pulses with analytic DRAG for each
    for A1 in [0.15, 0.20]:
        for A2 in [0.10, 0.15]:
            for sigma1 in [4.0, 5.0]:
                for sigma2 in [4.0, 5.0]:
                    for tc1 in [17.0, 18.0]:
                        for tc2 in [22.0, 23.0]:
                            alpha1 = compute_drag_alpha(A1, sigma1, tc1)
                            alpha2 = compute_drag_alpha(A2, sigma2, tc2)
                            starts.append([A1, sigma1, tc1, alpha1, A2, sigma2, tc2, alpha2])
    
    # Also add some without DRAG for comparison
    for A in [0.25]:
        for sigma in [6.0]:
            starts.append([A, sigma, 20.0, 0.0, 0.0, sigma, 20.0, 0.0])
    
    bounds = [
        (-0.4, 0.4),   # A
        (3.0, 10.0),   # sigma
        (12.0, 28.0),  # t_center
        (-0.3, 0.3),   # alpha
        (-0.4, 0.4),   # A2
        (3.0, 10.0),   # sigma2
        (12.0, 28.0),  # t_center2
        (-0.3, 0.3),   # alpha2
    ]
    
    best_score = 100.0
    best_pulse = None
    best_params = None
    
    # Try local optimizations from multiple starting points
    n_starts = len(starts)
    for idx, start in enumerate(starts):
        if elapsed() > _DEADLINE - 15:
            break
            
        try:
            res = scipy.optimize.minimize(objective, start, method='Nelder-Mead',
                                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7,
                                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                pulse_data = get_pulse(res.x)
                if pulse_data is not None:
                    best_pulse = pulse_data
        except Exception as e:
            pass
    
    # If optimization didn't find a good pulse, try a simpler approach
    if best_pulse is None or best_score > 5.0:
        # Try a constant-amplitude pulse with DRAG-like Q
        for A in [0.20, 0.25, 0.30]:
            for alpha in [0.0, 0.05, 0.10]:
                I = np.ones(N_STEPS) * A
                # Ramp up and down for slew constraint
                ramp = 5
                for i in range(ramp):
                    I[i] *= (i + 1) / ramp
                    I[-1-i] *= (i + 1) / ramp
                
                # Q from derivative (approximately zero for constant, but add edge effects)
                Q = np.zeros(N_STEPS)
                dI = np.diff(I)
                Q[:-1] = alpha * dI / DT
                Q[-1] = 0
                
                if _valid(I, Q):
                    sc = _score(I, Q)
                    if sc < best_score:
                        best_score = sc
                        best_pulse = (I, Q)
                        if elapsed() > _DEADLINE - 10:
                            break
            if elapsed() > _DEADLINE - 10:
                break

    # Fallback: simple constant pulse
    if best_pulse is None:
        I = np.ones(N_STEPS) * (np.pi / N_STEPS)
        Q = np.zeros(N_STEPS)
        if not _valid(I, Q):
            I = np.ones(N_STEPS) * 0.03
        best_pulse = (I, Q)

    I, Q = best_pulse
    return np.column_stack([I, Q])
