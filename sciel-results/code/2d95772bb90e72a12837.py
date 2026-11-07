"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_START = time.time()
_DEADLINE = 305.0  # safety margin before 330s kill

# ---- fixed hardware constants ----
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT  # 40 ns
ANH = -2.0 * np.pi * 0.25
AMP_MAX = 0.4
SLEW_MAX = 0.12
DET_MAX = 2.0 * np.pi * 0.002
T1 = 25000.0
TPHI = 40000.0

# 3x3 operators
_A = np.array([[0, 1, 0], [0, 0, np.sqrt(2.0)], [0, 0, 0]], dtype=complex)
_ADAG = _A.conj().T
_NOP = _ADAG @ _A  # n = a^dag a

# Lindblad operators
_L1 = _A / np.sqrt(T1)
_L2 = _NOP * np.sqrt(2.0 / TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute Lindblad terms: L rho L^dag - 0.5{L^dag L, rho}
# In Liouville space (vectorized), these are fixed matrices
def _vec(rho):
    return rho.flatten()

def _unvec(v):
    return v.reshape(3, 3)

# Build fixed Liouvillian components (vectorized)
# L rho L^dag -> (L^dag^T \otimes L) vec(rho)
_L1_TERM = np.kron(_L1ADAG.T, _L1)  # L1 rho L1^dag
_L2_TERM = np.kron(_L2ADAG.T, _L2)  # L2 rho L2^dag
# -0.5{L^dag L, rho} -> -0.5[(L^dag L)^T \otimes I + I \otimes (L^dag L)] vec(rho)
_L1DL = _L1ADAG @ _L1
_L2DL = _L2ADAG @ _L2
_ID3 = np.eye(3, dtype=complex)
_L1_ANTI = -0.5 * (np.kron(_L1DL.T, _ID3) + np.kron(_ID3, _L1DL))
_L2_ANTI = -0.5 * (np.kron(_L2DL.T, _ID3) + np.kron(_ID3, _L2DL))

_LIND_FIXED = _L1_TERM + _L2_TERM + _L1_ANTI + _L2_ANTI

# Drive terms: I*(a + a^dag) + i*Q*(a - a^dag)
# H_drive = 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
# In Liouville: -i*[H, rho] = -i*(H rho - rho H) -> -i*(H \otimes I^T - I \otimes H^T) vec(rho)
# Actually: d(rho)/dt = -i[H,rho] + Lindblad
# -i[H,rho] vectorized: -i * (H \otimes I - I \otimes H^T) vec(rho) ... let me be careful
# [H,rho] = H rho - rho H
# vec(H rho) = (I \otimes H) vec(rho) ... no
# vec(H rho) = (rho^T \otimes H) ... 
# Standard: vec(AXB) = (B^T \otimes A) vec(X)
# vec(H rho) = (I^T \otimes H) vec(rho) = (I \otimes H) vec(rho)
# vec(rho H) = (H^T \otimes I) vec(rho)
# So [H,rho] -> (I \otimes H - H^T \otimes I) vec(rho)
# -i[H,rho] -> -i*(I \otimes H - H^T \otimes I) vec(rho)

# Precompute the kron structures for drive
_IK = np.eye(9, dtype=complex)
# For H = 0.5*scale*(I_q*(a+a^dag) + i*Q_q*(a-a^dag))
# Let X_op = a + a^dag, Y_op = i*(a - a^dag) = i*a - i*a^dag
_XOP = _A + _ADAG
_YOP = 1j * (_A - _ADAG)

# -i * (I \otimes H - H^T \otimes I)
# H = 0.5*s*(Iq*XOP + Qq*YOP)
# I \otimes H = 0.5*s*(Iq*(I \otimes XOP) + Qq*(I \otimes YOP))
# H^T \otimes I = 0.5*s*(Iq*(XOP^T \otimes I) + Qq*(YOP^T \otimes I))
# -i*(I \otimes H - H^T \otimes I) = -0.5*i*s*(Iq*((I \otimes XOP) - (XOP^T \otimes I)) + Qq*((I \otimes YOP) - (YOP^T \otimes I)))

_KX = np.kron(_ID3, _XOP) - np.kron(_XOP.T, _ID3)
_KY = np.kron(_ID3, _YOP) - np.kron(_YOP.T, _ID3)

# Detuning: det*n + 0.5*anh*n*(n-1)
_N2 = _NOP @ _NOP
_NNM1 = _NOP @ (_NOP - _ID3)
# H_det = det*n + 0.5*anh*n*(n-1)
# -i*(I \otimes H_det - H_det^T \otimes I)
_KN = np.kron(_ID3, _NOP) - np.kron(_NOP.T, _ID3)
_KN2 = np.kron(_ID3, _NNM1) - np.kron(_NNM1.T, _ID3)
_KDET = _KN  # coefficient is det
_KANH = 0.5 * ANH * _KN2  # constant part of anharmonicity

# So full Liouvillian L = KDET*det + KANH + drive_part + LIND_FIXED
# drive_part = -0.5j * scale * (Iq*KX + Qq*KY)

def _make_seed_pulse():
    """DRAG-style Gaussian pulse with sigma=40/6 ns, area=pi."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    # Gaussian centered at T/2
    center = T_TOTAL / 2.0
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    # Offset to 0 at edges
    g[0] = 0.0
    g[-1] = 0.0
    # Scale area to pi
    area = np.trapz(g, t)
    if area > 0:
        g *= np.pi / area
    # I = g, Q = 0 (initial seed)
    pulse = np.zeros((N_STEPS, 2))
    pulse[:, 0] = g
    return pulse

def _enforce_constraints(pulse):
    """Enforce amplitude and slew constraints."""
    p = pulse.copy()
    # Amplitude constraint: sqrt(I^2+Q^2) <= 0.4
    for i in range(N_STEPS):
        amp = np.sqrt(p[i, 0]**2 + p[i, 1]**2)
        if amp > AMP_MAX:
            scale = AMP_MAX / amp
            p[i, 0] *= scale
            p[i, 1] *= scale
    # Slew constraint: ||p[i+1]-p[i]||/dt <= 0.12, including from zero
    # From zero into step 1
    d = np.sqrt(p[0, 0]**2 + p[0, 1]**2)
    max_slew = SLEW_MAX * DT
    if d > max_slew:
        scale = max_slew / d
        p[0, 0] *= scale
        p[0, 1] *= scale
    # Between steps
    for i in range(N_STEPS - 1):
        diff = p[i+1] - p[i]
        d = np.sqrt(diff[0]**2 + diff[1]**2)
        if d > max_slew:
            scale = max_slew / d
            p[i+1] = p[i] + scale * diff
    # From last step to zero
    d = np.sqrt(p[-1, 0]**2 + p[-1, 1]**2)
    if d > max_slew:
        scale = max_slew / d
        p[-1, 0] *= scale
        p[-1, 1] *= scale
    return p

def _liouvillian(det, scale, Iq, Qq):
    """Build full Liouvillian matrix for given parameters."""
    L = _KDET * det + _KANH + _LIND_FIXED
    L += -0.5j * scale * (Iq * _KX + Qq * _KY)
    return L

def _simulate_pulse(pulse, det, scale):
    """Simulate pulse and return final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    v = _vec(rho)
    for i in range(N_STEPS):
        Iq = pulse[i, 0]
        Qq = pulse[i, 1]
        L = _liouvillian(det, scale, Iq, Qq)
        # expm(L * dt)
        v = expm(L * DT) @ v
    return _unvec(v)

def _fidelity_with_X(rho):
    """Compute fidelity to X gate: F = <psi| X^dag rho X |psi> averaged over inputs."""
    # X gate in 3-level: |0><1| + |1><0|
    X = np.zeros((3, 3), dtype=complex)
    X[0, 1] = 1.0
    X[1, 0] = 1.0
    # For input |psi>, output should be X|psi>
    # Fidelity = <X psi| rho |X psi>
    # Average over 6 states
    states = []
    states.append(np.array([1, 0, 0], dtype=complex))  # |0>
    states.append(np.array([0, 1, 0], dtype=complex))  # |1>
    states.append(np.array([1, 1, 0], dtype=complex) / np.sqrt(2))  # (|0>+|1>)/sqrt(2)
    states.append(np.array([1, -1, 0], dtype=complex) / np.sqrt(2))  # (|0>-|1>)/sqrt(2)
    states.append(np.array([1, 1j, 0], dtype=complex) / np.sqrt(2))  # (|0>+i|1>)/sqrt(2)
    states.append(np.array([1, -1j, 0], dtype=complex) / np.sqrt(2))  # (|0>-i|1>)/sqrt(2)
    
    fid_sum = 0.0
    for psi in states:
        psi_x = X @ psi
        fid = np.real(np.vdot(psi_x, rho @ psi_x))
        fid_sum += fid
    return fid_sum / len(states)

def _score_pulse(pulse):
    """Compute score: mean + max infidelity over 9 cases."""
    det_grid = [-DET_MAX, 0.0, DET_MAX]
    scale_grid = [0.97, 1.0, 1.03]
    infs = []
    for det in det_grid:
        for sc in scale_grid:
            rho = _simulate_pulse(pulse, det, sc)
            fid = _fidelity_with_X(rho)
            inf = 1.0 - fid
            infs.append(inf)
    return np.mean(infs) + np.max(infs)

def _optimize(pulse, max_iters=200, step_size=0.01):
    """Simple gradient-free optimization using finite differences."""
    best_pulse = pulse.copy()
    best_score = _score_pulse(best_pulse)
    
    for it in range(max_iters):
        if time.time() - _START > _DEADLINE:
            break
        improved = False
        for dim in range(pulse.shape[0] * 2):
            i = dim // 2
            j = dim % 2
            # Try positive and negative perturbation
            for direction in [1, -1]:
                p_trial = best_pulse.copy()
                p_trial[i, j] += direction * step_size
                p_trial = _enforce_constraints(p_trial)
                s_trial = _score_pulse(p_trial)
                if s_trial < best_score:
                    best_pulse = p_trial
                    best_score = s_trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step_size *= 0.9
            if step_size < 1e-5:
                break
    return best_pulse, best_score

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        # Return seed for other gates
        return _make_seed_pulse()
    
    # Start with DRAG seed
    p0 = _make_seed_pulse()
    p0 = _enforce_constraints(p0)
    
    # Optimize
    p_best, s_best = _optimize(p0, max_iters=50, step_size=0.02)
    
    return p_best
# EVOLVE-BLOCK-END
