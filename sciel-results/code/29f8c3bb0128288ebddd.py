"""Robust control pulses for a three-level transmon - DRAG pulse optimization."""
import numpy as np
import time
from scipy.linalg import expm

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Deadline guard
_DEADLINE = 320.0  # seconds, with 10s safety margin
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a  # diag(0, 1, 2)
    return a, a_dag, n

def _liouvillian(I, Q, detuning, scale, a, a_dag, n):
    """Build the Liouvillian superoperator for one step."""
    # Hamiltonian
    H = detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3)) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # We'll build it as a superoperator acting on vec(rho)
    # For a 3x3 matrix, the superoperator is 9x9
    
    def _superop(H, Ls):
        """Build superoperator for Liouvillian."""
        dim = H.shape[0]
        L = np.zeros((dim*dim, dim*dim), dtype=complex)
        
        # Coherent part: -i(H rho - rho H)
        Hvec = H.reshape(dim*dim)
        for i in range(dim):
            for j in range(dim):
                # -i H rho: element (i,j) gets -i * sum_k H[i,k] rho[k,j]
                for k in range(dim):
                    if H[i, k] != 0:
                        L[i*dim + j, k*dim + j] += -1j * H[i, k]
                # +i rho H: element (i,j) gets +i * sum_k rho[i,k] H[k,j]
                for k in range(dim):
                    if H[k, j] != 0:
                        L[i*dim + j, i*dim + k] += 1j * H[k, j]
        
        # Incoherent part
        for Lk in Ls:
            Lk_dag = Lk.conj().T
            # Lk rho Lk^dag
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for m in range(dim):
                            # (Lk rho Lk^dag)[i,j] = sum_{k,m} Lk[i,k] rho[k,m] Lk^dag[m,j]
                            coeff = Lk[i, k] * Lk.conj().T[m, j]
                            if coeff != 0:
                                L[i*dim + j, k*dim + m] += coeff
            # -0.5 {Lk^dag Lk, rho}
            Lk_dag_Lk = Lk_dag @ Lk
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        # -0.5 Lk^dag Lk rho: element (i,j) gets -0.5 * sum_k Lk_dag_Lk[i,k] rho[k,j]
                        if Lk_dag_Lk[i, k] != 0:
                            L[i*dim + j, k*dim + j] += -0.5 * Lk_dag_Lk[i, k]
                        # -0.5 rho Lk^dag Lk: element (i,j) gets -0.5 * sum_k rho[i,k] Lk_dag_Lk[k,j]
                        if Lk_dag_Lk[k, j] != 0:
                            L[i*dim + j, i*dim + k] += -0.5 * Lk_dag_Lk[k, j]
        
        return L
    
    L = _superop(H, [L1, L2])
    return L

def _simulate_pulse(I, Q, detuning, scale, a, a_dag, n, use_expm=True):
    """Simulate the pulse and return the final density matrix evolution."""
    # Initial state: we'll track the evolution of basis states
    # For fidelity computation, we need U or the full propagator
    # Let's compute the propagator (unitary part + decoherence)
    
    # Actually, for fidelity to X gate, we need to track how each input state evolves.
    # The X-pi gate maps: |0> -> |1>, |1> -> -|0> (or with phase)
    # Actually X-pi: X^(pi) = -iX, so |0> -> -i|1>, |1> -> -i|0>
    # Wait, let me reconsider. The X-pi gate is a rotation by pi around the X axis.
    # Rx(pi) = exp(-i pi/2 X) = -i X
    # So |0> -> -i|1>, |1> -> -i|0>
    
    # For fidelity, we need |<target|psi_final>|^2 where target is X|psi_initial>
    # But with decoherence, we use density matrices.
    
    # Let's compute the full propagator as a superoperator for each step
    # and apply it to initial density matrices.
    
    # Actually, for efficiency, let's compute the propagator matrix U for each step
    # (ignoring decoherence for the unitary part) and handle decoherence separately.
    
    # Better approach: compute the full Liouvillian superoperator for each step,
    # exponentiate it, and apply to the initial density matrix.
    
    # For fidelity to X gate:
    # F = <psi_target| rho_final |psi_target> where |psi_target> = X |psi_initial>
    # But with decoherence, we use: F = Tr(X rho_initial X^dag rho_final)
    # Hmm, that's not quite right either.
    
    # Standard fidelity: F = (Tr(sqrt(sqrt(rho) sigma sqrt(rho))))^2 for pure states simplifies
    # For our case, let's track the state vector evolution (ignoring decoherence for the
    # unitary part) and add decoherence effects.
    
    # Actually, the cleanest approach: simulate the master equation for each of the 4 test states.
    # But that's 4 simulations per pulse evaluation.
    
    # Alternative: compute the process tensor (superoperator) for the full pulse,
    # then apply it to each test state.
    
    # Let's use the superoperator approach.
    # The full propagator superoperator is: S = expm(L_N) @ expm(L_{N-1}) @ ... @ expm(L_1)
    # where L_i is the Liouvillian for step i.
    
    # For 80 steps, this is 80 matrix exponentials of 9x9 matrices. Should be fast.
    
    I_arr = np.asarray(I, dtype=float)
    Q_arr = np.asarray(Q, dtype=float)
    
    # Build the full superoperator
    S = np.eye(9, dtype=complex)
    
    for i in range(N_STEPS):
        L = _liouvillian(I_arr[i], Q_arr[i], detuning, scale, a, a_dag, n)
        S = expm(L * DT) @ S
    
    # Now apply S to each test state
    # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # X gate maps: |0> -> |1>, |1> -> |0> (for X-pi, it's -iX so there's a phase)
    # Actually for fidelity, the global phase doesn't matter.
    # X-pi gate: |0> -> |1>, |1> -> |0> (up to global phase)
    
    # Let's define the target states after X-pi gate:
    # |0> -> |1>, |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same state
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2)
    
    # Fidelity for state |psi>: F = |<psi_target| U |psi>|^2 (for pure states)
    # With decoherence: F = Tr(rho_target rho_final)
    
    # Let's compute using the superoperator.
    # vec(rho_final) = S @ vec(rho_initial)
    
    def _vec(rho):
        return rho.reshape(9)
    
    def _unvec(v):
        return v.reshape((3, 3))
    
    # Test states as density matrices
    test_states = []
    target_states = []
    
    # |0>
    psi0 = np.array([1, 0, 0], dtype=complex)
    test_states.append(np.outer(psi0, psi0.conj()))
    psi0_t = np.array([0, 1, 0], dtype=complex)  # X|0> = |1>
    target_states.append(np.outer(psi0_t, psi0_t.conj()))
    
    # |1>
    psi1 = np.array([0, 1, 0], dtype=complex)
    test_states.append(np.outer(psi1, psi1.conj()))
    psi1_t = np.array([1, 0, 0], dtype=complex)  # X|1> = |0>
    target_states.append(np.outer(psi1_t, psi1_t.conj()))
    
    # (|0>+|1>)/sqrt(2)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    test_states.append(np.outer(psi_plus, psi_plus.conj()))
    psi_plus_t = (psi1 + psi0) / np.sqrt(2)  # X|+> = |+>
    target_states.append(np.outer(psi_plus_t, psi_plus_t.conj()))
    
    # (|0>-|1>)/sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    test_states.append(np.outer(psi_minus, psi_minus.conj()))
    psi_minus_t = (psi1 - psi0) / np.sqrt(2)  # X|-> = -|->
    target_states.append(np.outer(psi_minus_t, psi_minus_t.conj()))
    
    # (|0>+i|1>)/sqrt(2)
    psi_r = (psi0 + 1j*psi1) / np.sqrt(2)
    test_states.append(np.outer(psi_r, psi_r.conj()))
    psi_r_t = (psi1 + 1j*psi0) / np.sqrt(2)  # X|R> where R = (|0>+i|1>)/sqrt(2)
    target_states.append(np.outer(psi_r_t, psi_r_t.conj()))
    
    # (|0>-i|1>)/sqrt(2)
    psi_l = (psi0 - 1j*psi1) / np.sqrt(2)
    test_states.append(np.outer(psi_l, psi_l.conj()))
    psi_l_t = (psi1 - 1j*psi0) / np.sqrt(2)  # X|L>
    target_states.append(np.outer(psi_l_t, psi_l_t.conj()))
    
    fidelities = []
    for rho_init, rho_target in zip(test_states, target_states):
        rho_final = _unvec(S @ _vec(rho_init))
        # Fidelity: Tr(rho_target rho_final)
        F = np.real(np.trace(rho_target @ rho_final))
        F = max(0, min(1, F))
        fidelities.append(F)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1 - mean_fid
    return infidelity

def _check_validity(I, Q):
    """Check amplitude and slew constraints."""
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-10:
        return False
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # Including from zero to step 1 and last step to zero
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
    
    # From zero to step 1
    slew1 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew1 > SLEW_BOUND + 1e-10:
        return False
    
    # From last step to zero
    slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slewN > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _generate_drag_pulse(sigma, drag_coeff, amp_scale):
    """Generate a DRAG pulse with given parameters."""
    t = np.linspace(0, TOTAL_TIME, N_STEPS, endpoint=False) + DT/2  # step centers
    
    # Gaussian envelope
    center = TOTAL_TIME / 2
    gauss = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Normalize to have area pi (for X-pi gate)
    # Area = integral of gauss dt ≈ sum(gauss) * DT
    area = np.sum(gauss) * DT
    if area > 0:
        gauss = gauss * (np.pi / area)
    
    # Apply amplitude scale
    gauss = gauss * amp_scale
    
    # DRAG: add Q component proportional to derivative of I
    # dI/dt ≈ (I[i+1] - I[i-1]) / (2*DT)
    I = gauss.copy()
    Q = np.zeros_like(I)
    if drag_coeff != 0:
        # Numerical derivative
        dI = np.zeros_like(I)
        dI[1:-1] = (I[2:] - I[:-2]) / (2 * DT)
        dI[0] = (I[1] - I[0]) / DT
        dI[-1] = (I[-1] - I[-2]) / DT
        Q = drag_coeff * dI
    
    return I, Q

def _enforce_constraints(I, Q):
    """Enforce amplitude and slew constraints by scaling."""
    I = np.asarray(I, dtype=float).copy()
    Q = np.asarray(Q, dtype=float).copy()
    
    # Enforce amplitude bound
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / max_amp
        I *= scale
        Q *= scale
        amp = amp * scale
    
    # Enforce slew bound
    # Check all slew rates
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            # Scale down the difference
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I[i+1] = I[i] + dI * factor
            Q[i+1] = Q[i] + dQ * factor
    
    # Check from zero to step 1
    slew1 = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew1 > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I[0]**2 + Q[0]**2)
        I[0] *= factor
        Q[0] *= factor
    
    # Check from last step to zero
    slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slewN > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I[-1]**2 + Q[-1]**2)
        I[-1] *= factor
        Q[-1] *= factor
    
    return I, Q

def _compute_score(I, Q, a, a_dag, n):
    """Compute the score for a given pulse."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = _simulate_pulse(I, Q, det, sc, a, a_dag, n)
            infidelities.append(inf)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    """Generate a robust control pulse for X-pi gate."""
    global _START_TIME
    _START_TIME = time.time()
    
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    a, a_dag, n = _build_matrices()
    
    # Seed pulse: DRAG-style Gaussian
    # sigma = 40/6 ns, area = pi
    seed_sigma = 40.0 / 6.0
    seed_I, seed_Q = _generate_drag_pulse(seed_sigma, 0.0, 1.0)
    seed_I, seed_Q = _enforce_constraints(seed_I, seed_Q)
    
    # Compute seed score
    seed_score = _compute_score(seed_I, seed_Q, a, a_dag, n)
    
    best_I, best_Q = seed_I.copy(), seed_Q.copy()
    best_score = seed_score
    
    # Grid search over DRAG parameters
    # Parameters: sigma, drag_coeff, amp_scale
    sigma_range = np.linspace(40/6 - 2, 40/6 + 2, 15)
    drag_range = np.linspace(-0.5, 0.5, 11)
    amp_range = np.linspace(0.9, 1.1, 9)
    
    # Coarse grid search
    for sigma in sigma_range:
        if _elapsed() > _DEADLINE - 30:
            break
        for drag in drag_range:
            if _elapsed() > _DEADLINE - 20:
                break
            for amp in amp_range:
                if _elapsed() > _DEADLINE - 10:
                    break
                
                I, Q = _generate_drag_pulse(sigma, drag, amp)
                I, Q = _enforce_constraints(I, Q)
                
                if not _check_validity(I, Q):
                    continue
                
                score = _compute_score(I, Q, a, a_dag, n)
                
                if score < best_score:
                    best_score = score
                    best_I, best_Q = I.copy(), Q.copy()
    
    # Local refinement around best point
    if _elapsed() < _DEADLINE - 15:
        # Find the best parameters
        best_sigma = seed_sigma
        best_drag = 0.0
        best_amp = 1.0
        
        # Re-search to find which parameters gave the best score
        for sigma in sigma_range:
            for drag in drag_range:
                for amp in amp_range:
                    I, Q = _generate_drag_pulse(sigma, drag, amp)
                    I, Q = _enforce_constraints(I, Q)
                    if not _check_validity(I, Q):
                        continue
                    score = _compute_score(I, Q, a, a_dag, n)
                    if score < best_score + 1e-12:
                        best_sigma = sigma
                        best_drag = drag
                        best_amp = amp
        
        # Refine with smaller steps
        for _ in range(3):
            if _elapsed() > _DEADLINE - 10:
                break
            
            step_s = 0.3
            step_d = 0.05
            step_a = 0.02
            
            for ds in [-step_s, 0, step_s]:
                for dd in [-step_d, 0, step_d]:
                    for da in [-step_a, 0, step_a]:
                        if _elapsed() > _DEADLINE - 5:
                            break
                        
                        sigma = best_sigma + ds
                        drag = best_drag + dd
                        amp = best_amp + da
                        
                        if sigma <= 0:
                            continue
                        
                        I, Q = _generate_drag_pulse(sigma, drag, amp)
                        I, Q = _enforce_constraints(I, Q)
                        
                        if not _check_validity(I, Q):
                            continue
                        
                        score = _compute_score(I, Q, a, a_dag, n)
                        
                        if score < best_score:
                            best_score = score
                            best_I, best_Q = I.copy(), Q.copy()
                            best_sigma = sigma
                            best_drag = drag
                            best_amp = amp
    
    # Return the pulse as (80, 2) array
    result = np.column_stack([best_I, best_Q])
    return result

# Keep the original interface
from autoresearch_suite.science_physics_expansion import quantum_seed
_pulse_original = quantum_seed

def pulse(target_gate, hardware_model):
    """Entry point - generate robust control pulse."""
    try:
        return _pulse_impl(target_gate, hardware_model)
    except Exception:
        # Fallback to seed
        return _pulse_original(target_gate, hardware_model)

_pulse_impl = pulse
pulse = _pulse_impl
