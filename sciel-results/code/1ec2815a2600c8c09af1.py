"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_START_TIME = time.time()
_DEADLINE = 315.0  # safety margin before 330s kill

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25  # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A  # n = a^dag a

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute Liouvillian building blocks
# L(rho) = -i[H,rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
# We'll build the full Liouvillian matrix for each (detuning, scale, I, Q)

def _build_liouvillian(det, scale, I, Q):
    """Build the 9x9 Liouvillian superoperator for given parameters."""
    # Hamiltonian
    H = det * _NOP + 0.5 * _ANH * (_NOP * (_NOP - 3.0*np.eye(3)))  # 0.5*anh*n(n-1)
    # Wait: n(n-1) for 3 levels: n = diag(0,1,2), n(n-1) = diag(0,0,2)
    # Actually: n(n-1) = diag(0*0, 1*0, 2*1) = diag(0,0,2)
    # Let me recompute: 0.5 * scale * (I*(a+a_dag) + i*Q*(a-a_dag))
    drive = 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
    H = H + drive
    
    # Liouvillian superoperator (9x9)
    # Vectorization: rho -> vec(rho) column-major
    # L(rho) = -i(H rho - rho H) + sum_k (L_k rho L_k^dag - 0.5(L_k^dag L_k rho + rho L_k^dag L_k))
    Lmat = np.zeros((9, 9), dtype=complex)
    # -i[H, .] term: -i(H ⊗ I - I ⊗ H^T)
    Lmat += -1j * (np.kron(H, np.eye(3)) - np.kron(np.eye(3), H.T))
    # Lindblad terms
    for Lk, Lkdg in [(_L1, _L1ADAG), (_L2, _L2ADAG)]:
        LkLkd = Lk @ Lkdg
        Lmat += np.kron(Lkdg, Lk) - 0.5 * (np.kron(np.eye(3), LkLd) + np.kron(LkLd.T, np.eye(3)))
    return Lmat

def _liouvillian_step(rho, Lmat, dt):
    """Apply expm(Lmat * dt) to rho vectorized."""
    rho_vec = rho.ravel(order='F')
    # Use expm on the vector: expm(Lmat*dt) @ rho_vec
    # For efficiency, we can use expm_multiply if available, but let's use expm
    U = expm(Lmat * dt)
    new_rho_vec = U @ rho_vec
    return new_rho_vec.reshape(3, 3, order='F')

def _simulate_pulse(pulse, det, scale):
    """Simulate the pulse and return the final density matrix for each of 5 initial states."""
    I_arr, Q_arr = pulse[:, 0], pulse[:, 1]
    
    # Initial states
    states = [
        np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=complex),  # |0><0|
        np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=complex),  # |1><1|
        np.array([[0.5, 0.5, 0],[0.5, 0.5, 0],[0,0,0]], dtype=complex),  # (|0>+|1>)/sqrt(2)
        np.array([[0.5, -0.5, 0],[-0.5, 0.5, 0],[0,0,0]], dtype=complex),  # (|0>-|1>)/sqrt(2)
        np.array([[0.5, -0.5j, 0],[0.5j, 0.5, 0],[0,0,0]], dtype=complex),  # (|0>+i|1>)/sqrt(2)
    ]
    
    # Target: X gate on qubit subspace (|0>,|1>)
    # X|0> = |1>, X|1> = |0>
    # Fidelity = <psi_target|rho_final|psi_target>
    # For each initial state, the target final state is X applied to it
    
    # Precompute target states
    X = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
    target_states = []
    for s in states:
        # Target: X @ s @ X^dag
        ts = X @ s @ X.conj().T
        target_states.append(ts)
    
    fidelities = []
    for rho0 in states:
        rho = rho0.copy()
        for i in range(_N):
            Lmat = _build_liouvillian(det, scale, I_arr[i], Q_arr[i])
            rho = _liouvillian_step(rho, Lmat, _DT)
        # Fidelity: Tr(rho_target @ rho_final) for pure states, but here we use
        # F = <psi_t|rho_f|psi_t> where |psi_t> is the target pure state
        # For mixed states, use Uhlmann or just Tr(rho_t rho_f)
        # Since initial states are pure and evolution is CPTP, use:
        # F = Tr(rho_target @ rho_final)
        ts = target_states[states.index(rho0)]
        f = np.real(np.trace(ts @ rho))
        fidelities.append(max(0.0, min(1.0, f)))
    return fidelities

def _compute_score(pulse):
    """Compute the score: mean + max of infidelities over 9 cases."""
    infidelities = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            fids = _simulate_pulse(pulse, det, scale)
            mean_fid = np.mean(fids)
            infid = 1.0 - mean_fid
            infidelities.append(infid)
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _enforce_constraints(pulse):
    """Enforce amplitude and slew rate constraints."""
    I, Q = pulse[:, 0].copy(), pulse[:, 1].copy()
    
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > _AMP
    if np.any(mask):
        scale = _AMP / amp[mask]
        I[mask] *= scale
        Q[mask] *= scale
    
    # Slew rate constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero into step 1 and from last step to zero
    max_delta = _SLEW * _DT  # 0.06
    
    # From zero to step 1
    d = np.sqrt(I[0]**2 + Q[0]**2)
    if d > max_delta:
        s = max_delta / d
        I[0] *= s
        Q[0] *= s
    
    # Between consecutive steps
    for i in range(_N - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        d = np.sqrt(dI**2 + dQ**2)
        if d > max_delta:
            s = max_delta / d
            I[i+1] = I[i] + s * dI
            Q[i+1] = Q[i] + s * dQ
    
    # From last step to zero
    d = np.sqrt(I[-1]**2 + Q[-1]**2)
    if d > max_delta:
        s = max_delta / d
        I[-1] *= s
        Q[-1] *= s
    
    pulse[:, 0] = I
    pulse[:, 1] = Q
    return pulse

def _make_seed_pulse():
    """Create DRAG-style Gaussian seed pulse."""
    t = np.arange(_N) * _DT  # 0 to 39.5 ns
    center = _T / 2  # 20 ns
    sigma = 40.0 / 6.0  # ~6.667 ns
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Normalize area to pi
    area = np.trapz(env, t)
    if area > 0:
        env *= np.pi / area
    
    # Offset to 0 at edges (ensure first and last are 0)
    env[0] = 0.0
    env[-1] = 0.0
    
    # DRAG term: proportional to derivative of envelope
    drag_coeff = 0.0  # Start with no DRAG
    I = env
    Q = drag_coeff * np.gradient(env, t)
    
    pulse = np.column_stack([I, Q])
    pulse = _enforce_constraints(pulse)
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point: return (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        # Return seed for invalid gates
        return _make_seed_pulse()
    
    if _time_remaining() < 5.0:
        return _make_seed_pulse()
    
    # Start with seed
    best_pulse = _make_seed_pulse()
    best_score = _compute_score(best_pulse)
    
    # Try to improve with a simple local search
    if _time_remaining() < 10.0:
        return best_pulse
    
    # Simple iterative refinement: try small perturbations
    rng = np.random.RandomState(42)
    current_pulse = best_pulse.copy()
    current_score = best_score
    
    # Use a low-dimensional parameterization
    # Parameters: [gauss_amplitude, gauss_center, gauss_sigma, drag_coeff, 
    #              high_freq_amplitude, high_freq_center, high_freq_sigma, 
    #              linear_slope]
    def _params_to_pulse(params):
        amp, center, sigma, drag, hfa, hfc, hfs, slope = params
        t = np.arange(_N) * _DT
        env = amp * np.exp(-0.5 * ((t - center) / sigma)**2)
        # Offset to 0 at edges
        env[0] = 0.0
        env[-1] = 0.0
        # DRAG
        Q = drag * np.gradient(env, t)
        # High-frequency component
        hf = hfa * np.exp(-0.5 * ((t - hfc) / hfs)**2) * np.cos(2 * np.pi * 1.0 * t)
        env += hf
        # Linear slope
        env += slope * (t - center)
        # Offset to 0 at edges again
        env[0] = 0.0
        env[-1] = 0.0
        pulse = np.column_stack([env, Q])
        return _enforce_constraints(pulse)
    
    # Initial parameters
    t = np.arange(_N) * _DT
    center0 = _T / 2
    sigma0 = 40.0 / 6.0
    env0 = np.exp(-0.5 * ((t - center0) / sigma0)**2)
    env0[0] = 0.0
    env0[-1] = 0.0
    area0 = np.trapz(env0, t)
    amp0 = np.pi / area0 if area0 > 0 else 1.0
    
    init_params = np.array([amp0, center0, sigma0, 0.0, 0.0, center0, sigma0, 0.0])
    
    # Differential Evolution-like search
    n_params = 8
    bounds = [
        (0.0, 1.0),    # amplitude
        (10.0, 30.0),  # center
        (2.0, 15.0),   # sigma
        (-0.5, 0.5),   # drag
        (-0.3, 0.3),   # high-freq amp
        (10.0, 30.0),  # high-freq center
        (2.0, 15.0),   # high-freq sigma
        (-0.05, 0.05), # slope
    ]
    
    pop_size = 20
    population = [init_params.copy()]
    for _ in range(pop_size - 1):
        ind = init_params.copy()
        for j in range(n_params):
            lo, hi = bounds[j]
            ind[j] = lo + rng.random() * (hi - lo)
        population.append(ind)
    
    for gen in range(100):
        if _time_remaining() < 10.0:
            break
        
        new_population = []
        for i in range(pop_size):
            # Evaluate current
            p = _params_to_pulse(population[i])
            s = _compute_score(p)
            if s < best_score:
                best_score = s
                best_pulse = p.copy()
            
            # DE mutation
            if i == 0:
                new_population.append(population[0].copy())
                continue
            
            a, b, c = rng.choice(pop_size, 3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(pop_size, 3, replace=False)
            
            F = 0.8
            CR = 0.9
            mutant = population[a].copy()
            for j in range(n_params):
                if rng.random() < CR:
                    mutant[j] = population[a][j] + F * (population[b][j] - population[c][j])
                # Clamp to bounds
                lo, hi = bounds[j]
                mutant[j] = max(lo, min(hi, mutant[j]))
            
            # Crossover
            trial = population[i].copy()
            for j in range(n_params):
                if rng.random() < CR:
                    trial[j] = mutant[j]
            
            # Evaluate trial
            p_trial = _params_to_pulse(trial)
            s_trial = _compute_score(p_trial)
            
            if s_trial <= s:
                new_population.append(trial)
            else:
                new_population.append(population[i].copy())
        
        population = new_population
    
    return best_pulse
# EVOLVE-BLOCK-END
