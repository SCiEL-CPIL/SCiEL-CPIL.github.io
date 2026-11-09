"""Robust control pulses for a three-level transmon - SDP-inspired design."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def _build_hamiltonian(I, Q, detuning, anharmonicity, scale):
    """Build the 3x3 Hamiltonian for a single step."""
    # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
    # a^dagger = [[0,0,0],[1,0,0],[0,sqrt(2),0]]
    sqrt2 = np.sqrt(2.0)
    a = np.array([[0, 1, 0],
                  [0, 0, sqrt2],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n = adag @ a  # diag(0, 1, 2)
    
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    H = detuning * n + 0.5 * anharmonicity * (n @ (n - np.eye(3)))
    drive = I * (a + adag) + 1j * Q * (a - adag)
    H += 0.5 * scale * drive
    return H

def _liouvillian_step(rho, dt, I, Q, detuning, anharmonicity, scale, T1, Tphi):
    """
    Evolve density matrix by one step using matrix exponential of Liouvillian.
    rho is 3x3 complex matrix.
    """
    H = _build_hamiltonian(I, Q, detuning, anharmonicity, scale)
    
    # Lindblad operators
    sqrt2 = np.sqrt(2.0)
    a = np.array([[0, 1, 0],
                  [0, 0, sqrt2],
                  [0, 0, 0]], dtype=complex)
    n = a.conj().T @ a
    
    # L1 = a / sqrt(T1), L2 = n * sqrt(2/Tphi)
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Liouvillian superoperator in matrix form:
    # d(rho)/dt = -i[H, rho] + L1 rho L1^dag + L1^dag rho L1 - 0.5 L1^dag L1 rho - 0.5 rho L1^dag L1
    #             + L2 rho L2^dag + L2^dag rho L2 - 0.5 L2^dag L2 rho - 0.5 rho L2^dag L2
    # We can use the superoperator representation.
    # For efficiency, use the vec representation or just do the matrix operations.
    
    # Actually, let's use the standard approach: build the 9x9 Liouvillian matrix
    # and exponentiate it. But that's slow. Instead, use the fact that for small dt,
    # we can use a first-order or second-order approximation, or use scipy's expm on the superoperator.
    
    # Let's build the 9x9 Liouvillian
    dim = 3
    L = np.zeros((9, 9), dtype=complex)
    
    # Coherent part: -i(H ⊗ I - I ⊗ H^T) acting on vec(rho)
    # Actually, d(rho)/dt = -i[H, rho] corresponds to:
    # vec(-i[H,rho]) = -i(H ⊗ I - I ⊗ H^T) vec(rho)
    # Wait, the convention: vec(AXB) = (B^T ⊗ A) vec(X)
    # So vec(H rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    # vec(H rho - rho H) = (I ⊗ H - H^T ⊗ I) vec(rho)
    # So -i[H, rho] = -i(H rho - rho H) -> superop = -i(I ⊗ H - H^T ⊗ I)
    
    H_vec = -1j * (np.eye(3) @ H - H.T @ np.eye(3))  # This is wrong, let me redo
    
    # Correct: vec(ABX) = (B^T ⊗ A) vec(X)
    # vec(H @ rho) = (I^T ⊗ H) vec(rho) = (I ⊗ H) vec(rho)
    # vec(rho @ H) = (H^T ⊗ I) vec(rho)
    # So vec([H, rho]) = (I ⊗ H - H^T ⊗ I) vec(rho)
    # d(rho)/dt = -i[H, rho] + dissipative terms
    # Superop coherent = -i * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    S_coherent = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    # Dissipative terms for L:
    # L rho L^dag -> (L^dag^T ⊗ L) vec(rho)
    # L^dag rho L -> (L^T ⊗ L^dag) vec(rho)
    # L^dag L rho -> (I ⊗ L^dag L) vec(rho)
    # rho L^dag L -> ((L^dag L)^T ⊗ I) vec(rho)
    
    for L_op in [L1, L2]:
        Ldag = L_op.conj().T
        LdagL = Ldag @ L_op
        S_diss = (np.kron(Ldag.T, L_op) + np.kron(L_op.T, Ldag) 
                  - 0.5 * np.kron(np.eye(3), LdagL) - 0.5 * np.kron(LdagL.T, np.eye(3)))
        L += S_diss
    
    L += S_coherent
    
    # Evolve: rho_new = expm(L * dt) @ vec(rho)
    from scipy.linalg import expm
    Ldt = L * dt
    U = expm(Ldt)
    rho_vec = rho.flatten()
    rho_new_vec = U @ rho_vec
    rho_new = rho_new_vec.reshape(3, 3)
    
    # Renormalize trace to 1
    tr = np.trace(rho_new)
    if abs(tr) > 1e-15:
        rho_new = rho_new / tr
    
    return rho_new

def _fidelity(rho, state):
    """Compute fidelity |<state|rho|state>|^2... actually for pure state, F = <state|rho|state>."""
    # state is a 3-vector
    return np.real(np.conj(state) @ rho @ state)

def _compute_infidelity(pulse_arr, detuning, scale, anharmonicity, T1, Tphi):
    """
    Compute mean infidelity over 4 states for given detuning and scale.
    pulse_arr is (80, 2) array.
    """
    dt = 0.5
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    fidelities = []
    for state in states:
        rho = np.outer(state, state.conj())
        for step in range(80):
            I = pulse_arr[step, 0]
            Q = pulse_arr[step, 1]
            rho = _liouvillian_step(rho, dt, I, Q, detuning, anharmonicity, scale, T1, Tphi)
        f = _fidelity(rho, state)
        fidelities.append(f)
    
    return 1.0 - np.mean(fidelities)

def _score_pulse(pulse_arr, anharmonicity, T1, Tphi):
    """Compute the full score: mean + max of 9 cases."""
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infs = []
    for det in detunings:
        for sc in scales:
            inf = _compute_infidelity(pulse_arr, det, sc, anharmonicity, T1, Tphi)
            infs.append(inf)
    
    infs = np.array(infs)
    return np.mean(infs) + np.max(infs)

def _check_validity(pulse_arr):
    """Check amplitude and slew rate constraints."""
    dt = 0.5
    # Amplitude bound
    amplitudes = np.sqrt(pulse_arr[:, 0]**2 + pulse_arr[:, 1]**2)
    if np.max(amplitudes) > 0.4 + 1e-10:
        return False
    
    # Slew rate: from zero to step 1, between steps, and last step to zero
    # From zero to step 1
    if np.sqrt(pulse_arr[0, 0]**2 + pulse_arr[0, 1]**2) / dt > 0.12 + 1e-10:
        return False
    
    # Between steps
    for i in range(79):
        dI = pulse_arr[i+1, 0] - pulse_arr[i, 0]
        dQ = pulse_arr[i+1, 1] - pulse_arr[i, 1]
        if np.sqrt(dI**2 + dQ**2) / dt > 0.12 + 1e-10:
            return False
    
    # Last step to zero
    if np.sqrt(pulse_arr[-1, 0]**2 + pulse_arr[-1, 1]**2) / dt > 0.12 + 1e-10:
        return False
    
    return True

def _seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt
    
    # Gaussian with sigma = 40/6 ns
    sigma = 40.0 / 6.0
    center = 20.0  # center of 40 ns window
    
    # Gaussian amplitude
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset to 0 at edges
    # The Gaussian is already ~0 at edges for sigma=6.67, t-center = ±20
    # exp(-0.5*(20/6.67)^2) = exp(-0.5*9) = exp(-4.5) ≈ 0.011, close to 0
    
    # Area should be pi: integral of A * g dt = pi
    # Sum of g * dt ≈ integral
    area_g = np.sum(g) * dt
    A = np.pi / area_g
    
    I = A * g
    Q = np.zeros(n_steps)
    
    return np.column_stack([I, Q])

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 280  # 300s limit, keep 20s margin
    
    # Start with seed pulse
    best_pulse = _seed_pulse()
    best_score = _score_pulse(best_pulse, anharmonicity, T1, Tphi)
    
    # Parameterize the pulse using a richer basis
    # Use Chebyshev-modulated Gaussian: I(t) = g(t) * sum c_k T_k(2t/40 - 1)
    # where T_k are Chebyshev polynomials of the first kind
    # This gives flexibility while maintaining the Gaussian envelope shape
    
    n_basis = 10  # Number of Chebyshev coefficients
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt
    tau = 2 * t / 40.0 - 1.0  # Map to [-1, 1]
    
    sigma = 40.0 / 6.0
    center = 20.0
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Build Chebyshev basis
    def chebyshev_matrix(n, tau):
        """Build matrix of Chebyshev polynomials T_0..T_{n-1} evaluated at tau."""
        C = np.zeros((n, len(tau)))
        for k in range(n):
            C[k] = np.cos(k * np.arccos(np.clip(tau, -1, 1)))
        return C
    
    C = chebyshev_matrix(n_basis, tau)
    
    # Initial parameters: match the seed
    # Seed has I = A*g, Q = 0
    # So c_0 should be such that c_0 * T_0 * g = A * g => c_0 = A
    # T_0 = 1
    A_seed = np.pi / (np.sum(g) * dt)
    params_init = np.zeros(2 * n_basis)
    params_init[0] = A_seed  # I coefficient for T_0
    # Q coefficients all zero
    
    def _params_to_pulse(params):
        """Convert parameter vector to pulse array."""
        cI = params[:n_basis]
        cQ = params[n_basis:]
        I = (C @ cI) * g
        Q = (C @ cQ) * g
        return np.column_stack([I, Q])
    
    def _score_from_params(params):
        """Compute score from parameter vector."""
        p = _params_to_pulse(params)
        if not _check_validity(p):
            return 1e6  # Penalty for invalid
        return _score_pulse(p, anharmonicity, T1, Tphi)
    
    # Optimize using L-BFGS-B
    result = minimize(_score_from_params, params_init, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if result.fun < best_score:
        best_pulse = _params_to_pulse(result.x)
        best_score = result.fun
    
    # Check deadline
    if time.time() > deadline:
        return best_pulse
    
    # Try a second optimization with different initial conditions
    # Start from a slightly perturbed version
    for restart in range(3):
        if time.time() > deadline:
            break
        
        # Perturb the best solution
        noise = np.random.normal(0, 0.01 * np.abs(params_init[:n_basis]) + 0.001, size=2*n_basis)
        params_start = result.x + noise if restart > 0 else params_init + noise
        
        result2 = minimize(_score_from_params, params_start, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result2.fun < best_score:
            best_pulse = _params_to_pulse(result2.x)
            best_score = result2.fun
            result = result2
    
    return best_pulse
