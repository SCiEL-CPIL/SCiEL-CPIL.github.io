"""Improved program for robust X-pi gate control pulse."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import expm

def _simulate_infidelity(I, Q, detuning, scale, T1, Tphi, anharmonicity):
    """
    Simulate the fidelity of the X_pi gate given I, Q pulses and hardware parameters.
    Returns the mean infidelity over the 4 test states.
    """
    dt = 0.5
    n_steps = len(I)
    
    # Hardware parameters
    # a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    
    # n = a^dagger a
    n = a_dag @ a
    
    # Hamiltonian terms
    # H = detuning * n + 0.5 * anharmonicity * n (n - 1) + 0.5 * scale * (I (a + a^dagger) + i Q (a - a^dagger))
    # Note: The contract says "0.5 anharmonicity n (n - 1)".
    # n(n-1) = n^2 - n. For a 3-level system, n^2 = diag(0, 1, 4).
    # n(n-1) = diag(0, 0, 2).
    # So the anharmonic term is 0.5 * anharmonicity * 2 * |2><2| = anharmonicity * |2><2|.
    # Let's stick to the matrix form for safety.
    n_nm1 = n @ (n - np.eye(3))
    
    H_detuning = detuning * n
    H_anh = 0.5 * anharmonicity * n_nm1
    
    # Drive term
    # I (a + a^dagger) + i Q (a - a^dagger)
    # a + a^dagger = X operator
    # a - a^dagger = -i Y operator (since Y = -i(a-a_dag) => a-a_dag = iY? No.
    # a = (X - iY)/2, a_dag = (X + iY)/2
    # a - a_dag = -iY
    # i Q (a - a_dagger) = i Q (-iY) = Q Y
    # I (a + a_dagger) = I X
    # So Drive = 0.5 * scale * (I X + Q Y)
    
    X_op = a + a_dag
    Y_op = 1j * (a_dag - a) # Standard Y = (a_dag - a)/(-i) ? 
    # Let's use the explicit matrix from the contract formula to avoid sign errors.
    # Term: I (a + a^dagger) + i Q (a - a^dagger)
    drive_mat = I * (a + a_dag) + 1j * Q * (a - a_dag)
    
    H_drive = 0.5 * scale * drive_mat
    
    # Total Hamiltonian
    H_static = H_detuning + H_anh
    
    # Lindblad Operators
    # L1 = a / sqrt(T1)
    # L2 = n * sqrt(2 / Tphi)
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Liouvillian L = -i H rho + ...
    # We will construct the superoperator for each step.
    # For efficiency, we can precompute the static parts if they don't change, 
    # but H_drive changes with I, Q.
    
    # Test states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_ip = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_im = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Target operator X_pi: maps |0> -> |1>, |1> -> -|0> (up to phase)
    # The gate is X_pi. U_target = exp(-i pi/2 X) = -i X.
    # Fidelity is usually |<psi_target|psi_final>|^2.
    # For X_pi, |0> goes to -i|1>, |1> goes to -i|0>.
    
    infidelities = []
    
    for psi_init in [psi0, psi1, psi_plus, psi_minus, psi_ip, psi_im]:
        rho = np.outer(psi_init, psi_init.conj())
        
        # Evolution
        for k in range(n_steps):
            H_drive_step = 0.5 * scale * (I[k] * (a + a_dag) + 1j * Q[k] * (a - a_dag))
            H_total = H_static + H_drive_step
            
            # Construct Liouvillian
            # L(rho) = -i [H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
            # We use the matrix representation of the superoperator.
            # Vectorize rho.
            # S = -i (H \otimes I - I \otimes H^T) + ...
            
            # Efficient way:
            # rho_next = expm(L * dt) @ rho_vec
            # But constructing L every step is expensive.
            # Let's just use the explicit formula for small dt if possible?
            # No, dt=0.5ns, frequencies ~ GHz (rad/ns ~ 6). H*dt ~ 3. Not small.
            # We must exponentiate.
            
            # Superoperator matrix S such that vec(rho_next) = expm(S*dt) vec(rho)
            # S = -i (H \otimes I - I \otimes H^T) + sum_k (L_k \otimes L_k^* - 0.5 (I \otimes L_k^dag L_k + L_k^dag L_k \otimes I))
            
            # Note: vec(A X B) = (A \otimes B^T) vec(X)
            # [H, rho] = H rho - rho H => vec(H rho - rho H) = (I \otimes H^T - H \otimes I) vec(rho)
            # So -i [H, rho] => -i (I \otimes H^T - H \otimes I) = i (H \otimes I - I \otimes H^T)
            
            S = 1j * (np.kron(H_total, np.eye(3)) - np.kron(np.eye(3), H_total.T))
            
            # Dissipation
            for L in [L1, L2]:
                L_dag = L.conj().T
                # L rho L^dag => (L \otimes L^*) vec(rho)
                # {L^dag L, rho} => 0.5 (I \otimes (L^dag L)^T + (L^dag L) \otimes I) vec(rho)
                
                LL_dag = L @ L_dag
                term1 = np.kron(L, L.conj())
                term2 = 0.5 * (np.kron(np.eye(3), LL_dag.T) + np.kron(LL_dag, np.eye(3)))
                
                S += term1 - term2
            
            rho_vec = rho.reshape(9)
            rho_vec = expm(S * dt) @ rho_vec
            rho = rho_vec.reshape(3, 3)
        
        # Calculate fidelity
        # Target state for psi_init under X_pi
        # X_pi = exp(-i pi/2 X). 
        # X |0> = |1>, X |1> = |0>.
        # exp(-i pi/2 X) = -i X.
        # So target |psi_t> = -i X |psi_init>
        
        X_mat = a + a_dag
        psi_target = -1j * (X_mat @ psi_init)
        
        # Fidelity = |<psi_target| psi_final>|^2
        # Or Tr(rho_target rho) for pure states?
        # F = |<psi_target|psi_final>|^2
        # We need psi_final from rho. Since it's open system, rho is mixed.
        # Fidelity for pure target |psi_t> and state rho is <psi_t|rho|psi_t>.
        
        fid = np.real(np.vdot(psi_target, rho @ psi_target))
        # Clamp to [0,1]
        fid = np.clip(fid, 0.0, 1.0)
        infidelities.append(1.0 - fid)
        
    return np.mean(infidelities)

def _score_pulse(params):
    """
    Objective function for optimization.
    params: array of coefficients for I and Q basis expansions.
    """
    # Basis functions: Hermite-Gaussian
    # t = 0..39.5
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    t_center = 20.0
    sigma = 40.0 / 6.0
    
    # Hermite polynomials
    H0 = np.ones_like(t)
    H1 = 2 * (t - t_center) / sigma
    H2 = 4 * ((t - t_center) / sigma)**2 - 2
    
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Basis vectors
    # We use 3 terms for I and 3 terms for Q
    # params = [cI0, cI1, cI2, cQ0, cQ1, cQ2]
    cI0, cI1, cI2 = params[0], params[1], params[2]
    cQ0, cQ1, cQ2 = params[3], params[4], params[5]
    
    I = (cI0 * H0 + cI1 * H1 + cI2 * H2) * gauss
    Q = (cQ0 * H0 + cQ1 * H1 + cQ2 * H2) * gauss
    
    # Check constraints
    amp = np.sqrt(I**2 + Q**2)
    if np.any(amp > 0.4):
        return 1.0 + np.max(amp - 0.4) # Penalty
    
    # Slew rate
    for i in range(n_steps - 1):
        diff_I = I[i+1] - I[i]
        diff_Q = Q[i+1] - Q[i]
        dist = np.sqrt(diff_I**2 + diff_Q**2)
        if dist / dt > 0.12:
            return 1.0 + dist/dt - 0.12
            
    # Endpoints
    d0 = np.sqrt(I[0]**2 + Q[0]**2)
    if d0 / dt > 0.12:
        return 1.0 + d0/dt - 0.12
    d_end = np.sqrt(I[-1]**2 + Q[-1]**2)
    if d_end / dt > 0.12:
        return 1.0 + d_end/dt - 0.12
        
    # Calculate score
    # Hardware params
    anharmonicity = -2 * np.pi * 0.25e-3 # rad/ns
    T1 = 25000.0
    Tphi = 40000.0
    
    detunings = [-2 * np.pi * 0.002e-3, 0.0, 2 * np.pi * 0.002e-3]
    scales = [0.97, 1.0, 1.03]
    
    scores = []
    for det in detunings:
        for scl in scales:
            inf = _simulate_infidelity(I, Q, det, scl, T1, Tphi, anharmonicity)
            scores.append(inf)
            
    scores = np.array(scores)
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    
    return mean_score + max_score

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Uses a Hermite-Gaussian basis expansion optimized via Differential Evolution.
    """
    # Define bounds for optimization
    # Coefficients should be small enough to stay within amplitude limits
    # Max amplitude ~0.4. Gauss peak ~1.
    # c0 ~ 0.4, c1 ~ 0.4, c2 ~ 0.4
    bounds = [
        (-0.5, 0.5), # cI0
        (-0.5, 0.5), # cI1
        (-0.5, 0.5), # cI2
        (-0.5, 0.5), # cQ0
        (-0.5, 0.5), # cQ1
        (-0.5, 0.5), # cQ2
    ]
    
    # Run optimization
    # Use a limited number of iterations to fit within time limits
    # n_iter=20, popsize=10 -> ~200 evaluations
    # Each eval is 9 simulations of 80 steps.
    # 9 * 80 * expm(9x9) is fast.
    
    result = differential_evolution(_score_pulse, bounds, seed=42, maxiter=50, popsize=10, tol=1e-4, polish=False)
    
    params = result.x
    
    # Reconstruct pulse
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    t_center = 20.0
    sigma = 40.0 / 6.0
    
    H0 = np.ones_like(t)
    H1 = 2 * (t - t_center) / sigma
    H2 = 4 * ((t - t_center) / sigma)**2 - 2
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    cI0, cI1, cI2 = params[0], params[1], params[2]
    cQ0, cQ1, cQ2 = params[3], params[4], params[5]
    
    I = (cI0 * H0 + cI1 * H1 + cI2 * H2) * gauss
    Q = (cQ0 * H0 + cQ1 * H1 + cQ2 * H2) * gauss
    
    # Post-process: Enforce constraints strictly
    # 1. Amplitude
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > 0.4:
        scale = 0.4 / max_amp
        I *= scale
        Q *= scale
    
    # 2. Slew rate
    # We do a simple iterative clipping.
    # Forward pass
    for i in range(n_steps - 1):
        diff_I = I[i+1] - I[i]
        diff_Q = Q[i+1] - Q[i]
        dist = np.sqrt(diff_I**2 + diff_Q**2)
        max_dist = 0.12 * dt
        if dist > max_dist:
            ratio = max_dist / dist
            I[i+1] = I[i] + diff_I * ratio
            Q[i+1] = Q[i] + diff_Q * ratio
            
    # Backward pass to fix any issues introduced by forward pass
    for i in range(n_steps - 2, -1, -1):
        diff_I = I[i] - I[i+1]
        diff_Q = Q[i] - Q[i+1]
        dist = np.sqrt(diff_I**2 + diff_Q**2)
        max_dist = 0.12 * dt
        if dist > max_dist:
            ratio = max_dist / dist
            I[i] = I[i+1] + diff_I * ratio
            Q[i] = Q[i+1] + diff_Q * ratio
            
    # 3. Endpoints
    d0 = np.sqrt(I[0]**2 + Q[0]**2)
    max_d0 = 0.12 * dt
    if d0 > max_d0:
        scale0 = max_d0 / d0
        I[0] *= scale0
        Q[0] *= scale0
        
    d_end = np.sqrt(I[-1]**2 + Q[-1]**2)
    max_d_end = 0.12 * dt
    if d_end > max_d_end:
        scale_end = max_d_end / d_end
        I[-1] *= scale_end
        Q[-1] *= scale_end
        
    p = np.column_stack([I, Q])
    
    return p

# EVOLVE-BLOCK-END
