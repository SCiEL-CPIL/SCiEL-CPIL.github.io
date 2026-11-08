import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin

    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0
    
    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # The 6 states as specified
    sqrt2 = np.sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2
    ]
    
    # Target states for X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([-1, 1, 0], dtype=complex) / sqrt2,
        np.array([1j, 1, 0], dtype=complex) / sqrt2,
        np.array([-1j, 1, 0], dtype=complex) / sqrt2
    ]
    
    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Precompute constant parts
    H_anharm = 0.5 * alpha * (n_op @ (n_op - np.eye(3)))
    
    # Dissipator superoperator (constant)
    dim = 3
    d2 = dim * dim
    L_diss_super = np.zeros((d2, d2), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L1_term = np.kron(L_op, L_op.conj())
        L2_term = 0.5 * (np.kron(L_dag @ L_op, np.eye(dim)) + np.kron(np.eye(dim), (L_dag @ L_op).T))
        L_diss_super += L1_term - L2_term
    
    # Precompute identity for kron
    I3 = np.eye(3)
    
    # Precompute Liouvillian parts for all 9 cases
    liouvillian_parts = {}
    for delta in deltas:
        for scale in scales:
            H0 = delta * n_op + H_anharm
            HI = 0.5 * scale * (a_op + a_dag)
            HQ = 0.5j * scale * (a_op - a_dag)
            
            H_term_0 = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
            H_term_I = -1j * (np.kron(HI, I3) - np.kron(I3, HI.T))
            H_term_Q = -1j * (np.kron(HQ, I3) - np.kron(I3, HQ.T))
            
            L_base_0 = H_term_0 + L_diss_super
            liouvillian_parts[(delta, scale)] = (L_base_0, H_term_I, H_term_Q)
    
    # Precompute state vectors and target projectors
    rho0_vecs = []
    for idx in range(6):
        psi0 = states[idx]
        rho0 = np.outer(psi0, psi0.conj())
        rho0_vecs.append(rho0.flatten())
    
    target_projs = []
    for idx in range(6):
        t = targets[idx]
        target_projs.append(np.outer(t, t.conj()).flatten())

    # Basis: DCT-II
    # We use K coefficients for I and K for Q
    K = 12 
    # Construct DCT matrix for N points
    # scipy.fft.dct / idct are available but we need the matrix for fast evaluation in loop?
    # Actually, we can just use scipy.fft.idct for conversion in the objective.
    
    def coeffs_to_pulse(coeffs):
        """
        coeffs: (2K,) array. First K are I coeffs, next K are Q coeffs.
        Returns (80, 2) array.
        """
        I_coeffs = coeffs[:K]
        Q_coeffs = coeffs[K:]
        
        # idct type 2, normalized
        I = scipy_fft_idct(I_coeffs, type=2, norm='ortho', n=N)
        Q = scipy_fft_idct(Q_coeffs, type=2, norm='ortho', n=N)
        
        I_Q = np.column_stack((I, Q))
        
        # Repair constraints
        I_Q = repair_pulse(I_Q)
        
        return I_Q

    # Import idct locally to avoid global dependency issues if not imported at top
    from scipy.fft import idct as scipy_fft_idct

    def repair_pulse(I_Q):
        """
        Enforce amplitude and slew constraints.
        """
        # 1. Amplitude constraint
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        mask = amps > amp_bound
        if np.any(mask):
            scale = amp_bound / amps[mask]
            I_Q[mask, 0] *= scale
            I_Q[mask, 1] *= scale
        
        # 2. Slew constraint
        max_step = slew_bound * dt
        # Iterate to fix slew
        for _ in range(10):
            # Forward pass
            for i in range(1, N):
                diff = I_Q[i] - I_Q[i-1]
                dnorm = np.linalg.norm(diff)
                if dnorm > max_step:
                    # Move I_Q[i] towards I_Q[i-1]
                    # I_Q[i] = I_Q[i-1] + diff * (max_step / dnorm)
                    I_Q[i] = I_Q[i-1] + diff * (max_step / dnorm)
            
            # Backward pass
            for i in range(N - 2, -1, -1):
                diff = I_Q[i] - I_Q[i+1]
                dnorm = np.linalg.norm(diff)
                if dnorm > max_step:
                    I_Q[i] = I_Q[i+1] + diff * (max_step / dnorm)
            
            # Check boundaries
            amp0 = np.linalg.norm(I_Q[0])
            if amp0 > max_step:
                I_Q[0] *= max_step / amp0
            amplast = np.linalg.norm(I_Q[-1])
            if amplast > max_step:
                I_Q[-1] *= max_step / amplast
            
            # Re-check amplitude
            amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
            mask = amps > amp_bound
            if np.any(mask):
                scale = amp_bound / amps[mask]
                I_Q[mask, 0] *= scale
                I_Q[mask, 1] *= scale

        return I_Q

    def check_constraints(I_Q):
        """Check amplitude and slew constraints."""
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        # Slew from zero to first step
        if np.linalg.norm(I_Q[0]) / dt > slew_bound + 1e-9:
            return False
        # Slew from last step to zero
        if np.linalg.norm(I_Q[-1]) / dt > slew_bound + 1e-9:
            return False
        # Slew between consecutive steps
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-9:
            return False
        return True

    def calculate_score(I_Q):
        """
        Calculates the score: mean + max of the 9 cases.
        """
        case_scores = []
        for delta in deltas:
            for scale in scales:
                L_base_0, H_term_I, H_term_Q = liouvillian_parts[(delta, scale)]
                
                # Simulate 6 states
                # rho_vecs is (6, 9)
                rho_vecs = np.array(rho0_vecs) # shape (6, 9)
                
                for i in range(N):
                    I_val = I_Q[i, 0]
                    Q_val = I_Q[i, 1]
                    L_total = L_base_0 + I_val * H_term_I + Q_val * H_term_Q
                    U = expm(L_total * dt)
                    # Apply U to all 6 states
                    # rho_vecs is (6, 9). U is (9, 9).
                    # We want new_rho_vecs[j] = U @ rho_vecs[j]
                    # This is (U @ rho_vecs.T).T
                    rho_vecs = (U @ rho_vecs.T).T
                
                # Calculate fidelity
                infid_sum = 0.0
                for idx in range(6):
                    rho_mat = rho_vecs[idx].reshape(3, 3)
                    fid = np.real(np.trace(rho_mat @ target_projs[idx].reshape(3, 3)))
                    fid = np.clip(fid, 0, 1)
                    infid_sum += (1.0 - fid)
                
                mean_inf = infid_sum / 6.0
                case_scores.append(mean_inf)
        
        return np.mean(case_scores) + np.max(case_scores)

    # Initialize with seed Gaussian
    # Generate seed pulse
    t = np.linspace(0, 40, N)
    sigma_seed = 40.0 / 6.0
    A = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A * np.exp(-(t - 20.0)**2 / (2 * sigma_seed**2))
    Q_seed = np.zeros(N)
    I_Q_seed = np.column_stack((I_seed, Q_seed))
    I_Q_seed = repair_pulse(I_Q_seed)
    
    # Project seed onto DCT basis
    # Use dct type 2
    from scipy.fft import dct as scipy_fft_dct
    I_coeffs_seed = scipy_fft_dct(I_seed, type=2, norm='ortho')[:K]
    Q_coeffs_seed = scipy_fft_dct(Q_seed, type=2, norm='ortho')[:K]
    
    x0 = np.concatenate([I_coeffs_seed, Q_coeffs_seed])
    
    best_I_Q = I_Q_seed.copy()
    best_score = calculate_score(best_I_Q)
    
    def objective(coeffs):
        if time.time() > deadline - 5.0:
            return 1e6
        I_Q = coeffs_to_pulse(coeffs)
        if not check_constraints(I_Q):
            return 1e6
        score = calculate_score(I_Q)
        return score
    
    # Optimization
    # Use Nelder-Mead on the coefficients
    # Multiple restarts with different initializations
    
    # 1. Start from seed
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_I_Q = coeffs_to_pulse(res.x)
    except Exception as e:
        pass
    
    # 2. Try a few random restarts or perturbed seeds
    # If time permits
    if time.time() < deadline - 30.0:
        # Perturb seed
        for _ in range(3):
            x0_pert = x0 + np.random.normal(0, 0.1, size=len(x0))
            try:
                res = minimize(objective, x0_pert, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_I_Q = coeffs_to_pulse(res.x)
            except Exception:
                pass
            
            if time.time() > deadline - 10.0:
                break

    # Final check
    if not check_constraints(best_I_Q):
        best_I_Q = repair_pulse(best_I_Q)
        if not check_constraints(best_I_Q):
            best_I_Q = I_Q_seed

    return best_I_Q
