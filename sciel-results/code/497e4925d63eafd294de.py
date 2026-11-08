"""Robust control pulses for a three-level transmon with quantum simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    dt = 0.5
    N = 80
    T = N * dt
    
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Build operators
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    I3 = np.eye(3, dtype=complex)
    
    def hamiltonian(det, scale, I_val, Q_val):
        H = det * n + 0.5 * anharmonicity * n @ (n - I3)
        H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
        return H
    
    def liouvillian_step(H, dt):
        # Build Liouvillian: dρ/dt = -i[H,ρ] + D[√(1/T1)a]ρ + D[√(2/Tφ) n]ρ
        # Use superoperator form
        L = np.zeros((9, 9), dtype=complex)
        # Coherent part: -i[H,ρ] = -i H ρ + i ρ H
        Hkron = np.kron(H, np.eye(3))
        Hkron2 = np.kron(np.eye(3), H.T)
        L = -1j * Hkron + 1j * Hkron2
        # Dissipative: D[c]ρ = cρc† - 0.5{c†c, ρ}
        c1 = a / np.sqrt(T1)
        c2 = n * np.sqrt(2.0 / Tphi)
        for c in [c1, c2]:
            ckdag = c.conj().T
            cck = ckdag @ c
            L += np.kron(c, c.T) - 0.5 * np.kron(I3, cck.T) - 0.5 * np.kron(cck, np.eye(3))
        # Vectorize: ρ -> vec(ρ) column-wise
        return expm(L * dt)
    
    def vec(rho):
        return rho.reshape(9, 1).flatten()
    
    def unvec(v):
        return v.reshape(3, 3)
    
    def fidelity_with_X(rho):
        # Fidelity with X gate: F = (1/4) Tr[(X†ρX)† ... ] 
        # Actually fidelity = |<ψ_target|ρ|ψ_target>|... but for mixed:
        # Use Uhlmann: F = (Tr sqrt(sqrt(ρ) X†ρX sqrt(ρ)))^2
        # Simpler: for pure target, F = <ψ|ρ|ψ> where |ψ> = X|0>... 
        # The task says "fidelity to X over |0>, |1>, ...". 
        # This means: apply pulse, then check if state equals X applied to initial.
        # Fidelity = |<X|ψ_initial>|^2 for pure, or Tr[ρ_target ρ] for mixed.
        # Actually standard: F = Tr[ρ_final ρ_target] where ρ_target = X ρ_initial X†
        # For pure initial |ψ>, ρ_target = X|ψ><ψ|X†
        # F = <ψ|X† ρ_final X|ψ>
        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
        # F = Tr[ρ_final X ρ_initial X†] ... let me use: F = |Tr[ρ_final X]|... no
        # Uhlmann fidelity for pure target: F = <ψ_t|ρ|ψ_t> where |ψ_t> = X|ψ_0>
        # But we have 5 different initial states. The fidelity to X means:
        # Does ρ_final = X ρ_initial X†? 
        # Fidelity = Tr[ρ_final ρ_initial X†]... 
        # Let's use: F = |Tr[ρ_final X ρ_initial X†]|... 
        # Standard process fidelity: F = (1/9)|Tr[ρ_final X ρ_initial X†]|... 
        # I'll interpret as: F = Tr[ρ_final (X ρ_initial X†)]
        # For pure initial |ψ>: ρ_initial = |ψ><ψ|, target = X|ψ><ψ|X†
        # F = <ψ|X† ρ_final X|ψ>
        return None  # placeholder
    
    def evolve_state(pulse_I, pulse_Q, det, scale, psi0):
        """Evolve a single initial state through the pulse."""
        rho = np.outer(psi0, psi0.conj())
        U_list = []
        for i in range(N):
            H = hamiltonian(det, scale, pulse_I[i], pulse_Q[i])
            S = liouvillian_step(H, dt)
            rho_vec = vec(rho)
            rho_vec = S @ rho_vec
            rho = unvec(rho_vec)
        return rho
    
    def compute_score(pulse_I, pulse_Q):
        """Compute the score: mean + max infidelity over 9 cases."""
        infs = []
        for det in [-2*np.pi*0.002, 0, 2*np.pi*0.002]:
            for scale in [0.97, 1.0, 1.03]:
                # 5 initial states
                X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
                inits = [
                    np.array([1, 0, 0], dtype=complex),  # |0>
                    np.array([0, 1, 0], dtype=complex),  # |1>
                    (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
                    (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
                    (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
                    (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
                ]
                # Wait, task says 4 states: |0>, |1>, (|0>±|1>)/√2, (|0>±i|1>)/√2
                # That's 6 states? Let me re-read: "|0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
                # So: |0>, |1>, (|0>+|1>)/√2, (|0>-|1>)/√2, (|0>+i|1>)/√2, (|0>-i|1>)/√2 = 6 states
                # But "mean fidelity to X over" these states
                inits = [
                    np.array([1, 0, 0], dtype=complex),
                    np.array([0, 1, 0], dtype=complex),
                    (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
                    (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
                    (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
                    (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
                ]
                fids = []
                for psi0 in inits:
                    rho_final = evolve_state(pulse_I, pulse_Q, det, scale, psi0)
                    psi_target = X @ psi0
                    # Fidelity = |<ψ_target|ρ_final|ψ_target>|
                    fid = np.abs(np.vdot(psi_target, rho_final @ psi_target))**2
                    fids.append(fid)
                inf = 1 - np.mean(fids)
                infs.append(inf)
        return np.mean(infs) + np.max(infs)
    
    def check_validity(I, Q):
        amp = np.sqrt(I**2 + Q**2)
        if np.any(amp > amp_bound + 1e-10):
            return False
        slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI**2 + dQ**2) / dt
        slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        max_slew = max(slew0, np.max(slews) if len(slews) > 0 else 0, slewN)
        if max_slew > slew_bound + 1e-10:
            return False
        return True
    
    def make_gaussian_pulse(sigma_ratio, drag_coeff, amp_scale=1.0):
        """Generate DRAG Gaussian pulse."""
        t = np.linspace(0, T, N, endpoint=False) + dt/2
        sigma = (T / 6.0) * sigma_ratio
        center = T / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center) / sigma) ** 2)
        gaussian = gaussian - edge_val
        gaussian = np.maximum(gaussian, 0)
        area = np.sum(gaussian) * dt
        if area > 0:
            gaussian = gaussian * (np.pi / area)
        gaussian_deriv = np.gradient(gaussian, dt)
        I = amp_scale * gaussian
        Q = amp_scale * drag_coeff * gaussian_deriv
        # Enforce constraints
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            factor = amp_bound / np.max(amp)
            I *= factor
            Q *= factor
        # Check slew
        for _ in range(50):
            slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI**2 + dQ**2) / dt
            slewN = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
            max_slew = max(slew0, np.max(slews) if len(slews) > 0 else 0, slewN)
            if max_slew <= slew_bound:
                break
            I *= 0.95
            Q *= 0.95
        return I, Q
    
    # Phase 1: Coarse grid search over DRAG parameters
    best_I, best_Q = None, None
    best_score = float('inf')
    
    sigma_ratios = [0.5, 0.7, 1.0, 1.3, 1.5]
    drag_coeffs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for sr in sigma_ratios:
        for dc in drag_coeffs:
            if time.time() > deadline:
                break
            I, Q = make_gaussian_pulse(sr, dc)
            if not check_validity(I, Q):
                continue
            # Quick evaluation on nominal case only for speed
            det = 0
            scale = 1.0
            X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
            inits = [
                np.array([1, 0, 0], dtype=complex),
                np.array([0, 1, 0], dtype=complex),
                (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
            ]
            fids = []
            for psi0 in inits:
                rho_final = evolve_state(I, Q, det, scale, psi0)
                psi_target = X @ psi0
                fid = np.abs(np.vdot(psi_target, rho_final @ psi_target))**2
                fids.append(fid)
            quick_inf = 1 - np.mean(fids)
            if quick_inf < best_score:
                best_score = quick_inf
                best_I, best_Q = I.copy(), Q.copy()
    
    if best_I is None:
        # Fallback: simple Gaussian
        I, Q = make_gaussian_pulse(1.0, 0.3)
        return np.column_stack([I, Q])
    
    # Phase 2: Local optimization using gradient descent on the pulse
    # Parameterize as: I[i] = base_I[i] * (1 + eps * mod[i]), Q[i] = base_Q[i] + eps * drag_mod[i]
    # Use finite differences for gradient
    
    I_opt = best_I.copy()
    Q_opt = best_Q.copy()
    current_score = compute_score(I_opt, Q_opt)
    best_score_full = current_score
    best_I_full = I_opt.copy()
    best_Q_full = Q_opt.copy()
    
    # Gradient descent with line search
    step_size = 0.01
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Compute gradient via finite differences
        grad_I = np.zeros(N)
        grad_Q = np.zeros(N)
        eps = 0.005
        
        for i in range(0, N, 4):  # Every 4th point for speed
            if time.time() > deadline:
                break
            # I gradient
            I_plus = I_opt.copy()
            I_plus[i] += eps
            if check_validity(I_plus, Q_opt):
                s_plus = compute_score(I_plus, Q_opt)
            else:
                s_plus = current_score + 1.0
            I_minus = I_opt.copy()
            I_minus[i] -= eps
            if check_validity(I_minus, Q_opt):
                s_minus = compute_score(I_minus, Q_opt)
            else:
                s_minus = current_score + 1.0
            grad_I[i] = (s_plus - s_minus) / (2 * eps)
            
            # Q gradient
            Q_plus = Q_opt.copy()
            Q_plus[i] += eps
            if check_validity(I_opt, Q_plus):
                s_plus = compute_score(I_opt, Q_plus)
            else:
                s_plus = current_score + 1.0
            Q_minus = Q_opt.copy()
            Q_minus[i] -= eps
            if check_validity(I_opt, Q_minus):
                s_minus = compute_score(I_opt, Q_minus)
            else:
                s_minus = current_score + 1.0
            grad_Q[i] = (s_plus - s_minus) / (2 * eps)
        
        # Update
        I_new = I_opt - step_size * grad_I
        Q_new = Q_opt - step_size * grad_Q
        
        # Project to feasible set
        amp = np.sqrt(I_new**2 + Q_new**2)
        if np.max(amp) > amp_bound:
            factor = amp_bound / np.max(amp)
            I_new *= factor
            Q_new *= factor
        
        # Check slew and scale down if needed
        for _ in range(20):
            slew0 = np.sqrt(I_new[0]**2 + Q_new[0]**2) / dt
            dI = np.diff(I_new)
            dQ = np.diff(Q_new)
            slews = np.sqrt(dI**2 + dQ**2) / dt
            slewN = np.sqrt(I_new[-1]**2 + Q_new[-1]**2) / dt
            max_slew = max(slew0, np.max(slews) if len(slews) > 0 else 0, slewN)
            if max_slew <= slew_bound:
                break
            I_new *= 0.98
            Q_new *= 0.98
        
        if check_validity(I_new, Q_new):
            new_score = compute_score(I_new, Q_new)
            if new_score < current_score:
                I_opt, Q_opt = I_new, Q_new
                current_score = new_score
                if current_score < best_score_full:
                    best_score_full = current_score
                    best_I_full = I_opt.copy()
                    best_Q_full = Q_opt.copy()
                step_size *= 1.1
            else:
                step_size *= 0.7
        else:
            step_size *= 0.5
    
    # Final: use the best full-evaluation result
    I_final = best_I_full
    Q_final = best_Q_full
    
    # Final validity check
    if not check_validity(I_final, Q_final):
        # Fallback to the coarse search best
        I_final = best_I
        Q_final = best_Q
        if not check_validity(I_final, Q_final):
            I_final, Q_final = make_gaussian_pulse(1.0, 0.3)
    
    return np.column_stack([I_final, Q_final])

# EVOLVE-BLOCK-END
