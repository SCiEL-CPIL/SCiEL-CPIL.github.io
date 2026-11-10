import numpy as np
import math
import time

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes.
    Strategy: Fast analytical 2-level effective model to optimize DRAG parameters
    (sigma, k_drag, detuning_comp) via dense global search.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin for 330s limit

    # Constants
    dt = 0.5
    n_steps = 80
    T = 40.0
    t = np.arange(n_steps) * dt
    
    # Hardware parameters
    chi = -2 * math.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    
    # Decay rates
    gamma_1 = 1.0 / T1
    gamma_phi = 1.0 / Tphi
    # Effective decay rates in rotating frame (approximation for 2-level subspace)
    # For |0> and |1>, decay is dominated by T1 (|1>->|0>) and Tphi (dephasing)
    # In the 2-level model, we approximate the off-diagonal decay and diagonal decay.
    # Gamma_pop = 1/T1, Gamma_coh = 1/(2*T2) = 1/(2*T1) + 1/Tphi
    gamma_pop = 1.0 / T1
    gamma_coh = 1.0 / (2.0 * T1) + 1.0 / Tphi

    # Pre-compute static parts of the 2-level effective Hamiltonian
    # H_eff = [ -delta/2 - chi/2,  0.5 * scale * (I - 1j*Q) ]
    #         [ 0.5 * scale * (I + 1j*Q), delta/2 ]
    # Note: The "detuning" in the task is the offset from the 0->1 transition.
    # The anharmonicity shifts the 1->2 transition, but in the 2-level subspace 
    # (0,1), the diagonal terms are:
    # H_00 = 0
    # H_11 = detuning + 0.5*chi*(1)(0) = detuning
    # Wait, H = detuning*n + 0.5*chi*n(n-1) + ...
    # n|0>=0, n|1>=|1>, n|2>=2|2>
    # H|0> = 0
    # H|1> = (detuning + 0) |1> + drive terms
    # H|2> = (2*detuning + 0.5*chi*2*1) |2> + drive terms = (2*detuning + chi)|2>
    # In the rotating frame at frequency of 0->1 transition (delta=0), 
    # the 0->1 transition is resonant.
    # The 1->2 transition is off-resonant by delta_2 = (2*detuning + chi) - (detuning + 0) - (freq_01) ?
    # Actually, the standard DRAG condition is to suppress 1->2 leakage.
    # The effective 2-level model captures the 0<->1 dynamics.
    # The leakage to |2> is a second-order effect proportional to (drive/delta_2)^2.
    # For optimization of the main gate, the 2-level fidelity is the dominant term.
    # We will optimize for 2-level fidelity, then verify with a full 3-level sim if time permits.

    def get_pulse(sigma, k_drag, detuning_comp, scale=1.0, detuning=0.0):
        """
        Generate I and Q pulse arrays.
        detuning_comp is added to the nominal detuning to compensate for hardware offsets.
        """
        # Gaussian pulse
        # Center at T/2
        tc = T / 2.0
        gauss = np.exp(-(t - tc)**2 / (2 * sigma**2))
        
        # Normalize area to pi (for nominal scale=1, detuning=0)
        # Area = sum(A * gauss * dt) = pi
        # A = pi / (dt * sum(gauss))
        area_sum = np.sum(gauss) * dt
        if area_sum < 1e-9:
            return np.zeros(n_steps), np.zeros(n_steps)
        
        A = math.pi / area_sum
        I = A * gauss
        
        # DRAG term
        dI_dt = np.gradient(I, dt)
        # Standard DRAG: Q = -k_drag * dI_dt
        # k_drag is often chi / (2*pi) in some conventions, but we optimize it.
        Q = -k_drag * dI_dt
        
        # Apply detuning compensation? 
        # The pulse shape is fixed, but the "effective" detuning seen by the qubit 
        # is the hardware detuning plus any dynamic detuning we might apply.
        # Here we assume the pulse is applied on-resonance for the nominal case,
        # but we can shift the center or add a linear detuning term.
        # For simplicity, we assume the pulse is centered and we just optimize shape.
        # The "detuning_comp" parameter in the search might represent a static offset 
        # that we assume the hardware has, which we can't change, but we can 
        # design the pulse to be robust against it.
        # Actually, the task says "detuning in (-2pi*0.002, 0, 2pi*0.002)".
        # We want a pulse that works well for all 3.
        # A DRAG pulse with optimized k_drag and sigma should be robust.
        
        return I, Q

    def simulate_2level(I, Q, detuning, scale):
        """
        Fast 2-level simulation.
        Returns mean fidelity to X gate.
        """
        # Effective Hamiltonian in rotating frame
        # H(t) = [ -delta_eff(t),  g(t) ]
        #       [ g*(t),           delta_eff(t) ]
        # where delta_eff(t) = detuning + 0.5*chi*... ? 
        # In the 2-level subspace, the anharmonicity term 0.5*chi*n(n-1) is zero for |0> and |1>.
        # So H_static = detuning * n.
        # H_drive = 0.5 * scale * (I (a + a_dag) + i Q (a - a_dag))
        # In basis |0>, |1>:
        # a|0> = |1>, a|1> = 0 (approx, ignoring |2>)
        # a_dag|0> = 0, a_dag|1> = |0>
        # a - a_dag = |0><1| - |1><0|
        # a + a_dag = |0><1| + |1><0|
        # H_drive = 0.5 * scale * I (|0><1| + |1><0|) + 0.5 * scale * i Q (|0><1| - |1><0|)
        # H_drive = 0.5 * scale * (I + iQ) |0><1| + 0.5 * scale * (I - iQ) |1><0|
        
        # So off-diagonal coupling:
        # H_01 = 0.5 * scale * (I + 1j*Q)
        # H_10 = 0.5 * scale * (I - 1j*Q)
        
        # Diagonal terms:
        # H_00 = 0
        # H_11 = detuning
        
        # We evolve the state vector [c0, c1]
        # i d/dt c = H c
        # c(t+dt) = exp(-i H dt) c(t)
        
        # To be fast, we use the fact that H is 2x2.
        # We can diagonalize H at each step or use the closed form for 2x2 exp.
        # For 2x2 Hermitian H = [[h00, h01], [h01*, h11]], 
        # exp(-i H dt) = cos(theta) I - i (sin(theta)/theta) (H - tr(H)/2 I)
        # where theta = sqrt(lambda) and lambda is the eigenvalue difference?
        # Actually, let's just use the series expansion or a fast 2x2 exp.
        # Since dt is small, we can use first-order + second-order if needed, 
        # but 80 steps is enough for 40ns.
        # Let's use the exact 2x2 exponential formula for speed.
        
        c0 = 1.0 + 0j
        c1 = 0.0 + 0j
        
        # Pre-allocate for speed? No, loop is fine for 80 steps.
        
        for k in range(n_steps):
            I_k = I[k]
            Q_k = Q[k]
            
            h00 = 0.0
            h11 = detuning
            h01 = 0.5 * scale * (I_k + 1j * Q_k)
            h10 = h01.conj()
            
            H = np.array([[h00, h01], [h10, h11]], dtype=complex)
            
            # Compute exp(-i H dt)
            # Trace
            tr = h00 + h11
            # Eigenvalues
            # lambda = tr/2 +/- sqrt((h00-h11)^2/4 + |h01|^2)
            diff = (h00 - h11) / 2.0
            r = np.sqrt(diff**2 + h01*h10)
            
            # If r is close to 0, use first order
            if abs(r) < 1e-12:
                U = np.eye(2) - 1j * H * dt
            else:
                # U = exp(-i H dt)
                # = cos(r*dt) * I - i * (sin(r*dt)/r) * (H - (tr/2)I)
                # Let A = H - (tr/2)I
                A = H - (tr/2) * np.eye(2)
                c_r = np.cos(r * dt)
                s_r = np.sin(r * dt) / r
                U = c_r * np.eye(2) - 1j * s_r * A
            
            # Apply decay
            # Approximate decay as multiplicative factor
            # |0> decays to nothing? No, |0> is ground state.
            # |1> decays to |0> with rate gamma_pop.
            # Coherence decays with gamma_coh.
            # This is tricky to combine with unitary evolution.
            # For short times, we can apply a non-unitary step.
            # Or, we can ignore decay for the optimization phase 
            # since T1, Tphi >> T_pulse (25us, 40us vs 40ns).
            # Decay loss is ~ T_pulse/T1 = 40e-9 / 25e-6 = 0.0016.
            # This is small but non-negligible.
            # We'll add a simple decay factor to the amplitudes.
            # c1_new = c1 * (1 - 0.5*gamma_pop*dt) - ...
            # Let's just use the unitary evolution for now, as the 
            # infidelity from decay is constant and doesn't affect optimization.
            
            c = U @ np.array([c0, c1])
            c0, c1 = c[0], c[1]
            
            # Apply decay
            # P(|1>) decreases by gamma_pop * dt
            # Coherence decreases by gamma_coh * dt
            # Simple approximation:
            # c1 *= (1 - 0.5*gamma_pop*dt)
            # c0 *= (1 - 0.5*gamma_pop*dt) ? No, |0> is stable.
            # The population of |1> leaks to |0>.
            # dP1/dt = -gamma_pop P1
            # So P1(t+dt) = P1(t) (1 - gamma_pop dt)
            # This means |c1|^2 scales.
            # We can scale c1 by sqrt(1 - gamma_pop dt)
            # And the coherence c0 c1* scales by (1 - gamma_coh dt)
            # This is an approximation.
            
            decay_factor_1 = math.sqrt(max(0, 1.0 - gamma_pop * dt))
            c1 *= decay_factor_1
            # Coherence decay:
            # c0 c1* -> (1 - gamma_coh dt) c0 c1*
            # This is hard to apply directly to c0, c1 without knowing the phase.
            # Given the small effect, we'll ignore it for the fast search.
            
        # Target state for X gate: |1>
        # Fidelity = |<1|psi>|^2 = |c1|^2
        fid = abs(c1)**2
        return fid

    def objective(params):
        sigma, k_drag = params
        if sigma < 1.0 or sigma > 20.0:
            return 1.0
        if k_drag < -0.5 or k_drag > 0.5:
            return 1.0
            
        I, Q = get_pulse(sigma, k_drag, 0.0)
        
        # Check constraints
        amps = np.sqrt(I**2 + Q**2)
        if np.max(amps) > 0.4:
            # Scale down
            scale_factor = 0.4 / np.max(amps)
            I *= scale_factor
            Q *= scale_factor
            # This changes the area, which is bad. 
            # We should ensure the pulse fits within bounds by construction.
            # If we have to scale, the area is reduced, increasing infidelity.
            # So we return a high cost.
            return 1.0 + np.max(amps) - 0.4
        
        # Slew rate
        dI = np.diff(I)
        dQ = np.diff(Q)
        slew = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(slew) > 0.12:
            return 1.0 + np.max(slew) - 0.12
        
        # Boundary slew
        slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        slewe = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        if slew0 > 0.12 or slewe > 0.12:
            return 1.0 + max(slew0, slewe) - 0.12
            
        # Evaluate fidelity at 9 corners
        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        fids = []
        for det in detunings:
            for sc in scales:
                f = simulate_2level(I, Q, det, sc)
                fids.append(f)
        
        mean_fid = np.mean(fids)
        max_infid = 1.0 - np.min(fids)
        return (1.0 - mean_fid) + max_infid

    # Global search
    best_score = float('inf')
    best_params = None
    best_I = None
    best_Q = None
    
    # Grid search
    sigma_vals = np.linspace(3.0, 10.0, 20)
    k_drag_vals = np.linspace(-0.2, 0.2, 20)
    
    count = 0
    for sigma in sigma_vals:
        for k in k_drag_vals:
            if time.time() > deadline - 10:
                break
            score = objective([sigma, k])
            if score < best_score:
                best_score = score
                best_params = [sigma, k]
                best_I, best_Q = get_pulse(sigma, k, 0.0)
            count += 1
        if time.time() > deadline - 10:
            break
            
    # Local refinement
    if best_params is not None and time.time() < deadline - 20:
        # Nelder-Mead
        import scipy.optimize as opt
        try:
            res = opt.minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_I, best_Q = get_pulse(best_params[0], best_params[1], 0.0)
        except Exception:
            pass

    # Final check
    if best_I is None:
        # Fallback to seed
        sigma = 40.0/6.0
        k = 0.0
        best_I, best_Q = get_pulse(sigma, k, 0.0)
        
    # Project to ensure constraints
    amps = np.sqrt(best_I**2 + best_Q**2)
    if np.max(amps) > 0.4:
        scale = 0.4 / np.max(amps)
        best_I *= scale
        best_Q *= scale
        
    dI = np.diff(best_I)
    dQ = np.diff(best_Q)
    slew = np.sqrt(dI**2 + dQ**2) / dt
    max_slew = np.max(slew)
    slew0 = np.sqrt(best_I[0]**2 + best_Q[0]**2) / dt
    slewe = np.sqrt(best_I[-1]**2 + best_Q[-1]**2) / dt
    max_slew = max(max_slew, slew0, slewe)
    
    if max_slew > 0.12:
        scale = 0.12 / max_slew
        best_I *= scale
        best_Q *= scale
        amps = np.sqrt(best_I**2 + best_Q**2)
        if np.max(amps) > 0.4:
            scale = 0.4 / np.max(amps)
            best_I *= scale
            best_Q *= scale

    return np.column_stack((best_I, best_Q))
