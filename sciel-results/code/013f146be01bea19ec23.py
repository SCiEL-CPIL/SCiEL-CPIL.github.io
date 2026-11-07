"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse using spectral-nulling parameterization
    with two Gaussian envelopes optimized via Nelder-Mead.
    
    Returns (80, 2) array of I and Q amplitudes (rad/ns).
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    dt = 0.5
    n_steps = 80
    T = n_steps * dt  # 40 ns
    t = np.arange(n_steps) * dt + dt / 2.0  # step centers
    
    amp_bound = 0.4
    slew_bound = 0.12
    
    # --- Fast 2-level fidelity evaluation (effective Hamiltonian) ---
    def compute_fidelity_fast(I, Q, detuning, scale):
        """
        Compute mean fidelity to X gate over 5 states using 2-level effective model.
        The 2-level model captures the dominant physics: detuning causes rotation
        angle error and phase error. DRAG (Q component) suppresses |2> leakage.
        """
        # Effective Rabi frequency and phase at each step
        omega = scale * np.sqrt(I**2 + Q**2)  # rad/ns
        phase = np.arctan2(Q, I)
        
        # Integrated rotation angle
        angle = np.sum(omega) * dt
        
        # Phase error from detuning: the effective Hamiltonian has
        # H = (detuning/2)*sigma_z + (omega/2)*(cos(phi)*sigma_x + sin(phi)*sigma_y)
        # For small detuning, the rotation axis tilts. The net effect on fidelity
        # for a pi-pulse: infidelity ~ (angle_error)^2 + (phase_error)^2 + leakage^2
        #
        # More precisely, for a 2-level system with H = (d/2)s_z + (g/2)(c s_x + s s_y):
        # The rotation is about axis (c*s, c*c, d) with angle theta = sqrt(g^2+d^2)*T_eff
        # For a pi pulse: we need theta = pi and axis in xy-plane.
        # Deviation: if d != 0, axis tilts by alpha = arctan(d/g_avg)
        # and angle becomes sqrt(g^2 + d^2) * T instead of g * T.
        
        g_avg = np.mean(omega)  # average Rabi frequency
        if g_avg < 1e-12:
            return 0.0
        
        # Net rotation angle
        theta = np.sqrt(g_avg**2 + detuning**2) * T
        # Target angle is pi, so angle error:
        d_angle = theta - np.pi
        
        # Axis tilt: for a pi rotation about a tilted axis, the infidelity
        # for a state in the equator is ~ sin^2(alpha/2) where alpha is the
        # angle between the actual axis and the target axis.
        # alpha = arctan(detuning / g_avg) for the tilt from xy-plane
        tilt = np.arctan2(detuning, g_avg)
        
        # For X gate (rotation by pi about x-axis), the infidelity for state |psi> is:
        # 1 - |<psi| exp(-i*pi/2 * n_hat.sigma) |psi>|^2
        # where n_hat = (cos(phi)*cos(tilt), sin(phi)*cos(tilt), sin(tilt))
        # For phi=0 (pure I): n_hat = (cos(tilt), 0, sin(tilt))
        # The X gate maps |0> -> |1>, |1> -> -|0>
        # With tilt, the rotation is about a different axis.
        
        # Simplified: for small tilt, infidelity ~ (d_angle)^2/4 + (tilt^2 * sin^2(theta/2))
        # For theta ~ pi: sin^2(pi/2) = 1, so infidelity ~ d_angle^2/4 + tilt^2
        
        # But we also need to account for the phase (Q component shifts the rotation axis)
        # The Q component rotates the effective axis in the xy-plane.
        # Average phase:
        phi_avg = np.mean(phase)
        # For X gate, we want phi_avg = 0. Phase error:
        d_phi = phi_avg  # deviation from 0
        
        # Combined infidelity estimate (2-level):
        # The rotation is: R = exp(-i * theta/2 * (cos(phi_avg)*cos(tilt)*sx + sin(phi_avg)*cos(tilt)*sy + sin(tilt)*sz))
        # Target: X = exp(-i*pi/2 * sx)
        # Fidelity for state |psi>: |<psi| R |psi>|^2
        # For the 5 test states, the mean fidelity can be computed analytically.
        
        # Let's use the exact formula for a rotation by angle theta about axis n:
        # R = cos(theta/2)*I - i*sin(theta/2)*(n.x*sigma_x + n.y*sigma_y + n.z*sigma_z)
        # Fidelity of state |psi> under R: |<psi|psi> - i*sin(theta/2)*<psi|n.sigma|psi>|^2
        # For |0> = (1,0): <0|n.sigma|0> = n.z, so F = |1 - i*sin(theta/2)*n.z|^2 = 1 + sin^2(theta/2)*n.z^2 - 2*sin(theta/2)*n.z*Im(...)
        # Actually: <0|R|0> = cos(theta/2) - i*sin(theta/2)*n.z
        # |<0|R|0>|^2 = cos^2(theta/2) + sin^2(theta/2)*n.z^2 = 1 - sin^2(theta/2)*(1-n.z^2)
        # Hmm, let me just compute it numerically for the 5 states.
        
        # Build the rotation matrix
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        nx = np.cos(phi_avg) * np.cos(tilt)
        ny = np.sin(phi_avg) * np.cos(tilt)
        nz = np.sin(tilt)
        
        # R = c*I - i*s*(nx*sigma_x + ny*sigma_y + nz*sigma_z)
        # sigma_x = [[0,1],[1,0]], sigma_y = [[0,-i],[i,0]], sigma_z = [[1,0],[0,-1]]
        # n.sigma = [[nz, nx - i*ny], [nx + i*ny, -nz]]
        nsigma = np.array([[nz, nx - 1j*ny],
                           [nx + 1j*ny, -nz]])
        R = c * np.eye(2) - 1j * s * nsigma
        
        # Test states
        psi0 = np.array([1.0, 0.0])
        psi1 = np.array([0.0, 1.0])
        psi_p = np.array([1.0, 1.0]) / np.sqrt(2)
        psi_m = np.array([1.0, -1.0]) / np.sqrt(2)
        psi_ip = np.array([1.0, 1j]) / np.sqrt(2)
        psi_im = np.array([1.0, -1j]) / np.sqrt(2)
        
        states = [psi0, psi1, psi_p, psi_m, psi_ip, psi_im]
        # X gate: maps |0>->|1>, |1>->-|0>
        # X = [[0,1],[1,0]]
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        fidelities = []
        for psi in states:
            target = X @ psi
            actual = R @ psi
            fid = np.abs(np.vdot(target, actual))**2
            fidelities.append(fid)
        
        mean_fid = np.mean(fidelities)
        return mean_fid
    
    def leakage_penalty(I, Q, detuning, scale):
        """
        Estimate |2> leakage from the Q (DRAG) component.
        The leakage to |2> is proportional to the overlap of the pulse with
        the |1><-|2> transition. For a DRAG pulse, the Q component is
        designed to cancel this. A simple model: leakage ~ integral of
        I(t) * (I(t) + beta * dI/dt) * coupling / anharmonicity
        We use a proxy: the variance of the Q component relative to I.
        """
        # The |2> leakage for a 3-level system driven on |0><->|1> is
        # approximately: L ~ (g^2 / (2*|chi|)) * integral of |envelope(t)|^2 dt * correction
        # where g is the drive amplitude and chi is the anharmonicity.
        # The DRAG correction (Q ~ beta * dI/dt) cancels the first-order term.
        # Residual leakage ~ (1 - beta*beta_opt)^2 * base_leakage
        # For scoring, we add a small penalty proportional to the Q energy
        # that isn't needed for DRAG.
        
        # Simple proxy: leakage ~ (max_I / |chi|)^2 * (1 + |detuning/chi|)
        # This is very rough but captures the right scaling.
        chi = 2 * np.pi * 0.25  # |anharmonicity| in rad/ns
        max_I = np.max(np.abs(I))
        base_leak = (max_I / chi)**2
        # Detuning increases leakage:
        leak = base_leak * (1.0 + (detuning / chi)**2)
        # DRAG reduces it: if Q is proportional to dI/dt with optimal beta, leak ~ 0
        # If Q is zero (no DRAG), leak is maximal.
        # Scale by how much Q is present:
        I_energy = np.sum(I**2) * dt
        Q_energy = np.sum(Q**2) * dt
        if I_energy > 1e-12:
            drag_ratio = Q_energy / I_energy
            # Optimal drag_ratio for Gaussian is ~ (1/(4*sigma^2 * chi^2)) * something
            # For our purposes, a moderate Q reduces leakage:
            leak_reduction = np.tanh(5.0 * drag_ratio)  # saturates at 1
        else:
            leak_reduction = 0.0
        return leak * (1.0 - leak_reduction)
    
    def objective(params):
        """
        params: [A1, w1, t1, A2, w2, t2]
        I(t) = A1 * exp(-(t-t1)^2 / (2*w1^2))
        Q(t) = A2 * exp(-(t-t2)^2 / (2*w2^2))
        """
        A1, w1, t1, A2, w2, t2 = params
        
        # Clamp to valid ranges
        w1 = max(2.0, min(20.0, w1))
        w2 = max(2.0, min(20.0, w2))
        t1 = max(0.0, min(T, t1))
        t2 = max(0.0, min(T, t2))
        A1 = max(-0.4, min(0.4, A1))
        A2 = max(-0.4, min(0.4, A2))
        
        I = A1 * np.exp(-(t - t1)**2 / (2.0 * w1**2))
        Q = A2 * np.exp(-(t - t2)**2 / (2.0 * w2**2))
        
        p = np.column_stack([I, Q])
        
        # Check amplitude bound
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            return 1.0  # infeasible
        
        # Check slew rate
        p_prev = np.zeros(2)
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > slew_bound:
                return 1.0
            p_prev = p[i].copy()
        if np.linalg.norm(p[-1]) / dt > slew_bound:
            return 1.0
        
        # Area check: need approximately pi
        area = np.sum(I) * dt
        area_error = abs(area - np.pi)
        if area_error > 0.5:
            return 0.5 + area_error  # heavy penalty for wrong area
        
        # Compute fidelity across detuning and scale
        detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
        scales = [0.97, 1.0, 1.03]
        
        total_infid = 0.0
        max_infid = 0.0
        count = 0
        for d in detunings:
            for s in scales:
                fid = compute_fidelity_fast(I, Q, d, s)
                leak = leakage_penalty(I, Q, d, s)
                infid = 1.0 - fid + 10.0 * leak  # weight leakage
                total_infid += infid
                max_infid = max(max_infid, infid)
                count += 1
        
        mean_infid = total_infid / count
        score = mean_infid + max_infid
        
        # Add area penalty
        score += 0.1 * area_error**2
        
        return score
    
    # --- Optimization ---
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    # Initial guess: reasonable DRAG pulse
    # Main Gaussian centered at T/2, width ~ 6ns (sigma)
    # DRAG component: derivative of main, scaled
    x0 = np.array([
        0.15,   # A1 (main amplitude) - need area ~ pi, so A1*w1*sqrt(2*pi) ~ pi
        6.0,    # w1 (main width)
        20.0,   # t1 (main center)
        0.03,   # A2 (DRAG amplitude)
        6.0,    # w2 (DRAG width)
        20.0,   # t2 (DRAG center)
    ])
    
    # Adjust A1 for area: A1 * w1 * sqrt(2*pi) = pi => A1 = 1/(w1*sqrt(2*pi))
    x0[0] = 1.0 / (x0[1] * np.sqrt(2.0 * np.pi))
    
    best_x = x0.copy()
    best_score = objective(x0)
    
    # Nelder-Mead optimization
    from scipy.optimize import minimize
    
    # Use multiple restarts with different initial guesses
    initial_guesses = [
        x0,
        np.array([0.2, 5.0, 20.0, 0.04, 5.0, 20.0]),
        np.array([0.12, 8.0, 20.0, 0.02, 8.0, 20.0]),
        np.array([0.18, 4.5, 20.0, 0.05, 4.5, 20.0]),
        np.array([0.15, 6.0, 18.0, 0.03, 6.0, 22.0]),  # shifted DRAG
        np.array([0.15, 6.0, 22.0, 0.03, 6.0, 18.0]),  # shifted DRAG other way
    ]
    
    for i, x_init in enumerate(initial_guesses):
        if time.time() > deadline - 30.0:
            break
        
        # Adjust A1 for area
        w1_init = max(2.0, x_init[1])
        x_init = x_init.copy()
        x_init[0] = 1.0 / (w1_init * np.sqrt(2.0 * np.pi))
        
        try:
            result = minimize(
                objective, x_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True,
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Local refinement with tighter tolerance
    if time.time() < deadline - 20.0:
        try:
            result = minimize(
                objective, best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Construct final pulse
    A1, w1, t1, A2, w2, t2 = best_x
    w1 = max(2.0, min(20.0, w1))
    w2 = max(2.0, min(20.0, w2))
    t1 = max(0.0, min(T, t1))
    t2 = max(0.0, min(T, t2))
    
    I = A1 * np.exp(-(t - t1)**2 / (2.0 * w1**2))
    Q = A2 * np.exp(-(t - t2)**2 / (2.0 * w2**2))
    
    p = np.column_stack([I, Q])
    
    # Enforce area = pi (scale I if needed)
    area = np.sum(I) * dt
    if area > 1e-12:
        I = I * (np.pi / area)
        p = np.column_stack([I, Q])
    
    # Enforce constraints with scaling
    # Amplitude bound
    amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        p *= scale
    
    # Slew rate bound
    p_prev = np.zeros(2)
    max_slew = 0.0
    for i in range(n_steps):
        dp = np.linalg.norm(p[i] - p_prev) / dt
        max_slew = max(max_slew, dp)
        p_prev = p[i].copy()
    max_slew = max(max_slew, np.linalg.norm(p[-1]) / dt)
    
    if max_slew > slew_bound:
        p *= slew_bound / max_slew
    
    # Final validity check
    amp_final = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    if np.max(amp_final) > amp_bound * (1 + 1e-9):
        # Fallback: use a safe default
        t_centered = t - T / 2.0
        sigma = 8.0
        I = np.exp(-t_centered**2 / (2.0 * sigma**2))
        area = np.sum(I) * dt
        I = I * (np.pi / area)
        Q = np.zeros_like(I)
        p = np.column_stack([I, Q])
        max_slew = np.max(np.linalg.norm(np.diff(p, axis=0), axis=1) / dt)
        max_slew = max(max_slew, np.linalg.norm(p[0]) / dt, np.linalg.norm(p[-1]) / dt)
        if max_slew > slew_bound:
            p *= slew_bound / max_slew
    
    return p
# EVOLVE-BLOCK-END
