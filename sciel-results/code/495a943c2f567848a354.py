"""
Strategy: Direct optimization of the full pulse vector (160 DOF) using Nelder-Mead.
Instead of a parametric DRAG shape, we optimize the raw I/Q amplitudes directly.
This allows for complex pulse shapes that may better robustly suppress leakage
and detuning errors across the 9 test cases.
"""
import numpy as np
import time
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_low, scale_high = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    n_steps = 80
    total_time = n_steps * dt
    
    # Time points
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Pre-compute matrices
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_mat = adag @ a
    eye3 = np.eye(3)
    
    H0_base = 0.5 * anharmonicity * n_mat @ (n_mat - eye3)
    
    # Test states
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)
    
    X_gate = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
    
    def simulate_unitary(pulse_arr, delta, scale):
        """
        Fast simulation of the unitary evolution.
        pulse_arr is shape (80, 2).
        """
        U_total = np.eye(3, dtype=complex)
        
        # Pre-compute constant parts of H_int structure?
        # H_int = 0.5 * scale * (I (a + adag) + i Q (a - adag))
        # a + adag = sigma_x + sqrt(2) sigma_y (in 0-1, 1-2 blocks)
        # Let's just build H_t each step, it's small (3x3)
        
        for i in range(n_steps):
            I_i = pulse_arr[i, 0]
            Q_i = pulse_arr[i, 1]
            
            # H_int = 0.5 * scale * (I (a + adag) + 1j * Q (a - adag))
            # a + adag = [[0, 1, 0], [1, 0, sqrt(2)], [0, sqrt(2), 0]]
            # a - adag = [[0, 1, 0], [-1, 0, sqrt(2)], [0, -sqrt(2), 0]]
            # i * (a - adag) = [[0, i, 0], [-i, 0, i*sqrt(2)], [0, -i*sqrt(2), 0]]
            
            # H_t = delta * n + H0_base + 0.5 * scale * (I * (a+adag) + iQ * (a-adag))
            
            # n = diag(0, 1, 2)
            # H0_base = 0.5 * an * n(n-1)
            # n=0: 0
            # n=1: 0
            # n=2: 0.5 * an * 2 = an
            # So H0_base = diag(0, 0, an)
            
            # H_int components:
            # H_01 = 0.5 * scale * I
            # H_10 = 0.5 * scale * I
            # H_12 = 0.5 * scale * I * sqrt(2)
            # H_21 = 0.5 * scale * I * sqrt(2)
            # Q terms:
            # i * (a - adag)
            # (a-adag)_01 = 1 -> i
            # (a-adag)_10 = -1 -> -i
            # (a-adag)_12 = sqrt(2) -> i*sqrt(2)
            # (a-adag)_21 = -sqrt(2) -> -i*sqrt(2)
            
            # H_t[0,1] = 0.5 * scale * I + 0.5 * scale * i * Q * 1 = 0.5 * scale * (I + iQ)
            # H_t[1,0] = 0.5 * scale * I + 0.5 * scale * (-i) * Q * 1 = 0.5 * scale * (I - iQ)
            # H_t[1,2] = 0.5 * scale * I * sqrt(2) + 0.5 * scale * i * Q * sqrt(2) = 0.5 * scale * sqrt(2) * (I + iQ)
            # H_t[2,1] = 0.5 * scale * I * sqrt(2) + 0.5 * scale * (-i) * Q * sqrt(2) = 0.5 * scale * sqrt(2) * (I - iQ)
            
            # Diagonals:
            # H_t[0,0] = 0
            # H_t[1,1] = delta
            # H_t[2,2] = 2*delta + anharmonicity
            
            H_t = np.zeros((3, 3), dtype=complex)
            H_t[0, 0] = 0.0
            H_t[1, 1] = delta
            H_t[2, 2] = 2.0 * delta + anharmonicity
            
            H_t[0, 1] = 0.5 * scale * (I_i + 1j * Q_i)
            H_t[1, 0] = 0.5 * scale * (I_i - 1j * Q_i)
            H_t[1, 2] = 0.5 * scale * np.sqrt(2) * (I_i + 1j * Q_i)
            H_t[2, 1] = 0.5 * scale * np.sqrt(2) * (I_i - 1j * Q_i)
            
            # Exponentiate
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
            
        return U_total

    def compute_fidelity(U):
        fidelities = []
        for psi in states:
            target = X_gate @ psi
            out = U @ psi
            fid = np.abs(np.vdot(target, out))**2
            fidelities.append(fid)
        return np.mean(fidelities)

    def compute_score(pulse_arr):
        deltas = [-detuning_bound, 0, detuning_bound]
        scales = [scale_low, 1.0, scale_high]
        
        infidelities = []
        for d in deltas:
            for s in scales:
                U = simulate_unitary(pulse_arr, d, s)
                fid = compute_fidelity(U)
                infidelities.append(1 - fid)
        
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def enforce_constraints(p):
        """
        p is flat array of length 160 (I[0..79], Q[0..79]) or (80, 2).
        We will work with (80, 2) for clarity in projection.
        """
        if p.ndim == 1:
            p = p.reshape(80, 2)
            
        # Amplitude bound
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            p = p * (amp_bound / max_amp)
        
        # Slew rate bound
        # ||p[i+1] - p[i]|| / dt <= 0.12
        # This is a global constraint on the vector differences.
        # Simple projection: scale down if violated.
        max_slew = 0
        p_prev = np.zeros(2)
        for i in range(80):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > max_slew:
                max_slew = dp
            p_prev = p[i].copy()
        
        dp_last = np.linalg.norm(p[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last
            
        if max_slew > slew_bound:
            p = p * (slew_bound / max_slew)
            
        return p

    def objective_flat(x_flat):
        """
        Objective function for scipy.optimize.minimize.
        x_flat is length 160.
        """
        p = x_flat.reshape(80, 2)
        p = enforce_constraints(p)
        score = compute_score(p)
        return score

    # Initial guess: A reasonable DRAG pulse
    # Gaussian sigma = 40/6 ~ 6.67
    sigma = 6.67
    main_gauss = np.exp(-t_centered**2 / (2 * sigma**2))
    area_main = np.sum(main_gauss) * dt
    main_gauss = main_gauss * (np.pi / area_main)
    
    # DRAG component: derivative
    d_drag = np.gradient(main_gauss, dt)
    # Typical DRAG alpha is around 0.1-0.3 for this anharmonicity?
    # Let's start with a small Q component
    alpha_drag = 0.2
    Q_init = alpha_drag * d_drag
    
    p_init = np.column_stack([main_gauss, Q_init])
    p_init = enforce_constraints(p_init)
    x0 = p_init.flatten()
    
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_x = x0.copy()
    best_score = objective_flat(x0)
    
    # Nelder-Mead on 160 dimensions is expensive.
    # We need to be careful with the time budget.
    # Each objective call is 9 simulations of 80 steps.
    # 9 * 80 = 720 eigh calls.
    # eigh on 3x3 is very fast, but Python loop overhead is significant.
    # Let's estimate time per objective call.
    
    t0 = time.time()
    _ = objective_flat(x0)
    t1 = time.time()
    time_per_call = t1 - t0
    
    # Nelder-Mead with N=160 needs roughly 2*N initial simplex points.
    # That's 320 calls.
    # If time_per_call is 0.1s, 320 * 0.1 = 32s just for initial simplex.
    # Then iterations...
    # We might not finish a full Nelder-Mead in 300s for 160 dims.
    
    # Alternative: Use a coarser search or reduce dimensions?
    # Or use a different optimizer?
    # scipy.optimize.minimize with method='Powell' might be better for high dims?
    # Or 'Nelder-Mead' with aggressive maxiter?
    
    # Let's try Nelder-Mead but with a tight time check.
    # If it's too slow, we fall back to the parametric result.
    
    try:
        # We can't easily pass a time limit to minimize.
        # We will run a limited number of iterations.
        # For N=160, maxiter=100 is likely not enough for convergence, but might improve.
        # Let's try to run it and interrupt if time runs out.
        
        # Actually, Nelder-Mead in 160D is very slow to converge.
        # A better strategy for high dims might be to optimize a lower-dimensional
        # representation that is more flexible than 4 params but less than 160.
        # E.g., a spline with 10-20 control points?
        
        # Let's try 20 control points for I and 20 for Q, interpolated to 80.
        # Total dims: 40.
        # This is a "materially different" strategy: Spline-based pulse shaping.
        
        def spline_pulse(params):
            """
            params: [I_ctrl_0..19, Q_ctrl_0..19]
            """
            I_ctrl = params[:20]
            Q_ctrl = params[20:]
            
            # Control points at uniform intervals
            t_ctrl = np.linspace(-total_time/2, total_time/2, 20)
            t_eval = t_centered
            
            # Cubic spline interpolation
            # Need to handle boundaries.
            # Use np.interp for linear? Or scipy.interpolate.CubicSpline?
            # Linear is safer for constraints but less flexible.
            # Let's use linear interpolation for robustness and speed.
            I_vals = np.interp(t_eval, t_ctrl, I_ctrl)
            Q_vals = np.interp(t_eval, t_ctrl, Q_ctrl)
            
            p = np.column_stack([I_vals, Q_vals])
            return enforce_constraints(p)

        def objective_spline(x_flat):
            p = spline_pulse(x_flat)
            return compute_score(p)
            
        # Initial guess for spline: Sample the initial DRAG pulse at 20 points
        t_ctrl = np.linspace(-total_time/2, total_time/2, 20)
        I_init_ctrl = np.interp(t_ctrl, t_centered, p_init[:, 0])
        Q_init_ctrl = np.interp(t_ctrl, t_centered, p_init[:, 1])
        x0_spline = np.concatenate([I_init_ctrl, Q_init_ctrl])
        
        best_x_spline = x0_spline.copy()
        best_score_spline = objective_spline(x0_spline)
        
        if best_score_spline < best_score:
            best_score = best_score_spline
            best_x = x0_spline # Keep track of type?
            # We need to distinguish which space we are in.
            # Let's just run the spline optimization and compare final scores.
            
        t0 = time.time()
        _ = objective_spline(x0_spline)
        t1 = time.time()
        time_per_call_spline = t1 - t0
        
        # Run Nelder-Mead on 40 dimensions
        # This is much more feasible.
        res = minimize(objective_spline, x0_spline, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7,
                                'adaptive': True})
        
        if time.time() < deadline:
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
                # Mark that best_x is in spline space
                # We need to reconstruct the pulse from best_x at the end.
                # Let's store the pulse directly to avoid ambiguity.
                best_pulse = spline_pulse(best_x)
            else:
                best_pulse = spline_pulse(best_x_spline)
        else:
            best_pulse = spline_pulse(best_x_spline)
            
    except Exception as e:
        print(f"Optimization error: {e}")
        best_pulse = p_init
        
    # Final validity check
    amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amp) > amp_bound:
        best_pulse *= amp_bound / np.max(amp)
    
    max_slew = 0
    p_prev = np.zeros(2)
    for i in range(80):
        dp = np.linalg.norm(best_pulse[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = best_pulse[i].copy()
    
    dp_last = np.linalg.norm(best_pulse[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last
        
    if max_slew > slew_bound:
        best_pulse *= slew_bound / max_slew
        
    return best_pulse
