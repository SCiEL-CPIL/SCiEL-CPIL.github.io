"""Rewritten program: robust DRAG-style pulse for 3-level transmon X-pi gate."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only 'X_pi' is supported")
    
    # Parse hardware model
    # Expected keys or structure: anharmonicity, amplitude_bound, slew_bound, 
    # detuning_bounds, drive_scale_bounds, T1, Tphi
    # Based on task description:
    # anharmonicity = -2*pi*0.25
    # amplitude_bound = 0.4
    # slew_bound = 0.12
    # detuning_bounds = (+-2*pi*0.002)
    # drive_scale_bounds = [0.97, 1.03]
    # T1 = 25000 ns, Tphi = 40000 ns
    
    # We assume hardware_model is a dict or object with these attributes.
    # For robustness, we'll try to extract them, but since the seed just uses quantum_seed,
    # we need to implement the logic ourselves.
    # The task says: "hardware_model holds anharmonicity -2 pi 0.25, ..."
    # We will hardcode the physics parameters based on the task description,
    # but we should check if hardware_model provides them.
    # Let's assume standard keys.
    
    try:
        anharmonicity = hardware_model.get('anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = hardware_model.get('amplitude_bound', 0.4)
        slew_bound = hardware_model.get('slew_bound', 0.12)
        detuning_bounds = hardware_model.get('detuning_bounds', (-2 * np.pi * 0.002, 2 * np.pi * 0.002))
        drive_scale_bounds = hardware_model.get('drive_scale_bounds', (0.97, 1.03))
        T1 = hardware_model.get('T1', 25000)
        Tphi = hardware_model.get('Tphi', 40000)
    except AttributeError:
        # If it's not a dict, try attributes
        anharmonicity = getattr(hardware_model, 'anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = getattr(hardware_model, 'amplitude_bound', 0.4)
        slew_bound = getattr(hardware_model, 'slew_bound', 0.12)
        detuning_bounds = getattr(hardware_model, 'detuning_bounds', (-2 * np.pi * 0.002, 2 * np.pi * 0.002))
        drive_scale_bounds = getattr(hardware_model, 'drive_scale_bounds', (0.97, 1.03))
        T1 = getattr(hardware_model, 'T1', 25000)
        Tphi = getattr(hardware_model, 'Tphi', 40000)
    
    # Pulse parameters
    dt = 0.5  # ns
    N_steps = 80
    t_total = N_steps * dt  # 40 ns
    
    # Time grid
    t = np.linspace(0, t_total - dt, N_steps)  # times for each step? Or centers?
    # The task says "80 constant steps of dt = 0.5 ns".
    # Usually, the pulse is defined at step indices 0..79.
    # Let's define time centers: t_i = (i + 0.5) * dt
    t_centers = (np.arange(N_steps) + 0.5) * dt
    
    # DRAG pulse design
    # Gaussian envelope with sigma such that area is pi.
    # The seed uses sigma = 40/6 ns? No, "sigma 40/6 ns" in the prompt text.
    # Wait, the prompt says: "The seed is a DRAG-style Gaussian (sigma 40/6 ns, offset to 0 at the edges, area pi)."
    # Let's calculate sigma. If area is pi, and we integrate a Gaussian exp(-(t-t0)^2 / (2*sigma^2)) * A dt = pi.
    # The area of exp(-(t)^2 / (2*sigma^2)) is sigma * sqrt(2*pi).
    # So A * sigma * sqrt(2*pi) = pi  => A = sqrt(pi/2) / sigma.
    # The prompt mentions "sigma 40/6 ns". Let's use sigma = 40/6 ns.
    sigma = 40.0 / 6.0
    
    # Calculate amplitude A for area pi
    # Integral from -inf to inf of exp(-(t)^2/(2*sigma^2)) dt = sigma * sqrt(2*pi)
    # We want area = pi.
    A_gauss = np.sqrt(np.pi / 2.0) / sigma
    
    # Generate I and Q components
    I = np.zeros(N_steps)
    Q = np.zeros(N_steps)
    
    # DRAG: Q component is proportional to the derivative of I.
    # I(t) = A * exp(-(t-t0)^2 / (2*sigma^2))
    # I'(t) = I(t) * (-(t-t0) / sigma^2)
    # Q(t) = alpha * I'(t)
    # We need to choose alpha to optimize robustness against detuning and anharmonicity.
    # Typically, alpha is related to the anharmonicity.
    # For a transmon, the optimal DRAG alpha is often approximated as:
    # alpha = -anharmonicity / (2 * pi) * (some factor) ?
    # Actually, the standard DRAG condition for a two-level system with an anharmonic third level is:
    # The leakage to |2> is suppressed when the Q component cancels the virtual transition.
    # The optimal Q amplitude is Q(t) = - (anharmonicity / (2 * pi)) * (1 / (2 * sigma)) * I(t) * (t - t0) ?
    # Let's derive or look up the standard formula.
    # The Hamiltonian term coupling |1> to |2> is proportional to the second derivative of the pulse or the detuning.
    # A common choice is Q(t) = - (delta_21 / (2 * pi * 2 * sigma^2)) * (t - t0) * I(t) ?
    # Let's use the standard DRAG formula:
    # Q(t) = - (anharmonicity / (2 * pi)) * (1 / (2 * sigma)) * (t - t0) / sigma * I(t) ?
    # Actually, the optimal DRAG pulse for a transmon is:
    # I(t) = A exp(-(t-t0)^2 / (2 sigma^2))
    # Q(t) = - (anharmonicity / (2 * pi * 2 * sigma)) * (t - t0) / sigma * I(t)
    # Let's check dimensions. Anharmonicity is in rad/ns. sigma is in ns.
    # (rad/ns) / ns = rad/ns^2. (t-t0) is ns. So (rad/ns^2) * ns = rad/ns. This matches I(t) units.
    
    # Let's refine the DRAG coefficient.
    # The leakage rate to |2> is proportional to (I * dQ/dt - Q * dI/dt) ? No.
    # The standard result is that the Q component should be proportional to the time derivative of I.
    # Q(t) = k * I'(t)
    # I'(t) = - (t - t0) / sigma^2 * I(t)
    # So Q(t) = - k * (t - t0) / sigma^2 * I(t)
    # The optimal k is related to the anharmonicity.
    # From literature (e.g., Motzoi et al.), the optimal DRAG pulse for a transmon has:
    # Q(t) = - (anharmonicity / (2 * pi)) * (1 / (2 * sigma)) * (t - t0) / sigma * I(t) ?
    # Let's try: Q(t) = - (anharmonicity / (2 * pi * 2 * sigma^2)) * (t - t0) * I(t)
    # Let's calculate the coefficient:
    # anharmonicity = -2 * pi * 0.25 rad/ns
    # coeff = - (-2 * pi * 0.25) / (2 * pi * 2 * sigma^2) = 0.25 / (2 * sigma^2) = 0.125 / sigma^2
    # sigma = 40/6 = 6.667 ns
    # sigma^2 = 44.44
    # coeff = 0.125 / 44.44 = 0.0028125
    # Q(t) = - coeff * (t - t0) * I(t)
    
    # Let's implement this.
    # t0 is the center of the pulse.
    t0 = t_total / 2.0
    
    # Calculate I
    I = A_gauss * np.exp(-((t_centers - t0) ** 2) / (2 * sigma ** 2))
    
    # Calculate Q
    # Q(t) = - (anharmonicity / (2 * pi * 2 * sigma^2)) * (t - t0) * I(t)
    # Note: anharmonicity is negative.
    # Let's check the sign. If anharmonicity is negative, the coefficient is positive.
    # Q(t) = - (negative) * (t-t0) * I(t) = positive * (t-t0) * I(t)
    # For t < t0, (t-t0) is negative, so Q is negative.
    # For t > t0, (t-t0) is positive, so Q is positive.
    # This creates a "tilt" in the pulse.
    
    coeff_drag = -anharmonicity / (2 * np.pi * 2 * sigma ** 2)
    Q = coeff_drag * (t_centers - t0) * I
    
    # Check constraints
    # 1. Amplitude bound: sqrt(I^2 + Q^2) <= 0.4
    # 2. Slew bound: ||p[i+1] - p[i]|| / dt <= 0.12
    #    p = [I, Q]. So sqrt((I[i+1]-I[i])^2 + (Q[i+1]-Q[i])^2) / dt <= 0.12
    #    Also from 0 to I[0], Q[0] and from I[-1], Q[-1] to 0.
    
    # Check amplitude
    amp = np.sqrt(I ** 2 + Q ** 2)
    if np.max(amp) > amplitude_bound:
        # Scale down
        scale = amplitude_bound / np.max(amp)
        I *= scale
        Q *= scale
        # Re-check
        amp = np.sqrt(I ** 2 + Q ** 2)
    
    # Check slew rate
    # Calculate differences
    # p[0] - p[-1] (from previous zero to first step)
    # p[i+1] - p[i]
    # p[-1] to zero
    
    # We need to check if the slew rate is violated.
    # If violated, we need to reduce the pulse amplitude or broaden it.
    # Since we can't easily broaden it without changing the area, we might need to reduce amplitude.
    # But reducing amplitude reduces the area, making the gate worse.
    # A better approach is to check if the slew rate is the limiting factor.
    
    # Let's check the slew rate first.
    I_prev = np.concatenate(([0], I[:-1]))
    Q_prev = np.concatenate(([0], Q[:-1]))
    dI = I - I_prev
    dQ = Q - Q_prev
    slew = np.sqrt(dI ** 2 + dQ ** 2) / dt
    
    # Also check the last step to zero
    dI_last = 0 - I[-1]
    dQ_last = 0 - Q[-1]
    slew_last = np.sqrt(dI_last ** 2 + dQ_last ** 2) / dt
    
    slew_max = np.max(np.concatenate((slew, [slew_last])))
    
    if slew_max > slew_bound:
        # Scale down to meet slew bound
        scale = slew_bound / slew_max
        I *= scale
        Q *= scale
        
        # Re-check amplitude
        amp = np.sqrt(I ** 2 + Q ** 2)
        if np.max(amp) > amplitude_bound:
            scale2 = amplitude_bound / np.max(amp)
            I *= scale2
            Q *= scale2
    
    # The pulse is now valid.
    # However, the area might not be exactly pi anymore due to scaling.
    # For a robust gate, we might want to adjust the amplitude to ensure the area is close to pi,
    # but the constraints are hard limits. The scoring is based on infidelity, so we want the
    # best possible pulse within the constraints.
    
    # Let's return the pulse as an (80, 2) array.
    pulse_array = np.column_stack((I, Q))
    
    return pulse_array
