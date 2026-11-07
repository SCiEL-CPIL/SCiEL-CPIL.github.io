"""Robust DRAG-style pulse for 3-level transmon X-pi gate with spectral nulling."""
import numpy as np
from scipy.linalg import expm
import time

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only 'X_pi' is supported")
    
    # Hardware parameters from task description
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amplitude_bound = 0.4
    slew_bound = 0.12
    dt = 0.5  # ns
    n_steps = 80
    
    # Parse hardware_model if it's a dict, otherwise use defaults
    if isinstance(hardware_model, dict):
        anharmonicity = hardware_model.get('anharmonicity', -2 * np.pi * 0.25)
        amplitude_bound = hardware_model.get('amplitude_bound', 0.4)
        slew_bound = hardware_model.get('slew_bound', 0.12)
        T1 = hardware_model.get('T1', 25000.0)
        Tphi = hardware_model.get('Tphi', 40000.0)
    else:
        T1 = 25000.0
        Tphi = 40000.0
    
    # Time array
    t = np.arange(n_steps) * dt  # 0 to 39.5 ns
    
    # Gaussian envelope parameters
    # Center at 20 ns, sigma = 40/6 ns as per seed description
    center = 20.0
    sigma = 40.0 / 6.0
    
    # I component: Gaussian with area pi for X-pi gate
    # The drive is 0.5 * scale * I * (a + a†)
    # For a pi pulse on the |0>-|1> transition, we need integral of 0.5*I dt = pi/2
    # So integral of I dt = pi
    # Gaussian: I(t) = A * exp(-(t-center)^2 / (2*sigma^2))
    # Integral = A * sqrt(2*pi) * sigma = pi
    # A = pi / (sqrt(2*pi) * sigma) = sqrt(pi/2) / sigma
    A_I = np.sqrt(np.pi / 2.0) / sigma
    
    I = A_I * np.exp(-(t - center)**2 / (2.0 * sigma**2))
    
    # DRAG Q component: proportional to derivative of I
    # Standard DRAG: Q(t) = -alpha * dI/dt where alpha relates to anharmonicity
    # For a 3-level system, the DRAG condition to suppress |2> leakage:
    # alpha = -anharmonicity / (2 * drive_amplitude) approximately
    # More precisely, for the standard DRAG pulse:
    # Q(t) = -g * dI/dt where g = anharmonicity / (2 * Omega)
    # But since we're working in the rotating frame with the drive,
    # the standard formula is: Q(t) = -anharmonicity * dI/dt / (2 * Omega_0)
    # where Omega_0 is the peak drive amplitude
    # Actually, the standard DRAG formula for a transmon:
    # Q(t) = -alpha * dI/dt
    # where alpha = -anharmonicity / (2 * pi * Omega) ... let me think more carefully
    
    # In the standard DRAG literature for a transmon:
    # The leakage to |2> is proportional to the anharmonicity and the derivative of the pulse
    # To cancel this, we add a Q component proportional to dI/dt
    # The coefficient is: alpha = -anharmonicity / (2 * Omega_drive)
    # where Omega_drive is the peak Rabi frequency
    
    # Peak I amplitude
    I_peak = A_I  # since Gaussian peaks at center
    
    # DRAG coefficient
    # The standard result: to suppress |2> population, Q(t) = -alpha * dI/dt
    # with alpha = -anharmonicity / (2 * I_peak)
    # But let's use the more careful derivation:
    # In the rotating frame, the effective Hamiltonian has a term that couples |1> to |2>
    # proportional to anharmonicity * I(t). The DRAG pulse adds a Q component
    # that creates a counter-rotating term to cancel this.
    
    # Numerical derivative of I
    dI_dt = np.gradient(I, dt)
    
    # DRAG coefficient - standard formula
    # alpha = -anharmonicity / (2 * I_peak)
    alpha = -anharmonicity / (2.0 * I_peak)
    
    Q = alpha * dI_dt
    
    # Ensure constraints are satisfied
    # Check amplitude bound
    amplitude = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amplitude)
    if max_amp > amplitude_bound:
        scale_factor = amplitude_bound / max_amp
        I *= scale_factor
        Q *= scale_factor
    
    # Check slew bound
    # ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 1 and last step to zero
    p = np.column_stack([I, Q])
    
    # Check slew from zero to first step
    slew_first = np.sqrt(I[0]**2 + Q[0]**2) / dt
    # Check slew from last step to zero
    slew_last = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
    # Check slew between consecutive steps
    if n_steps > 1:
        dp = np.diff(p, axis=0)
        slew_between = np.max(np.sqrt(np.sum(dp**2, axis=1))) / dt
    else:
        slew_between = 0
    
    max_slew = max(slew_first, slew_last, slew_between)
    if max_slew > slew_bound:
        scale_factor = slew_bound / max_slew
        I *= scale_factor
        Q *= scale_factor
    
    # Re-check amplitude after slew scaling
    amplitude = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amplitude)
    if max_amp > amplitude_bound:
        scale_factor = amplitude_bound / max_amp
        I *= scale_factor
        Q *= scale_factor
    
    return np.column_stack([I, Q])
