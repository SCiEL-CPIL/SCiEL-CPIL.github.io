def generate_drag_pulse():
    t = np.linspace(0, DURATION, N_STEPS, endpoint=False) + DT/2
    sigma = DURATION / 6.0
    g = np.exp(-t**2 / (2 * sigma**2))
    g_prime = -t / sigma**2 * g
    
    area = np.sum(g) * DT
    scale_g = np.pi / area
    
    alpha_drag = 0.15 # Tune this if needed
    
    I = scale_g * g
    Q = scale_g * alpha_drag * g_prime
    
    # Check amplitude
    amps = np.sqrt(I**2 + Q**2)
    if np.max(amps) > AMP_BOUND:
        factor = AMP_BOUND / np.max(amps)
        I *= factor
        Q *= factor
        
    pulse = np.column_stack((I, Q))
    return pulse
