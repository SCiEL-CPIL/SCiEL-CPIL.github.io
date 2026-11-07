def refine_pulse(pulse_init):
    # Use L-BFGS-B on 160 params
    # Objective: score_pulse(pulse)
    # Gradient: Finite difference
    
    def obj(pulse_flat):
        p = pulse_flat.reshape(80, 2)
        p = project_pulse(p)
        return score_pulse(p)
        
    # L-BFGS-B with finite diff
    # This will be slower, but we have time.
    # However, projection makes it non-smooth.
    # Let's just use the basis expansion result. It's robust.
    
    return pulse_init
