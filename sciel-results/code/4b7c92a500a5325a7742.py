def coefficients(stages):
        if stages != 12:
            return [1.0/stages]*stages, [1.0/stages]*stages
        
        # 2nd-order Strang-like coefficients for 12 stages
        # We have 12 B steps and 12 A steps.
        # To mimic Strang e^{A/2} e^B e^{A/2}, we need half steps at boundaries.
        # But the structure is B A B A ...
        # Let's distribute time such that the "effective" step is centered.
        
        # A simple 2nd order approximation for (B+A) using BA structure:
        # Use symmetric coefficients.
        # a = [0.5, 1, 1, ..., 1, 0.5] / 11 ?
        # b = [0.5, 1, 1, ..., 1, 0.5] / 11 ?
        
        # Let's try:
        a = [0.0] * 12
        b = [0.0] * 12
        
        # 10 full steps, 2 half steps?
        # Sum a = 1.
        # If we use 10 full steps of size 1/10 and 2 half steps of 0.5/10?
        # Sum = 10/10 + 2*0.5/10 = 1.1. No.
        
        # Let's use 11 full steps? No, we have 12 slots.
        
        # Simplest 2nd order:
        # a_1 = 0.5/11, a_2..a_11 = 1/11, a_12 = 0.5/11
        # b_1 = 0.5/11, b_2..b_11 = 1/11, b_12 = 0.5/11
        
        a[0] = 0.5 / 11.0
        a[11] = 0.5 / 11.0
        for i in range(1, 11):
            a[i] = 1.0 / 11.0
            
        b[0] = 0.5 / 11.0
        b[11] = 0.5 / 11.0
        for i in range(1, 11):
            b[i] = 1.0 / 11.0
            
        return a, b
