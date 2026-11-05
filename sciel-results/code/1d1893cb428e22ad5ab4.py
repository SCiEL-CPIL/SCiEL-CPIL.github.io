import numpy as np

N = 400

def construct():
        # Start with seed
        best_h = np.ones(N)
        best_val = abs(2 * N * np.max(np.convolve(best_h, best_h)) / np.sum(best_h) ** 2)
        
        # Optimization over amplitude and phase for a few frequencies
        # h = c0 + c1 * cos(2*pi*k*i/N + phase)
        # We normalize so sum(h) = 1 to simplify, or just compare raw values.
        # Since value is scale invariant, we can fix sum(h)=1.
        # h_i = 1/N + A * cos(...)
        # sum(h) = 1 + A * sum(cos(...)).
        # If k is integer, sum(cos) is 0. So sum(h) = 1.
        
        for k in range(1, 10): # Try low frequencies
            # The cosine mode has sum 0 for integer k
            mode = np.cos(2 * np.pi * k * np.arange(N) / N)
            # Search amplitude A
            # h = 1/N + A * mode
            # We want to minimize V(h)
            # We know V(1/N) = 2.
            # Let's check small A.
            
            # Simple ternary search or grid for A in [-0.5, 0.5]
            # Actually, just check a few points.
            for A in np.linspace(-0.1, 0.1, 10):
                h = 1.0/N + A * mode
                val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
        
        return best_h
