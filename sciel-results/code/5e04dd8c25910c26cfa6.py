import numpy as np

N = 600

def construct():
    # Define the grid
    dx = (0.25 - (-0.25)) / N
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + dx/2
    
    # Initialize with a raised cosine
    # f(x) = cos^2(pi * x / 0.5) = cos^2(2 * pi * x)
    h = np.cos(np.pi * x / 0.25)**2  # Wait, period is 0.5?
    # On [-0.25, 0.25], cos(2*pi*x) goes from cos(-pi/2)=0 to cos(0)=1 to cos(pi/2)=0.
    # So cos(2*pi*x)**2 is a bump.
    h = np.cos(2 * np.pi * x)**2
    
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    # Simple local refinement using coordinate descent
    # Objective: Minimize max(f*f) / (||f||_1 * max(f))
    # We can fix ||f||_1 = 1 and max(f) = 1? No, they are coupled.
    # Let's just minimize the raw ratio.
    
    def score(h):
        # Compute autoconvolution using FFT
        # Pad to avoid circular convolution
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        C = np.fft.ifft(H * H).real
        # The autoconvolution is scaled by dx
        C = C * dx
        # Find the max
        max_c = np.max(C)
        
        # Calculate norms
        l1 = np.sum(h) * dx
        linf = np.max(h)
        
        if l1 == 0 or linf == 0:
            return float('inf')
            
        return max_c / (l1 * linf)
    
    current_score = score(h)
    
    # Coordinate Descent
    # We will try to adjust each h_i
    # To speed up, we only do a few passes
    for _ in range(5):
        for i in range(N):
            # Try a small perturbation
            original_h = h[i]
            # Try increasing
            for factor in [1.1, 1.01, 0.99, 0.9]:
                h[i] = original_h * factor
                if h[i] < 0: continue
                s = score(h)
                if s < current_score:
                    current_score = s
                else:
                    h[i] = original_h
                    break
            h[i] = original_h # Reset before trying other factors? No, the loop above breaks on first improvement or tries all.
            # Let's do a simpler local search: try +/- 10%
            best_h = h[i]
            best_s = current_score
            for factor in [0.9, 1.0, 1.1]:
                h[i] = original_h * factor
                if h[i] < 0: continue
                s = score(h)
                if s < best_s:
                    best_s = s
                    best_h = h[i]
            h[i] = best_h
            current_score = best_s

    return list(h)
