import numpy as np

N = 100

def construct():
    # Try different chirp rates
    best_score = float('inf')
    best_h = np.full(N, 0.5)
    
    # The phase is pi * alpha * n^2
    # We want the sum of cos to be close to 0
    # And the autocorrelation of cos to be close to 0
    
    n = np.arange(N)
    
    # Sweep alpha
    # alpha determines the frequency sweep
    # Small alpha: slow sweep, might not average out
    # Large alpha: fast sweep, averages out well but might alias
    # Let's try a range of alphas
    
    for alpha in np.linspace(0.01, 5.0, 100):
        phase = np.pi * alpha * n**2
        h = 0.5 + 0.5 * np.cos(phase)
        
        # Check constraints
        if not (np.all(h >= 0) and np.all(h <= 1)):
            continue
            
        # Check sum
        # Sum should be close to N/2
        # We can normalize if needed, but cosine sum is small
        # Let's adjust to ensure sum is exactly N/2?
        # The task says "sum is exactly len/2".
        # The cosine term sum is not exactly 0.
        # We can project h onto the hyperplane sum(h) = N/2
        # h_new = h - (sum(h) - N/2) / N
        # This might push values out of [0,1].
        # Given the small variance, it should be fine.
        
        sum_h = np.sum(h)
        h_adj = h - (sum_h - N/2) / N
        
        if np.any(h_adj < -1e-9) or np.any(h_adj > 1 + 1e-9):
            # If projection violates bounds, skip or clip
            # Clipping will change the sum
            # Let's just check if the raw h is good enough
            # Or use a more careful construction
            pass
        else:
            h_use = h_adj
            
            # Calculate max overlap
            # S_k = sum h_i (1 - h_{i-k})
            # Using circular or linear?
            # Let's assume linear for now as it's the harder case and "shifted" usually implies linear in 1D arrays unless "circular" is specified.
            # However, if circular is intended, the score would be ~25.
            # If linear, score ~25 is a huge win.
            
            max_overlap = 0
            for k in range(N):
                # Linear shift
                # h[0..N-k-1] and h[k..N-1]
                if k == 0:
                    # h_i (1-h_i)
                    s = np.sum(h_use * (1 - h_use))
                else:
                    h1 = h_use[:N-k]
                    h2 = h_use[k:]
                    s = np.sum(h1 * (1 - h2))
                if s > max_overlap:
                    max_overlap = s
            
            if max_overlap < best_score:
                best_score = max_overlap
                best_h = h_use
                
    # If the best found is still bad, fallback
    if best_score > 49:
         best_h = np.full(N, 0.5)
         
    return best_h
