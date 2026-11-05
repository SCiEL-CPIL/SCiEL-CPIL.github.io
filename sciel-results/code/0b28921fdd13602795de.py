"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using random search over Zadoff-Chu + Chirp + Window parameters,
    with optimized DC offset for each candidate.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    t = np.arange(N, dtype=np.float64)
    # Normalized time for windowing, centered at 0
    n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    def optimize_dc(x):
        """Finds the optimal DC offset c to add to x to minimize the value."""
        S_x = np.sum(x)
        C_x = np.convolve(x, x)
        U = np.convolve(x, np.ones(N))
        T = np.convolve(np.ones(N), np.ones(N))
        
        def obj(c):
            conv_c = C_x + 2*c*U + c**2*T
            m = np.max(conv_c)
            s = S_x + N*c
            if abs(s) < 1e-12:
                return float('inf')
            val = abs(2 * N * m / (s * s))
            return val
        
        # The optimal DC is often near 0 or such that sum(h) is large relative to peak.
        # We search in a reasonable range.
        best_c = 0.0
        best_obj = obj(0.0)
        
        # Grid search followed by local refinement
        # Range of DC offset relative to signal amplitude
        amp = np.max(np.abs(x))
        if amp < 1e-10:
            return x
            
        # Search from -2*amp to 2*amp
        candidates = np.linspace(-2*amp, 2*amp, 20)
        for c_start in candidates:
            # Local refinement around c_start
            res = minimize_scalar(obj, bounds=(c_start-amp*0.1, c_start+amp*0.1), method='bounded', options={'xatol': 1e-10})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_c = res.x
        
        return x + best_c

    def generate_signal(mu, alpha, beta, win_type, win_width):
        """
        Generate a candidate signal.
        mu: ZC parameter (integer usually, but float ok for smoothness)
        alpha: Chirp rate
        beta: Phase
        win_type: 0=Rect, 1=Hann, 2=Hamming, 3=Blackman, 4=Gaussian
        win_width: 0.1 to 1.0 (fraction of N used for window support if not full)
                   For simplicity, we apply window on full N but scale width via Gaussian sigma or taper edges.
        """
        # ZC sequence
        zc = np.exp(-1j * np.pi * mu * t * (t + 1) / N)
        # Chirp
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        # Phase
        phase = np.exp(1j * beta)
        
        signal = zc * chirp * phase
        
        # Windowing
        # We use win_width to control the effective support or taper
        if win_type == 0: # Rectangular
            window = np.ones(N)
        elif win_type == 1: # Hann
            # Standard Hann is over full N
            # We can modify to be narrower? 
            # Let's stick to standard windows but vary the "effective" length by zeroing edges?
            # Actually, just using standard windows is fine. win_width can be used to zero out edges.
            window = 0.5 - 0.5 * np.cos(2 * np.pi * n_norm)
        elif win_type == 2: # Hamming
            window = 0.54 - 0.46 * np.cos(2 * np.pi * n_norm)
        elif win_type == 3: # Blackman
            window = 0.42 - 0.5 * np.cos(2 * np.pi * n_norm) + 0.08 * np.cos(4 * np.pi * n_norm)
        elif win_type == 4: # Gaussian
            # sigma controlled by win_width (1.0 = wide, 0.1 = narrow)
            sigma = win_width * N / 2.0
            window = np.exp(-0.5 * (n_norm / (win_width if win_width > 0.01 else 0.01))**2)
        else:
            window = np.ones(N)
            
        # Apply edge zeroing based on win_width for non-Gaussian to effectively shorten the window
        # If win_width < 1.0, zero out the outer (1-win_width)/2 fraction
        if win_width < 1.0 and win_type != 4:
            edge = int((1.0 - win_width) * N / 2.0)
            if edge > 0:
                window[:edge] = 0
                window[-edge:] = 0
                # Smooth the transition? For now, hard cut is simpler and often robust
                # Or better: multiply by a ramp
                ramp = np.linspace(0, 1, edge)
                window[:edge] *= ramp
                window[-edge:] *= ramp[::-1]

        h_ac = np.real(signal) * window
        return h_ac

    # Main search loop
    np.random.seed(42)
    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    best_val_found = float('inf')
    best_h_found = None
    
    # Number of iterations to attempt
    # We estimate ~1ms per eval (conv + DC opt). So 10000 evals in 10s.
    # DC opt is the bottleneck. Let's try 2000 iterations.
    
    for i in range(5000):
        if time.time() - start_time > max_time:
            break
            
        # Random parameters
        mu = np.random.uniform(0, 30)
        alpha = np.random.uniform(-5, 5)
        beta = np.random.uniform(0, 2 * np.pi)
        win_type = np.random.randint(0, 5)
        win_width = np.random.uniform(0.5, 1.0)
        
        h_ac = generate_signal(mu, alpha, beta, win_type, win_width)
        
        # Skip if signal is trivial
        if np.max(np.abs(h_ac)) < 1e-6:
            continue
            
        h_full = optimize_dc(h_ac)
        
        val = evaluate(h_full)
        
        if val < best_val_found:
            best_val_found = val
            best_h_found = h_full.copy()
            
            # If we find a very good value, we might want to refine around it
            # But for now, just keep searching as random search is robust
            if best_val_found < 1.05:
                # We could break, but let's try to squeeze a bit more
                # Or just break to save time?
                # Let's break if we are significantly better than seed
                if best_val_found < 1.02:
                    break

    if best_h_found is not None and best_val_found < best_val:
        return best_h_found
    return best_h
# EVOLVE-BLOCK-END
