"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Piecewise linear tent basis approach
    # We'll use a small number of knots to define the shape
    # The heights are nonnegative and defined on N=50 equal intervals
    
    # Strategy: Use a few tent functions (triangular bumps) with variable
    # positions and amplitudes, combined to create the height profile.
    
    def evaluate_heights(h):
        """Compute the score for given heights h (length N)."""
        h = np.asarray(h, dtype=float)
        g = np.convolve(h, h)
        m = len(g)
        # y = [0, g[0], ..., g[m-1], 0]
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        # Spacing w for equally spaced points on [-0.5, 0.5] with m+2 points
        w = 1.0 / (m + 1)
        
        # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
        
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 == 0 or norm_inf == 0:
            return 0.0
        return l2 / (norm_1 * norm_inf)
    
    # Tent basis functions: triangular bumps
    # We'll parameterize with a few tents
    
    def make_tent(center, width, amplitude, n):
        """Create a tent function on n points in [-0.25, 0.25]."""
        x = np.linspace(-0.25, 0.25, n)
        # Triangle centered at 'center' with half-width 'width'
        h = np.maximum(0, amplitude * (1 - np.abs(x - center) / width))
        return h
    
    # Start with a symmetric shape that's known to work well
    # Based on findings: [1, 1.8, 1.8, 1] pattern works better than uniform
    
    # Use a small number of parameterized tents
    # Try different configurations
    
    best_score = 0.0
    best_h = None
    
    # Seed: uniform
    h_uniform = np.ones(N)
    score_uniform = evaluate_heights(h_uniform)
    best_score = score_uniform
    best_h = h_uniform.copy()
    
    # Try various tent configurations
    # We'll use 2-4 tents with variable centers, widths, and amplitudes
    
    np.random.seed(42)
    
    for n_tents in [2, 3, 4]:
        for trial in range(2000):
            if time.time() > deadline:
                break
            
            # Generate random tent parameters
            tents = []
            for _ in range(n_tents):
                center = np.random.uniform(-0.2, 0.2)
                width = np.random.uniform(0.05, 0.25)
                amplitude = np.random.uniform(0.5, 2.0)
                tents.append((center, width, amplitude))
            
            # Sum the tents
            h = np.zeros(N)
            for center, width, amplitude in tents:
                h += make_tent(center, width, amplitude, N)
            
            # Ensure nonnegative
            h = np.maximum(0, h)
            
            # Normalize to avoid trivial scaling issues
            # Actually, the score is scale-invariant, so we don't need to normalize
            # But let's keep values reasonable
            if np.max(h) > 0:
                h = h / np.max(h) * 1.0  # normalize max to 1
            
            score = evaluate_heights(h)
            if score > best_score:
                best_score = score
                best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Also try some structured patterns
    # Symmetric bump patterns
    
    # Pattern: single central bump
    for width in np.linspace(0.1, 0.4, 10):
        if time.time() > deadline:
            break
        h = make_tent(0.0, width, 1.0, N)
        if np.max(h) > 0:
            h = h / np.max(h)
            score = evaluate_heights(h)
            if score > best_score:
                best_score = score
                best_h = h.copy()
    
    # Pattern: two symmetric bumps
    for sep in np.linspace(0.05, 0.2, 10):
        for width in np.linspace(0.05, 0.15, 5):
            if time.time() > deadline:
                break
            h = make_tent(-sep, width, 1.0, N) + make_tent(sep, width, 1.0, N)
            if np.max(h) > 0:
                h = h / np.max(h)
                score = evaluate_heights(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Pattern: plateau (like the [1, 1.8, 1.8, 1] finding)
    # Create a shape with a flat top
    for plateau_width in np.linspace(0.05, 0.2, 10):
        for total_width in np.linspace(plateau_width + 0.05, 0.4, 5):
            if time.time() > deadline:
                break
            h = np.zeros(N)
            x = np.linspace(-0.25, 0.25, N)
            # Trapezoidal shape
            inner = total_width / 2 - plateau_width / 2
            outer = total_width / 2
            for i, xi in enumerate(x):
                if abs(xi) <= plateau_width / 2:
                    h[i] = 1.0
                elif abs(xi) <= outer:
                    h[i] = 1.0 - (abs(xi) - plateau_width / 2) / inner if inner > 0 else 0
            h = np.maximum(0, h)
            if np.max(h) > 0:
                h = h / np.max(h)
                score = evaluate_heights(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Final local optimization around best_h using small perturbations
    if best_h is not None and time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            # Small random perturbation
            h_perturbed = best_h + np.random.normal(0, 0.05, N)
            h_perturbed = np.maximum(0, h_perturbed)
            if np.max(h_perturbed) > 0:
                h_perturbed = h_perturbed / np.max(h_perturbed)
                score = evaluate_heights(h_perturbed)
                if score > best_score:
                    best_score = score
                    best_h = h_perturbed.copy()
    
    # Return as list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
