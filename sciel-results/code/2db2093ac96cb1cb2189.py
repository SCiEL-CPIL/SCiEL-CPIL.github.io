"""Optimized program for second autocorrelation inequality."""
import numpy as np
import time

N = 50

def _evaluate(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return -np.inf
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    # spacing w: points are linspace(-0.5, 0.5, m+2), so w = 1.0/(m+1)
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y has length m+2, so i goes from 0 to m (inclusive), that's m+1 terms
    y_sq = y**2
    y_prod = y * np.roll(y, -1)  # y[i] * y[i+1]
    # But roll wraps around; we need y[i]*y[i+1] for i=0..m
    # y has indices 0..m+1, we need pairs (0,1), (1,2), ..., (m, m+1)
    y_prod = y[:m+1] * y[1:m+2]
    y_next_sq = y[1:m+2]**2
    
    l2 = (w / 3.0) * np.sum(y_sq[:m+1] + y_prod + y_next_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 <= 0 or norm_inf <= 0:
        return -np.inf
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _tent_basis(knots, amplitudes, x):
    """
    Piecewise linear tent function on [0, 1] with given knots and amplitudes.
    knots: sorted array of knot positions in [0, 1], length K
    amplitudes: array of length K, the value at each knot
    x: array of points in [0, 1]
    Returns: piecewise linear interpolation
    """
    # Extend with boundary zeros for proper tent shape
    knots_ext = np.concatenate(([0.0], knots, [1.0]))
    amps_ext = np.concatenate(([0.0], amplitudes, [0.0]))
    return np.interp(x, knots_ext, amps_ext)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Map 50 heights to positions on [-1/4, 1/4]
    # The heights are on equal intervals of [-1/4, 1/4]
    # So the 50 points are at: -1/4 + i * (1/2) / 50 for i = 0..49
    # Actually, "equal intervals of [-1/4, 1/4]" means 50 intervals, 51 points?
    # Or 50 points dividing the interval into 49 subintervals?
    # Let's re-read: "Choose 50 nonnegative step heights h on equal intervals of [-1/4, 1/4]"
    # I think this means 50 heights at equally spaced points in [-1/4, 1/4]
    # The evaluator uses np.linspace(-0.5, 0.5, m+2) for the convolution points
    # But the heights h are just 50 values that get convolved
    
    # Let me think about this differently. The heights h are 50 values.
    # g = np.convolve(h, h) has length 2*50-1 = 99
    # m = 99
    # y = [0, g[0], ..., g[98], 0] has length 101
    # points = np.linspace(-0.5, 0.5, 101), so w = 1.0/100 = 0.01
    
    # The objective is:
    # l2 = sum_{i=0}^{99} (0.01/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # norm_1 = sum(|g|) / 100
    # norm_inf = max(|g|)
    # value = l2 / (norm_1 * norm_inf)
    
    # Strategy: Use a few-parameter family. Try a Gaussian-like or tent shape.
    # The optimal shape for autocorrelation ratio might be a sharp peak or a specific profile.
    
    # Let's try optimizing a small set of parameters:
    # 1. A single tent/triangle shape with variable width and height
    # 2. A Gaussian with variable width
    # 3. A combination of a few tents
    
    best_value = -np.inf
    best_h = np.ones(N)
    
    # First, evaluate the seed
    seed_val = _evaluate(best_h)
    best_value = seed_val
    
    # Try a family of shapes: sum of K tents
    # Parameterize by: for each tent, center and width; and amplitude
    # To keep it simple, try 1-3 tents with symmetric placement
    
    # Map parameter space:
    # For K tents, we have K*3 parameters (center, width, amplitude)
    # But centers should be in [-0.25, 0.25] mapped to [0, 49]
    # Width in [1, 20] or so
    # Amplitude in [0.1, 5.0]
    
    # Let's do a grid search over a small parameter space, then local refinement
    
    # Try single tent shapes
    # Tent at center c with half-width w_t, amplitude A
    # h[i] = A * max(0, 1 - |i - c| / w_t)
    
    # Grid search
    centers = np.arange(10, 40, 1.0)  # symmetric-ish
    widths = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    amps = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    for c in centers:
        for wt in widths:
            for A in amps:
                if time.time() > deadline:
                    break
                h = np.zeros(N)
                for i in range(N):
                    val = A * max(0.0, 1.0 - abs(i - c) / wt)
                    h[i] = val
                val = _evaluate(h)
                if val > best_value:
                    best_value = val
                    best_h = h.copy()
    
    # Try double tent shapes (two peaks)
    if time.time() < deadline:
        c1_range = np.arange(5, 25, 2.0)
        c2_range = np.arange(25, 45, 2.0)
        widths2 = np.array([3.0, 5.0, 7.0, 10.0])
        amps2 = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        
        for c1 in c1_range:
            for c2 in c2_range:
                for wt1 in widths2:
                    for wt2 in widths2:
                        for A1 in amps2:
                            for A2 in amps2:
                                if time.time() > deadline:
                                    break
                                h = np.zeros(N)
                                for i in range(N):
                                    h[i] = A1 * max(0.0, 1.0 - abs(i - c1) / wt1)
                                    h[i] += A2 * max(0.0, 1.0 - abs(i - c2) / wt2)
                                val = _evaluate(h)
                                if val > best_value:
                                    best_value = val
                                    best_h = h.copy()
    
    # Try Gaussian shapes
    if time.time() < deadline:
        centers_g = np.arange(15, 35, 1.0)
        sigmas = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0])
        amps_g = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
        
        for c in centers_g:
            for sig in sigmas:
                for A in amps_g:
                    if time.time() > deadline:
                        break
                    i_arr = np.arange(N, dtype=np.float64)
                    h = A * np.exp(-0.5 * ((i_arr - c) / sig) ** 2)
                    val = _evaluate(h)
                    if val > best_value:
                        best_value = val
                        best_h = h.copy()
    
    # Local refinement using coordinate ascent on the best shape
    if time.time() < deadline:
        h = best_h.copy()
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = h[i]
                for factor in [0.5, 0.75, 1.25, 1.5, 2.0]:
                    h[i] = old_val * factor
                    if h[i] < 0:
                        h[i] = 0
                    val = _evaluate(h)
                    if val > best_value + 1e-12:
                        best_value = val
                        improved = True
                    else:
                        h[i] = old_val
                        break
                if time.time() > deadline:
                    break
            if not improved:
                break
    
    # Ensure nonnegative and finite
    best_h = np.maximum(best_h, 0.0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    
    return list(best_h)
