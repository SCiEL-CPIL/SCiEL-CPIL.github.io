import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # length of g = convolve(h, h)

def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    # l2 calculation
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

    best_h = np.ones(N)
    best_val = compute_value(best_h)

    # Strategy: Convex-Concave Fractional Program via Reparameterization
    # We exploit the scale invariance of the objective.
    # value(h) = l2(h) / (norm_1(h) * norm_inf(h))
    # Let h = a * u, where u is a shape vector (u >= 0) and a > 0 is scale.
    # l2(a*u) = a^2 * l2(u)
    # norm_1(a*u) = a * norm_1(u)
    # norm_inf(a*u) = a * norm_inf(u)
    # value(a*u) = (a^2 * l2(u)) / (a^2 * norm_1(u) * norm_inf(u)) = l2(u) / (norm_1(u) * norm_inf(u))
    
    # The objective is invariant to scaling. We can fix the scale to simplify the optimization landscape.
    # A common trick is to fix norm_inf(u) = 1 or norm_1(u) = 1.
    # However, fixing norm_inf = 1 introduces non-smoothness (max function).
    # Fixing norm_1 = 1 is smoother (linear constraint).
    
    # Let's reparameterize h such that sum(h) = 1 (which implies norm_1(g) is related to sum(h)^2).
    # Actually, norm_1(g) = sum(|conv(h,h)|). If h >= 0, conv(h,h) >= 0, so norm_1(g) = sum(conv(h,h)) = (sum(h))^2.
    # So if we fix sum(h) = S, then norm_1(g) = S^2.
    # norm_inf(g) = max(conv(h,h)).
    # l2 is quadratic in h.
    
    # Let h = S * u, where sum(u) = 1.
    # value = l2(S*u) / (norm_1(S*u) * norm_inf(S*u))
    # l2(S*u) = S^2 * l2(u)
    # norm_1(S*u) = S^2 * norm_1(u) = S^2 * (sum(u))^2 = S^2 * 1 = S^2
    # norm_inf(S*u) = S^2 * norm_inf(u)
    # value = S^2 * l2(u) / (S^2 * S^2 * norm_inf(u)) ... wait.
    
    # Let's re-evaluate.
    # g = conv(h, h).
    # If h = S*u, g = S^2 * conv(u, u) = S^2 * gu.
    # norm_1(g) = sum(|S^2 * gu|) = S^2 * sum(gu) = S^2 * (sum(u))^2.
    # If sum(u) = 1, norm_1(g) = S^2.
    # norm_inf(g) = max(|S^2 * gu|) = S^2 * max(gu).
    # l2 term involves y which is derived from g.
    # y = [0, g, 0]. So y scales with S^2.
    # l2 = sum( ... y^2 ... ) scales with (S^2)^2 = S^4.
    # value = S^4 * l2(u) / (S^2 * S^2 * norm_inf(u)) = l2(u) / norm_inf(u).
    
    # So, if we fix sum(h) = 1, the objective becomes:
    # maximize l2(h) / norm_inf(h)
    # subject to h >= 0, sum(h) = 1.
    
    # This is a convex-concave fractional program?
    # l2(h) is convex (quadratic form).
    # norm_inf(h) is convex.
    # Maximizing convex/convex is generally hard, but we can use the Dinkelbach's algorithm or
    # simply optimize the ratio directly using a change of variables or penalty.
    
    # Alternatively, we can fix norm_inf(h) = 1 and maximize l2(h) subject to sum(h)=1 and h>=0.
    # But norm_inf(h) = max(conv(h,h)) is non-smooth.
    
    # Let's try the approach: Fix sum(h) = 1.
    # Objective: f(h) = l2(h) / max(conv(h,h)).
    # We can use the substitution t = 1/max(conv(h,h)).
    # Or simply use scipy.optimize.minimize with the ratio, but normalize h to sum=1 inside the objective.
    
    # Let's define an objective that takes a vector z, normalizes it to sum=1 (if positive),
    # and returns -value.
    
    def objective_normalized(z):
        # z is the raw parameter
        # We want h >= 0. Let's use softmax or just clip and normalize.
        # Clipping and normalizing is non-smooth at 0.
        # Let's use h = exp(z) / sum(exp(z)) ? No, that restricts h to be very small if z is negative.
        # Let's use h = z / sum(z) if z >= 0.
        
        # To handle non-negativity smoothly, let's map z to h via h_i = max(z_i, 0).
        # Then normalize.
        
        h = np.maximum(z, 0)
        s = np.sum(h)
        if s < 1e-12:
            return 0.0 # invalid
        h = h / s
        
        val = compute_value(h)
        return -val

    # We need a good starting point.
    # The seed is uniform.
    # Let's try to find a shape that concentrates mass to increase l2/norm_inf.
    # l2 is roughly integral of g^2. norm_inf is max g.
    # To maximize ratio, we want g to be as "flat" as possible relative to its max?
    # Or rather, we want l2 (which is sum of squares) to be large compared to max.
    # This implies we want the autocorrelation to be spread out, not spiky.
    # A wide, smooth h leads to a wide, smooth g.
    # A narrow h leads to a narrow, spiky g.
    # For a box of width W, g is a triangle. max is W^2/4 (approx). l2 is integral of triangle^2.
    # Ratio might be better for wider boxes.
    
    # Let's initialize with several shapes normalized to sum=1.
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N))
    
    # 2. Triangular
    x = np.linspace(-1, 1, N)
    starts.append(np.maximum(0, 1 - np.abs(x)))
    
    # 3. Gaussian
    for sigma in [0.2, 0.5, 1.0]:
        starts.append(np.exp(-x**2 / (2*sigma**2)))
        
    # 4. Cosine bump
    for width in [0.5, 1.0]:
        mask = np.abs(x) < width
        h = np.zeros(N)
        h[mask] = 0.5 * (1 + np.cos(np.pi * x[mask] / width))
        starts.append(h)

    # Optimize each start
    for h0 in starts:
        if time.time() > deadline:
            break
            
        # Normalize start
        h0 = np.maximum(0, h0)
        if np.sum(h0) > 0:
            h0 = h0 / np.sum(h0)
        else:
            continue
            
        # Use L-BFGS-B on the unnormalized z, but objective normalizes.
        # Actually, it's better to optimize on the simplex directly or use log-transform?
        # Let's stick to z where h = normalize(max(z,0)).
        
        # Initial z can be log(h0 + 1e-6) to ensure positivity in the soft sense?
        # No, if we use max(z,0), we should start with z=h0.
        
        try:
            result = minimize(
                objective_normalized,
                h0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N, # z >= 0
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = np.maximum(0, result.x)
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt / s
                v_opt = compute_value(h_opt)
                if v_opt > best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # If we have time, do random restarts
    if time.time() < deadline - 20:
        rng = np.random.default_rng(50)
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random shape
            h_rand = rng.random(N)
            # Mix with best
            h_rand = 0.5 * h_rand + 0.5 * best_h
            h_rand = np.maximum(0, h_rand)
            s = np.sum(h_rand)
            if s > 0:
                h_rand = h_rand / s
                
            try:
                result = minimize(
                    objective_normalized,
                    h_rand,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                h_opt = np.maximum(0, result.x)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt / s
                    v_opt = compute_value(h_opt)
                    if v_opt > best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    return best_h.tolist()
