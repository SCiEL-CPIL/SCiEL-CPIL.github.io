"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s limit

# EVOLVE-BLOCK-START
def compute_value(h):
    """
    Computes the objective value for a given height array h.
    h: numpy array of length N, non-negative.
    """
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(h):
    """Negative of compute_value for minimization."""
    if h is None or not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-15:
        return 0.0
    return -compute_value(h)

def coord_descent(h, max_sweeps=30, tol=1e-10, deadline=None):
    """
    Coordinate descent: for each coordinate, find optimal 1D value via golden section.
    Since the objective is scale-invariant, we normalize h to unit sum before each sweep.
    """
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-15:
        return h
    best_val = -neg_value(h)
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Save old value
            old = h[i]
            # Normalize h to unit sum for scale invariance
            s = np.sum(h)
            if s < 1e-15:
                break
            h_norm = h / s
            
            # Try golden section search for optimal h[i]
            # We search over h[i] in [0, 2*s] (since normalized, h[i] in [0,2])
            # Actually, let's search over the unnormalized value but normalize internally
            # Better: search over the normalized coordinate directly
            
            # Golden section search on h[i] in [0, 2.0] (normalized space)
            a, b = 0.0, 2.0
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            h_c = h_norm.copy()
            h_c[i] = c
            fc = -neg_value(h_c)
            
            h_d = h_norm.copy()
            h_d[i] = d
            fd = -neg_value(h_d)
            
            for _ in range(40):
                if deadline is not None and time.time() > deadline:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    h_c = h_norm.copy()
                    h_c[i] = c
                    fc = -neg_value(h_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    h_d = h_norm.copy()
                    h_d[i] = d
                    fd = -neg_value(h_d)
                
            # Best of c and d
            if fc < fd:
                best_x = c
                best_f = fc
            else:
                best_x = d
                best_f = fd
            
            # Update h (unnormalized, but we'll normalize at end)
            h = h_norm * s  # restore scale
            h[i] = best_x * s
            h = np.maximum(h, 0.0)
            
            # Check improvement
            new_val = -neg_value(h)
            if new_val > best_val + tol:
                best_val = new_val
                improved = True
        
        if not improved:
            break
    
    return h

def nelder_mead_polish(h, deadline=None, maxiter=200):
    """Nelder-Mead local polish on the normalized h."""
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h
    h_norm = h / s
    
    def obj(x):
        x = np.maximum(x, 0.0)
        return -compute_value(x)
    
    try:
        res = minimize(obj, h_norm, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.x is not None and np.all(np.isfinite(res.x)):
            res.x = np.maximum(res.x, 0.0)
            rs = np.sum(res.x)
            if rs > 1e-15:
                res.x = res.x / rs
            return res.x
    except Exception:
        pass
    return h_norm

def construct():
    global START_TIME
    deadline = START_TIME + TIME_LIMIT
    
    # Initial seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Generate diverse initial guesses - concentrated smooth shapes
    x = np.linspace(-0.25, 0.25, N)
    initial_guesses = []
    
    # 1. Uniform (Seed)
    initial_guesses.append(np.ones(N))
    
    # 2. Gaussians with various widths (concentrated)
    for sigma in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18]:
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        initial_guesses.append(h_gauss)
        
    # 3. Cosine bumps with different widths
    for width in [0.5, 0.7, 1.0, 1.3]:
        h_cos = np.maximum(np.cos(np.pi * x / width), 0)
        h_cos = h_cos ** 2  # smoother
        initial_guesses.append(h_cos)
    
    # 4. Polynomial bumps
    for p in [2, 4, 6, 8]:
        h_poly = (1 - (2*x)**2)**p
        h_poly = np.maximum(h_poly, 0)
        initial_guesses.append(h_poly)
    
    # 5. Triangle
    h_tri = np.maximum(1 - np.abs(2*x), 0)
    initial_guesses.append(h_tri)
    
    # 6. Flat-top (uniform on subinterval)
    for half_w in [0.05, 0.08, 0.10, 0.12]:
        h_flat = np.zeros(N)
        mask = np.abs(x) <= half_w
        h_flat[mask] = 1.0
        initial_guesses.append(h_flat)
    
    # 7. Super-Gaussian (flatter top)
    for n in [4, 6, 8]:
        h_supg = np.exp(-np.abs(x)**n / (0.15**n))
        initial_guesses.append(h_supg)
    
    for h0 in initial_guesses:
        if time.time() > deadline - 5:
            break
        
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) < 1e-15:
            continue
        
        # Phase 1: Coordinate descent
        h_cd = coord_descent(h0, max_sweeps=20, deadline=deadline - 10)
        val_cd = compute_value(h_cd)
        
        # Phase 2: Nelder-Mead polish
        h_nm = nelder_mead_polish(h_cd, deadline=deadline - 5, maxiter=150)
        val_nm = compute_value(h_nm)
        
        # Pick best
        if val_nm >= val_cd:
            h_candidate = h_nm
            val_candidate = val_nm
        else:
            h_candidate = h_cd
            val_candidate = val_cd
        
        if val_candidate > best_val:
            best_val = val_candidate
            best_h = h_candidate
    
    # Final safety: ensure finite and non-negative
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)
# EVOLVE-BLOCK-END
