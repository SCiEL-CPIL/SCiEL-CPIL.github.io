# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def _area_fast(x, n=64, M=20000):
    """Compute union area via quadrature at M heights."""
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
    return float(np.mean(union_len))

def _area_coarse(x, n=64, M=1000):
    """Coarse area for fast optimization."""
    return _area_fast(x, n, M)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 630.0
    safety_margin = 10.0
    
    # Initializations
    # 1. Fully overlapping (degenerate minimum for area, but invalid for "needle" if interpreted strictly? 
    #    Actually, the problem doesn't forbid overlap. It minimizes union area.
    #    If x_i are all 0, L_i = a_i * y. 
    #    At y=0, L_i=0. At y=1, L_i=a_i.
    #    The union of these triangles is the triangle with vertices (0,0), (1/64, 0), (1, 1).
    #    Area = 0.5 * base * height = 0.5 * (1/64) * 1 = 1/128.
    #    Wait, the union area formula:
    #    At height y, the cross sections are [a_i*y, a_i*y + (1-y)/64].
    #    If x_i=0, L_i = a_i*y.
    #    a_i = (i+1)/64.
    #    The intervals are [ (i+1)/64 * y, (i+1)/64 * y + (1-y)/64 ].
    #    Let's check overlap.
    #    Interval i and i+1:
    #    L_i = i/64 * y + 1/64 * y? No, a_i = (i+1)/64.
    #    L_i = (i+1)/64 * y.
    #    L_{i+1} = (i+2)/64 * y.
    #    Gap = L_{i+1} - L_i = y/64.
    #    Width w = (1-y)/64.
    #    If y/64 < (1-y)/64 => y < 1-y => y < 0.5, they overlap.
    #    If y > 0.5, they are disjoint.
    #    So at y=0.5, they just touch.
    #    This configuration is actually quite good.
    
    inits = []
    
    # 1. Zero initialization (all x_i = 0)
    inits.append(np.zeros(n, dtype=np.float64))
    
    # 2. Small random perturbations around zero
    rng = np.random.default_rng(42)
    for scale in [0.01, 0.05, 0.1, 0.2]:
        inits.append(rng.uniform(-scale, scale, size=n))
        
    # 3. "Aligned" initialization from knowledge: x_i = -a_i + 0.5
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    inits.append(-a + 0.5)
    
    # 4. Keich-like construction (from current program, kept as a fallback)
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_pos = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_pos[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    inits.append(keich_pos)
    
    # 5. Random
    inits.append(rng.uniform(0.0, 1.0, size=n))

    best_x = None
    best_area = np.inf

    # Strategy: Use L-BFGS-B on a subset of initializations with a time budget.
    # L-BFGS-B is good for smooth-ish problems. The area function is piecewise linear (in y) but the union operation makes it non-differentiable at boundaries.
    # However, numerical gradients (finite differences) often work well enough for L-BFGS-B to find a local minimum.
    
    # We will run optimization on the most promising initializations.
    # First, screen with coarse area to pick the best starting point.
    
    coarse_areas = []
    for x0 in inits:
        if time.time() > deadline - safety_margin - 10.0:
            break
        area_c = _area_coarse(x0, n, 1000)
        coarse_areas.append(area_c)
        
    # Sort initializations by coarse area
    sorted_inits = sorted(zip(coarse_areas, inits), key=lambda t: t[0])
    
    # Try L-BFGS-B on the top 3 initializations
    for i in range(min(3, len(sorted_inits))):
        if time.time() > deadline - safety_margin - 30.0:
            break
            
        area_c, x0 = sorted_inits[i]
        
        # If this coarse area is already worse than our best fine area, skip?
        # No, optimization might improve it significantly.
        
        x_opt, area_opt, info = so.minimize(
            _area_coarse, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50}
        )
        
        # Check if optimization succeeded
        if area_opt < best_area:
            # Fine evaluate
            area_fine = _area_fast(x_opt, n, 20000)
            if area_fine < best_area:
                best_area = area_fine
                best_x = x_opt.copy()
                
    # If L-BFGS-B didn't find anything better (or failed), fall back to the best coarse initialization
    if best_x is None or _area_fast(sorted_inits[0][1], n, 20000) < best_area:
        best_x = sorted_inits[0][1].copy()
        best_area = _area_fast(best_x, n, 20000)

    # Try a few more random restarts with L-BFGS-B if time permits
    if time.time() < deadline - safety_margin - 60.0:
        for _ in range(5):
            if time.time() > deadline - safety_margin - 10.0:
                break
            x0 = rng.uniform(-0.2, 0.8, size=n)
            x_opt, area_opt, _ = so.minimize(
                _area_coarse, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-7}
            )
            if area_opt < best_area * 0.99: # Only fine-evaluate if significantly better
                area_fine = _area_fast(x_opt, n, 20000)
                if area_fine < best_area:
                    best_area = area_fine
                    best_x = x_opt.copy()

    if best_x is None:
        best_x = np.zeros(n, dtype=np.float64)

    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
