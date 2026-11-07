import numpy as np
import time
from scipy.optimize import brentq, minimize_scalar

def _hermite_even_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in ascending power order."""
    H0 = np.array([1.0])
    H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
    H8 = np.array([1680.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
    H12 = np.array([10321920.0, 0.0, -5529600.0, 0.0, 8709120.0, 0.0, -4915200.0, 0.0, 1032192.0, 0.0, -98304.0, 0.0, 4096.0])
    return H0, H4, H8, H12

def _build_q(c0, c1, c2):
    """Build q(x) = g(x)/x^2 coefficients in ascending power order."""
    H0, H4, H8, H12 = _hermite_even_coeffs()
    # c3 from g(0)=0: c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=10321920
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 10321920.0
    # Build g coefficients
    g = np.zeros(13)
    g[0] += c0 * H0[0]
    for i in range(len(H4)):
        g[i] += c1 * H4[i]
    for i in range(len(H8)):
        g[i] += c2 * H8[i]
    for i in range(len(H12)):
        g[i] += c3 * H12[i]
    # q = g / x^2, so q coefficients are g[2:]
    q = g[2:]
    # Trim trailing zeros
    while len(q) > 1 and abs(q[-1]) < 1e-15:
        q = q[:-1]
    return q

def _eval_q(qcoeffs, x):
    """Evaluate polynomial q at x using Horner's method."""
    result = 0.0
    for c in reversed(qcoeffs):
        result = result * x + c
    return result

def _largest_sign_change_root(qcoeffs):
    """Find the largest real root of q where q changes sign."""
    # Find all real roots using numpy
    roots = np.roots(qcoeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            real_roots.append(float(r.real))
    real_roots.sort()
    
    best_r = None
    for r in real_roots:
        # Check sign change with adaptive epsilon
        for eps in [1e-6, 1e-5, 1e-4, 1e-3]:
            vm = _eval_q(qcoeffs, r - eps)
            vp = _eval_q(qcoeffs, r + eps)
            if vm * vp < 0:
                if best_r is None or r > best_r:
                    best_r = r
                break
    return best_r

def _objective(c):
    """Compute the value r^2/(2*pi) for coefficient vector c."""
    c0, c1, c2 = c
    q = _build_q(c0, c1, c2)
    r = _largest_sign_change_root(q)
    if r is None:
        return 1e12
    return (r * r) / (2.0 * np.pi)

def _local_refine(c_start, max_iter=500, tol=1e-10):
    """Local refinement using coordinate descent with line search."""
    c = np.array(c_start, dtype=float)
    best_val = _objective(c)
    
    for _ in range(max_iter):
        improved = False
        for i in range(3):
            # Try small perturbations in each coordinate
            for scale in [1e-3, 1e-4, 1e-5, 1e-6]:
                for direction in [-1, 1]:
                    c_new = c.copy()
                    c_new[i] += direction * scale
                    val = _objective(c_new)
                    if val < best_val - 1e-12:
                        c = c_new
                        best_val = val
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    return c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 10s safety margin
    
    # Seed
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed_c.copy()
    best_val = _objective(best_c)
    
    # Phase 1: Grid exploration around seed
    # Explore a range of values for each coefficient
    base = seed_c.copy()
    ranges = [
        (base[0] - 0.5, base[0] + 0.5, 8),
        (base[1] - 0.05, base[1] + 0.05, 8),
        (base[2] - 0.002, base[2] + 0.002, 8),
    ]
    
    # Do a coarse grid search
    grid_points = []
    for d0 in np.linspace(ranges[0][0], ranges[0][1], ranges[0][2]):
        for d1 in np.linspace(ranges[1][0], ranges[1][1], ranges[1][2]):
            for d2 in np.linspace(ranges[2][0], ranges[2][1], ranges[2][2]):
                if time.time() > deadline:
                    return best_c
                grid_points.append([d0, d1, d2])
    
    # Evaluate all grid points
    grid_results = []
    for pt in grid_points:
        if time.time() > deadline:
            break
        val = _objective(pt)
        grid_results.append((val, pt))
    
    grid_results.sort(key=lambda x: x[0])
    
    # Refine top candidates
    for val, pt in grid_results[:30]:
        if time.time() > deadline:
            break
        c_ref, v_ref = _local_refine(pt, max_iter=300)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref
    
    # Phase 2: Random restarts from diverse points
    np.random.seed(12345)
    for i in range(100):
        if time.time() > deadline:
            break
        # Generate random points in a broader region
        c_rand = np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.01, 0.01),
        ])
        c_ref, v_ref = _local_refine(c_rand, max_iter=200)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref
    
    # Phase 3: Final polish on best found
    if time.time() < deadline - 20:
        c_final, v_final = _local_refine(best_c, max_iter=1000, tol=1e-12)
        if v_final < best_val:
            best_val = v_final
            best_c = c_final
    
    return best_c.tolist()
