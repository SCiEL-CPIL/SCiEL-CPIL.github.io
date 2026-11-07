"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_value(c0, c1, c2):
    """Compute the value = r^2 / (2*pi) for given coefficients."""
    # Hermite polynomial values at 0
    H0_0 = 1
    H4_0 = 12
    H8_0 = 10080
    H12_0 = 10321920
    
    # c3 so that g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Build q(x) = g(x)/x^2 as polynomial in y = x^2
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 4608x^6 + 30240x^4 - 60480x^2 + 10080
    # H12 = 4096x^12 - 147456x^10 + 2257920x^8 - 15930720x^6 + 47056320x^4 - 54432000x^2 + 10321920
    
    # q(x) = g(x)/x^2, so coefficients of x^0, x^2, ..., x^10 in g give coefficients of q
    # g(x) = sum_{k=0}^{6} a_k * x^{2k}, with a_0 = 0 (since g(0)=0)
    # q(x) = sum_{k=0}^{6} a_k * x^{2k-2} = sum_{j=0}^{5} a_{j+1} * x^{2j}
    
    # Coefficients of g(x) for x^{2k}:
    # k=0: c0*1 + c1*12 + c2*10080 + c3*10321920 = 0 (by construction)
    # k=1: c1*(-48) + c2*(-60480) + c3*(-54432000)
    # k=2: c1*16 + c2*30240 + c3*47056320
    # k=3: c2*(-4608) + c3*(-15930720)
    # k=4: c2*256 + c3*2257920
    # k=5: 0
    # k=6: c3*4096
    
    a1 = c1 * (-48) + c2 * (-60480) + c3 * (-54432000)
    a2 = c1 * 16 + c2 * 30240 + c3 * 47056320
    a3 = c2 * (-4608) + c3 * (-15930720)
    a4 = c2 * 256 + c3 * 2257920
    a5 = 0
    a6 = c3 * 4096
    
    # q(x) = a1 + a2*x^2 + a3*x^4 + a4*x^6 + a5*x^8 + a6*x^10
    # In terms of y = x^2: Q(y) = a1 + a2*y + a3*y^2 + a4*y^3 + a5*y^4 + a6*y^5
    
    # Find roots of Q(y) = 0 where y >= 0 (since y = x^2)
    # Then r = sqrt(largest such root where sign changes)
    
    coeffs = np.array([a6, a5, a4, a3, a2, a1])  # highest degree first
    roots_y = np.roots(coeffs)
    
    # Filter real, non-negative roots
    real_roots = []
    for r_y in roots_y:
        if abs(r_y.imag) < 1e-10 and r_y.real >= -1e-10:
            real_roots.append(max(0.0, r_y.real))
    
    if not real_roots:
        return float('inf')
    
    # For each candidate root y, check if Q changes sign there
    # Q(y) = a6*y^5 + a5*y^4 + a4*y^3 + a3*y^2 + a2*y + a1
    def Q(y):
        return a6*y**5 + a5*y**4 + a4*y**3 + a3*y**2 + a2*y + a1
    
    best_r = 0
    for y_root in real_roots:
        if y_root < 1e-12:
            continue
        # Check sign change: evaluate Q at y_root - eps and y_root + eps
        eps = 1e-8 * max(1.0, abs(y_root))
        q_minus = Q(y_root - eps)
        q_plus = Q(y_root + eps)
        if q_minus * q_plus < 0:
            # Sign change, this is a valid root
            r = np.sqrt(y_root)
            if r > best_r:
                best_r = r
    
    if best_r == 0:
        return float('inf')
    
    return best_r**2 / (2 * np.pi)


def construct():
    start_time = time.time()
    deadline = start_time + 180.0
    
    # Seed coefficients
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_value = _compute_value(*seed)
    best_coeffs = seed.copy()
    best_value = seed_value
    
    # Try local optimization with different starting points
    random_state = np.random.RandomState(42)
    
    # Generate multiple starting points
    starts = [seed.copy()]
    # Add perturbations around seed
    for _ in range(20):
        perturb = random_state.randn(3) * 0.1
        starts.append(seed + perturb)
    # Some broader explorations
    for _ in range(10):
        starts.append(random_state.randn(3) * 0.5)
    
    for start in starts:
        if time.time() > deadline - 10.0:
            break
        
        def objective(x):
            return _compute_value(x[0], x[1], x[2])
        
        try:
            result = minimize(objective, start, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        def objective(x):
            return _compute_value(x[0], x[1], x[2])
        
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-16})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
