import numpy as np
from scipy.special import lpmv, factorial
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    k = 10
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2 * np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0 / (4 * np.pi) * factorial(10 - m) / factorial(10 + m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    return Y, weights_2d

def _compute_f(coeffs, Y):
    """Compute f = sum_m c_m Y_m on the grid."""
    return np.tensordot(coeffs, Y, axes=(0, 0))

def _objective_value(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = _compute_f(coeffs, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(np.abs(f)**4 * w)**0.25
    return norm4 / norm2

def _power_iteration(c0, Y, weights_2d, max_iter=200, tol=1e-15):
    """
    Power iteration for the cubic eigenvalue problem.
    The EL equation: proj(f^3) = lambda * c, where f = sum c_m Y_m
    Since ||c||=1 (L2 orthonormal), we iterate: c <- proj(f^3) / ||proj(f^3)||
    """
    c = c0.copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c0
    c = c / nrm
    
    w = weights_2d
    prev_c = None
    
    for it in range(max_iter):
        f = _compute_f(c, Y)
        f3 = f**3
        # Project f^3 onto each harmonic: p_m = sum f3 * Y_m * w
        p = np.array([np.sum(f3 * Y[m] * w) for m in range(21)])
        
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-15:
            break
        
        new_c = p / p_norm
        
        if prev_c is not None:
            diff = np.linalg.norm(new_c - prev_c)
            if diff < tol:
                break
        prev_c = new_c
        c = new_c
    
    return c

def _newton_polish(c0, Y, weights_2d, max_iter=50, tol=1e-14):
    """
    Newton's method on the sphere to solve the KKT system:
    Minimize g(c) = sum f^4 * w subject to ||c||^2 = 1
    
    Gradient: grad_m = 4 * sum f^3 * Y_m * w
    Hessian: H_mn = 12 * sum f^2 * Y_m * Y_n * w
    
    KKT: [H c; c^T 0] [dc; nu] = [-grad; 0]
    Then project dc onto tangent space: dc_tan = dc - (c·dc)c
    Line search with backtracking.
    """
    c = c0.copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return c0
    c = c / nrm
    
    w = weights_2d
    n = 21
    
    # Precompute for efficiency
    for newton_iter in range(max_iter):
        f = _compute_f(c, Y)
        f2 = f**2
        f3 = f**3
        f4 = f**4
        
        # Gradient: grad_m = 4 * sum f^3 * Y_m * w
        grad = 4.0 * np.array([np.sum(f3 * Y[m] * w) for m in range(n)])
        
        # Hessian: H_mn = 12 * sum f^2 * Y_m * Y_n * w
        # Efficient computation using weighted Y
        sqrt_fw = np.sqrt(np.maximum(f2 * w, 0.0))
        weighted_Y = Y * sqrt_fw[None, :, :]
        H = 12.0 * np.einsum('mij,nij->mn', weighted_Y, weighted_Y)
        
        # KKT system
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = H
        A[:n, n] = c
        A[n, :n] = c
        b = np.zeros(n + 1)
        b[:n] = -grad
        
        try:
            sol = np.linalg.solve(A, b)
            delta_c = sol[:n]
        except np.linalg.LinAlgError:
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                delta_c = sol[:n]
            except:
                break
        
        # Project onto tangent space
        c_dot_dc = np.dot(c, delta_c)
        delta_tan = delta_c - c_dot_dc * c
        
        if np.linalg.norm(delta_tan) < tol:
            break
        
        # Line search
        f4_current = np.sum(f4 * w)
        alpha = 1.0
        improved = False
        for ls_iter in range(40):
            c_trial = c + alpha * delta_tan
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-15:
                alpha *= 0.5
                continue
            c_trial = c_trial / c_trial_norm
            
            f_trial = _compute_f(c_trial, Y)
            f4_trial = np.sum(f_trial**4 * w)
            
            if f4_trial < f4_current - 1e-15:
                c = c_trial
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
    
    return c

def _generate_starts(n_random=40, seed=42):
    """Generate diverse starting points."""
    rng = np.random.RandomState(seed)
    starts = []
    
    # Structured starts
    # 1. All equal weights
    v = np.ones(21) / np.sqrt(21)
    starts.append(v)
    
    # 2. Only m=0
    v = np.zeros(21); v[10] = 1.0
    starts.append(v)
    
    # 3. Only |m|=10
    v = np.zeros(21); v[0] = 1.0/np.sqrt(2); v[20] = 1.0/np.sqrt(2)
    starts.append(v)
    
    # 4. High m emphasis (|m| >= 8)
    v = np.zeros(21)
    for m in [8, 9, 10]:
        v[10+m] = 1.0/np.sqrt(6)
        v[10-m] = 1.0/np.sqrt(6)
    starts.append(v)
    
    # 5. Low m emphasis (|m| <= 2)
    v = np.zeros(21)
    for m in [0, 1, 2]:
        v[10+m] = 1.0/np.sqrt(5)
        if m > 0:
            v[10-m] = 1.0/np.sqrt(5)
    nrm = np.linalg.norm(v)
    starts.append(v / nrm)
    
    # 6. Cosine weighting
    v = np.array([np.cos(np.pi * m / 20) for m in range(-10, 11)])
    starts.append(v / np.linalg.norm(v))
    
    # 7. Sine weighting (high m)
    v = np.array([np.sin(np.pi * (abs(m) + 1) / 11) for m in range(-10, 11)])
    starts.append(v / np.linalg.norm(v))
    
    # 8. Alternating signs
    v = np.array([(1.0 if m % 2 == 0 else -1.0) for m in range(-10, 11)])
    starts.append(v / np.linalg.norm(v))
    
    # 9. Only even m
    v = np.zeros(21)
    for m in range(-10, 11, 2):
        v[10 + m] = 1.0
    starts.append(v / np.linalg.norm(v))
    
    # 10. Only odd m
    v = np.zeros(21)
    for m in range(-10, 11, 2):
        v[10 + m] = 1.0
    starts.append(v / np.linalg.norm(v))
    
    # 11. Only positive m
    v = np.zeros(21)
    for m in range(0, 11):
        v[10 + m] = 1.0
    starts.append(v / np.linalg.norm(v))
    
    # 12. Only negative m
    v = np.zeros(21)
    for m in range(-10, 0):
        v[10 + m] = 1.0
    starts.append(v / np.linalg.norm(v))
    
    # Random starts
    for _ in range(n_random):
        v = rng.randn(21)
        starts.append(v / np.linalg.norm(v))
    
    # Sparse random starts
    for _ in range(15):
        n_active = rng.randint(2, 12)
        active_idx = rng.choice(21, size=n_active, replace=False)
        v = np.zeros(21)
        v[active_idx] = rng.randn(n_active)
        starts.append(v / np.linalg.norm(v))
    
    # Sign-structured random
    for _ in range(10):
        v = rng.choice([-1, 1], size=21) * np.abs(rng.randn(21))
        starts.append(v / np.linalg.norm(v))
    
    return starts

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses power iteration + Newton polishing with multiple diverse initializations.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Precompute harmonics and weights
    Y, weights_2d = _build_harmonics()
    
    best_coeffs = None
    best_score = np.inf
    
    def _update_best(c):
        nonlocal best_coeffs, best_score
        if c is None:
            return
        if not np.all(np.isfinite(c)):
            return
        s = _objective_value(c, Y, weights_2d)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Generate all starting points
    starts = _generate_starts(n_random=40, seed=42)
    
    # Phase 1: Power iteration from all starts
    for i, d in enumerate(starts):
        if time.time() - start_time > deadline - 180:
            break
        
        c_fp = _power_iteration(d, Y, weights_2d, max_iter=300, tol=1e-15)
        _update_best(c_fp)
    
    # Phase 2: Newton polish from best power iteration results
    if time.time() - start_time < deadline - 90:
        # Re-run power iteration from top candidates and polish with Newton
        # First, collect unique local minima from power iteration
        candidates = []
        seen = set()
        for d in starts:
            if time.time() - start_time > deadline - 90:
                break
            c_fp = _power_iteration(d, Y, weights_2d, max_iter=200, tol=1e-14)
            # Use rounded tuple as key to identify duplicates
            key = tuple(np.round(c_fp, 8))
            if key not in seen:
                seen.add(key)
                candidates.append(c_fp)
        
        # Newton polish each unique candidate
        for c_cand in candidates:
            if time.time() - start_time > deadline - 30:
                break
            c_newton = _newton_polish(c_cand, Y, weights_2d, max_iter=80, tol=1e-14)
            _update_best(c_newton)
    
    # Phase 3: Local search around best solution
    if best_coeffs is not None and time.time() - start_time < deadline - 30:
        for i in range(20):
            if time.time() - start_time > deadline - 15:
                break
            # Small random perturbation
            x0 = best_coeffs + np.random.randn(21) * 0.05
            x0 = x0 / np.linalg.norm(x0)
            
            # Power iteration from perturbed point
            c_fp = _power_iteration(x0, Y, weights_2d, max_iter=100, tol=1e-14)
            c_newton = _newton_polish(c_fp, Y, weights_2d, max_iter=50, tol=1e-14)
            _update_best(c_newton)
    
    # Phase 4: Final high-precision polish
    if best_coeffs is not None and time.time() - start_time < deadline - 10:
        c_final = _newton_polish(best_coeffs, Y, weights_2d, max_iter=100, tol=1e-15)
        _update_best(c_final)
    
    # Validate and return
    if best_coeffs is None:
        return np.ones(21)
    
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs if max_abs > 0 else np.ones(21)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)
    
    if not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    
    return best_coeffs.astype(np.float64)
