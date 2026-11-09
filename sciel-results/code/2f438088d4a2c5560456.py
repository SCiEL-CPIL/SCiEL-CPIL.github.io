import numpy as np
from scipy.optimize import minimize
import time

def _compute_entropies(P):
    n = 10
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys)
    
    def proj_entropy(vals, n_bins, offset=0):
        p_flat = P.flatten()
        v_flat = (vals + offset).flatten()
        counts = np.bincount(v_flat, weights=p_flat, minlength=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        if len(probs) == 0: return 0.0
        return -np.sum(probs * np.log2(probs))
        
    H_d = proj_entropy(X - Y, 19, 9)
    H_0 = proj_entropy(X, 10, 0)
    H_1 = proj_entropy(X + Y, 19, 0)
    H_2 = proj_entropy(X + 2*Y, 28, 0)
    
    return H_d, H_0, H_1, H_2

def _compute_ratio(P):
    H_d, H_0, H_1, H_2 = _compute_entropies(P)
    M = max(H_0, H_1, H_2)
    if M < 1e-9:
        return 0.0
    return H_d / M

def _parametric_P(params):
    """
    P(x,y) proportional to exp(-alpha*x^2 - beta*y^2 - gamma*x*y)
    params = [alpha, beta, gamma]
    """
    alpha, beta, gamma = params
    xs = np.arange(10)
    ys = np.arange(10)
    X, Y = np.meshgrid(xs, ys)
    
    log_probs = -alpha * X**2 - beta * Y**2 - gamma * X * Y
    log_probs -= np.max(log_probs)
    P = np.exp(log_probs)
    s = P.sum()
    if s < 1e-300:
        return np.ones((10, 10)) / 100.0
    P /= s
    return P

def _parametric_objective(params):
    P = _parametric_P(params)
    return -_compute_ratio(P)

def _parametric_gradient(params):
    """Numerical gradient for the 3D parametric space"""
    eps = 1e-6
    grad = np.zeros(3)
    base_obj = _parametric_objective(params)
    for i in range(3):
        params_plus = params.copy()
        params_plus[i] += eps
        obj_plus = _parametric_objective(params_plus)
        grad[i] = (obj_plus - base_obj) / eps
    return grad

def _refine_full(P_init):
    """Refine a good 10x10 distribution using full 100D optimization"""
    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)
    
    def objective(z):
        P = softmax(z).reshape(10, 10)
        return -_compute_ratio(P)
    
    def gradient(z):
        P = softmax(z).reshape(10, 10)
        H_d, H_0, H_1, H_2 = _compute_entropies(P)
        M = max(H_0, H_1, H_2)
        if M < 1e-9:
            return np.zeros(100)
        
        n = 10
        xs = np.arange(n)
        ys = np.arange(n)
        X, Y = np.meshgrid(xs, ys)
        
        def proj_grad(vals, n_bins, offset=0):
            p_flat = P.flatten()
            v_flat = (vals + offset).flatten()
            counts = np.bincount(v_flat, weights=p_flat, minlength=n_bins)
            counts_safe = np.where(counts > 0, counts, 1.0)
            log_probs = np.log2(counts_safe)
            idx = (vals + offset)
            grad_vals = -1.0/np.log(2) * (log_probs[idx] + 1.0)
            return grad_vals

        G_d = proj_grad(X - Y, 19, 9)
        G_0 = proj_grad(X, 10, 0)
        G_1 = proj_grad(X + Y, 19, 0)
        G_2 = proj_grad(X + 2*Y, 28, 0)
        
        Hs = [H_0, H_1, H_2]
        Gs = [G_0, G_1, G_2]
        max_idx = np.argmax(Hs)
        G_m = Gs[max_idx]
        
        grad_P_f = (M * G_d - H_d * G_m) / (M * M)
        
        P_flat = P.flatten()
        grad_P_flat = grad_P_f.flatten()
        
        dot = np.dot(P_flat, grad_P_flat)
        grad_z = P_flat * (grad_P_flat - dot)
        
        return grad_z
    
    z_init = np.log(P_init.flatten() + 1e-300)
    
    try:
        res = minimize(
            objective, 
            z_init, 
            method='L-BFGS-B', 
            jac=gradient,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        P_refined = softmax(res.x).reshape(10, 10)
        return P_refined
    except Exception:
        return P_init

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # safety margin
    
    best_score = -np.inf
    best_P = None
    
    # Phase 1: Search in 3D parametric space
    # Try a grid of initial points in (alpha, beta, gamma) space
    np.random.seed(42)
    
    param_starts = []
    
    # Systematic grid
    for alpha in np.linspace(-1.5, 1.5, 8):
        for beta in np.linspace(-1.5, 1.5, 8):
            for gamma in np.linspace(-1.5, 1.5, 8):
                param_starts.append([alpha, beta, gamma])
    
    # Add random samples
    for _ in range(50):
        a = np.random.uniform(-2, 2)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-2, 2)
        param_starts.append([a, b, c])
    
    # Optimize each starting point in 3D space
    for i, p_init in enumerate(param_starts):
        if time.time() > deadline - 120:
            break
            
        try:
            res = minimize(
                _parametric_objective,
                p_init,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-8}
            )
            
            P_candidate = _parametric_P(res.x)
            score = _compute_ratio(P_candidate)
            
            if score > best_score:
                best_score = score
                best_P = P_candidate
        except Exception:
            pass
    
    # Phase 2: Refine the best parametric solution in full 100D space
    if best_P is not None and time.time() < deadline - 60:
        P_refined = _refine_full(best_P)
        score_refined = _compute_ratio(P_refined)
        
        if score_refined > best_score:
            best_score = score_refined
            best_P = P_refined
    
    # Phase 3: Try multiple refinement starts from different parametric solutions
    if time.time() < deadline - 30:
        # Generate several diverse parametric solutions and refine each
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            
            a = np.random.uniform(-1.5, 1.5)
            b = np.random.uniform(-1.5, 1.5)
            c = np.random.uniform(-1.5, 1.5)
            
            P_try = _parametric_P([a, b, c])
            
            # Quick optimize in 3D
            try:
                res3 = minimize(
                    _parametric_objective,
                    [a, b, c],
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'fatol': 1e-10}
                )
                P_opt3 = _parametric_P(res3.x)
            except:
                P_opt3 = P_try
            
            score3 = _compute_ratio(P_opt3)
            
            if score3 > best_score * 0.9:  # Only refine promising ones
                P_ref = _refine_full(P_opt3)
                score_ref = _compute_ratio(P_ref)
                
                if score_ref > best_score:
                    best_score = score_ref
                    best_P = P_ref
    
    # Fallback
    if best_P is None:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = np.clip(best_P, 0, None)
        s = best_P.sum()
        if s <= 1e-9:
            best_P = np.ones((10, 10)) / 100.0
        else:
            best_P /= s
    
    return best_P
