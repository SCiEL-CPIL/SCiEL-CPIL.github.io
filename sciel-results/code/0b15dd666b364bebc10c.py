# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _safe_evaluate(x, y, form_id, p):
    """Safely evaluate a candidate form, returning inf on failure."""
    try:
        n = len(x)
        if n == 0:
            return np.array([])
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (10,):
            return np.full(n, np.inf)
        if not np.all(np.isfinite(p)):
            return np.full(n, np.inf)
        
        a, b, c, d, e, f, g, h, i, j = p
        
        if form_id == 0:
            # Linear + quadratic
            pred = a*x + b*y + c + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + i*x**2*y + j*x*y**2
        elif form_id == 1:
            # Trigonometric
            pred = a*np.sin(x) + b*np.sin(y) + c*np.cos(x) + d*np.cos(y) + e*np.sin(x*y) + f*np.cos(x*y) + g*x + h*y + i*x*y + j
        elif form_id == 2:
            # Exponential
            pred = a*np.exp(x) + b*np.exp(y) + c*np.exp(-x) + d*np.exp(-y) + e*np.exp(x*y) + f*np.exp(-x*y) + g*x + h*y + i*x*y + j
        elif form_id == 3:
            # Mixed poly-trig
            pred = a*x + b*y + c + d*np.sin(x) + e*np.sin(y) + f*x**2 + g*y**2 + h*x*y + i*np.cos(x) + j*np.cos(y)
        elif form_id == 4:
            # Higher order polynomial
            pred = a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + i*x**2*y + j*x*y**2
        elif form_id == 5:
            # Rational-like (avoid division by zero with softplus)
            sx = np.log1p(np.exp(x))
            sy = np.log1p(np.exp(y))
            pred = a*x + b*y + c*sx + d*sy + e*np.tanh(x) + f*np.tanh(y) + g*np.tanh(x*y) + h*x**2 + i*y**2 + j
        elif form_id == 6:
            # Fourier-like
            pred = a*np.sin(x) + b*np.sin(2*x) + c*np.sin(y) + d*np.sin(2*y) + e*np.sin(x+y) + f*np.sin(x-y) + g*x + h*y + i*x*y + j
        elif form_id == 7:
            # Log-sigmoid + poly
            tx = np.tanh(x)
            ty = np.tanh(y)
            pred = a*tx + b*ty + c*tx*ty + d*x + e*y + f*x**2 + g*y**2 + h*x*y + i*tx**2 + j*ty**2
        elif form_id == 8:
            # Exp of scaled args
            pred = a*np.exp(0.5*x) + b*np.exp(0.5*y) + c*np.exp(-0.5*x) + d*np.exp(-0.5*y) + e*x + f*y + g*x*y + h*x**2 + i*y**2 + j
        elif form_id == 9:
            # Cubic with cross terms
            pred = a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + i*x*y*(x+y) + j*(x**2 - y**2)
        elif form_id == 10:
            # Sine/cosine of sum/diff
            pred = a*np.sin(x+y) + b*np.sin(x-y) + c*np.cos(x+y) + d*np.cos(x-y) + e*x + f*y + g*x*y + h*x**2 + i*y**2 + j
        elif form_id == 11:
            # Softplus features
            sp_x = np.log1p(np.exp(2*x))
            sp_y = np.log1p(np.exp(2*y))
            pred = a*sp_x + b*sp_y + c*x + d*y + e*x**2 + f*y**2 + g*x*y + h*np.tanh(x) + i*np.tanh(y) + j
        elif form_id == 12:
            # Pure polynomial up to degree 3 with all terms
            pred = a + b*x + c*y + d*x**2 + e*y**2 + f*x*y + g*x**3 + h*y**3 + i*x**2*y + j*x*y**2
        elif form_id == 13:
            # Gaussian-like bumps
            pred = a*np.exp(-x**2) + b*np.exp(-y**2) + c*np.exp(-(x+y)**2) + d*np.exp(-(x-y)**2) + e*x + f*y + g*x*y + h*x**2 + i*y**2 + j
        elif form_id == 14:
            # Tanh-based
            pred = a*np.tanh(x) + b*np.tanh(y) + c*np.tanh(x+y) + d*np.tanh(x-y) + e*x + f*y + g*x*y + h*x**2 + i*y**2 + j
        else:
            return np.full(n, np.inf)
        
        if not np.all(np.isfinite(pred)):
            return np.full(n, np.inf)
        return pred
    except Exception:
        return np.full(len(x), np.inf)


def _fit_form(x, y, form_id, max_time=10.0):
    """Fit a single form using BFGS, return (pred, mse) or (None, inf)."""
    n = len(x)
    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15
    
    best_mse = np.inf
    best_pred = None
    
    # Try multiple initializations
    inits = [
        np.ones(10),
        np.zeros(10),
        np.random.randn(10) * 0.1,
        np.random.randn(10) * 1.0,
        np.array([1, 1, 1, 0.1, 0.1, 0.1, 0, 0, 0, 0], dtype=np.float64),
    ]
    
    t_start = time.time()
    
    for init in inits:
        if time.time() - t_start > max_time:
            break
        
        def objective(p):
            pred = _safe_evaluate(x, y, form_id, p)
            if not np.all(np.isfinite(pred)):
                return 1e30
            return np.mean((pred - y)**2)
        
        try:
            result = minimize(objective, init, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
            if result.fun < best_mse and np.isfinite(result.fun):
                best_mse = result.fun
                best_pred = _safe_evaluate(x, y, form_id, result.x)
        except Exception:
            continue
    
    if best_pred is None:
        return None, np.inf
    return best_pred, best_mse


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Main entry point: try multiple forms, return best prediction."""
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n = len(x)
    if n == 0:
        return np.array([])
    
    # Quick baseline: use the provided params with a simple form
    baseline_pred = params[0]*x + params[1]*y + params[2] + params[3]*x**2 + params[4]*y**2 + params[5]*x*y
    if not np.all(np.isfinite(baseline_pred)):
        baseline_pred = np.zeros(n)
    baseline_mse = np.mean((baseline_pred - y)**2)
    
    best_mse = baseline_mse
    best_pred = baseline_pred.copy()
    
    # Try each form with time budget
    forms_to_try = [0, 3, 4, 1, 5, 7, 2, 6, 8, 9, 10, 11, 12, 13, 14]
    
    for fid in forms_to_try:
        if time.time() > deadline:
            break
        
        # Allocate remaining time proportionally
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        per_form_time = min(remaining / max(1, len(forms_to_try) - forms_to_try.index(fid)), 15.0)
        
        pred, mse = _fit_form(x, y, fid, max_time=per_form_time)
        
        if pred is not None and mse < best_mse:
            best_mse = mse
            best_pred = pred
        
        # Check if we've found a near-perfect fit
        if best_mse < 1e-10:
            break
    
    # Final safety check
    if not np.all(np.isfinite(best_pred)):
        best_pred = np.zeros(n)
    
    return best_pred
# EVOLVE-BLOCK-END
