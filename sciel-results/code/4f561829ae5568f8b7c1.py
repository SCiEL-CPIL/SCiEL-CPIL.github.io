"""Direct geometric construction for logarithmic energy on sphere."""
import numpy as np
from math import fsum, log
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
SAFETY = 0.85


def _energy(pts):
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return 1e18
            terms.append(-log(d))
    return fsum(terms)


def _cartesian_energy_and_grad(X):
    """Energy and gradient for points on sphere (Cartesian coords, projected)."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(X)
    diff = X[:, None, :] - X[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e-30)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    E = -np.sum(np.log(r)) * 0.5
    # dE/dX_i = sum_{j!=i} (X_i - X_j) / r_ij^2
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    dE_dX = np.sum(diff * inv_r2[:, :, None], axis=1)
    # Project gradient to tangent space: g_tan = g - (g·x)x
    g_tan = dE_dX - np.sum(dE_dX * X, axis=1, keepdims=True) * X
    return E, g_tan


def _riemannian_cg(X0, max_iter=2000, tol=1e-12, time_limit=None):
    """Riemannian conjugate gradient on the sphere with geodesic line search."""
    X = X0.copy()
    n = len(X)
    
    # Compute initial gradient
    E_prev, g = _cartesian_energy_and_grad(X)
    p = -g.copy()
    
    best_E = E_prev
    best_X = X.copy()
    g_prev = None

    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Geodesic line search
        alpha = 0.1
        best_alpha = 0.0
        best_f = E_prev
        best_X_new = X.copy()
        best_g_new = g.copy()
        
        for ls_iter in range(40):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Move along geodesic: X_new = cos(alpha)*X + sin(alpha)*p/||p||
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-15:
                break
            p_unit = p / p_norm
            X_new = np.cos(alpha) * X + np.sin(alpha) * p_unit
            X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
            
            f_new, g_new = _cartesian_energy_and_grad(X_new)
            
            # Armijo condition: f_new <= f_curr - c * alpha * ||g||^2
            # Simplified: just accept if it improves, or use a simple backtracking
            if f_new < best_f:
                best_f = f_new
                best_alpha = alpha
                best_X_new = X_new
                best_g_new = g_new
                # If improvement is sufficient, we can stop early or continue to find a better alpha
                # Standard CG usually accepts the first sufficient decrease or does Wolfe.
                # Here we just take the best found in a small window or break on first good.
                # Let's break on first improvement to save time, or do a few more.
                # For robustness, let's just take the best of a few tries.
                # Actually, standard backtracking starts large and shrinks.
                # This loop increases alpha? No, alpha is fixed at 0.1 in this snippet?
                # The original code had alpha=0.01 fixed and looped 30 times but alpha didn't change?
                # Ah, the original code had a bug: alpha was not updated in the loop.
                # Let's implement proper backtracking.
                pass # Placeholder for logic restructure
            
            # Proper Backtracking:
            # We need to reset alpha or shrink it.
            # Let's rewrite the inner loop properly in the replacement.
            # Since I can't easily restructure the whole function in a single small edit without risk,
            # I will replace the whole function body in the next step if needed, 
            # but for now, let's just fix the UnboundLocalError by initializing variables.
            
            # The original code's line search was flawed (alpha constant).
            # I will replace the entire function with a corrected version.
            break # Exit this dummy loop to avoid confusion, real logic below.

        # --- Real Line Search Logic ---
        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4
        g_norm_sq = np.sum(g**2)
        
        accepted = False
        for ls_iter in range(30):
            if time_limit is not None and time.time() > time_limit:
                break
                
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-15:
                break
                
            p_unit = p / p_norm
            X_new = np.cos(alpha) * X + np.sin(alpha) * p_unit
            X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
            
            f_new, g_new = _cartesian_energy_and_grad(X_new)
            
            # Armijo condition
            if f_new <= E_prev - c1 * alpha * g_norm_sq:
                accepted = True
                break
            alpha *= 0.5
            
        if not accepted:
            # If no step accepted, try a very small step or break
            if alpha < 1e-6:
                break
            # Use the last computed X_new if it improved, else stay
            if f_new < E_prev:
                accepted = True
            else:
                break # Stuck, break

        if accepted:
            X = X_new
            g = g_new
            E_prev = f_new
            
            if E_prev < best_E:
                best_E = E_prev
                best_X = X.copy()
        
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Update direction (Polak-Ribiere)
        if it > 0 and g_prev is not None:
            beta = np.sum(g * (g - g_prev)) / (np.sum(g_prev * g_prev) + 1e-30)
            beta = max(0, beta)
            p = -g + beta * p
        else:
            p = -g
        
        g_prev = g.copy()
    
    return best_X


def _opt_cartesian(X0, maxiter=1000):
    """Optimize in Cartesian coords with projection."""
    try:
        res = minimize(_cartesian_energy_and_grad, X0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x
    except Exception:
        return X0


def construct():
    start = time.time()
    dl = start + DEADLINE * SAFETY

    def _fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1 - z ** 2))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    best = None
    best_E = 1e18

    # Try multiple initializations
    np.random.seed(42)
    initializations = [_fibonacci_spiral()]
    for _ in range(5):
        X = np.random.randn(N, 3)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        initializations.append(X)

    for X0 in initializations:
        if time.time() > dl:
            break
        Xopt = _opt_cartesian(X0, maxiter=800)
        Xopt /= np.linalg.norm(Xopt, axis=1, keepdims=True)
        E = _energy(Xopt)
        if E < best_E:
            best_E = E
            best = Xopt.copy()

    if best is None:
        best = _fibonacci_spiral()

    # Final refinement with Riemannian CG if time permits
    if time.time() < dl - 10.0:
        Xref = _riemannian_cg(best, max_iter=1000, tol=1e-13, time_limit=dl - 5.0)
        Xref /= np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best = Xref
            best_E = Eref

    # Additional L-BFGS polish if time remains
    if time.time() < dl - 3.0:
        Xref2 = _opt_cartesian(best, maxiter=300)
        Xref2 /= np.linalg.norm(Xref2, axis=1, keepdims=True)
        Eref2 = _energy(Xref2)
        if Eref2 < best_E:
            best = Xref2

    return best
