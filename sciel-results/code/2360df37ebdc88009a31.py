import numpy as np
import scipy.linalg
import scipy.integrate
import time

# Precompute the eigen-decomposition of the 10x10 5-point stencil Laplacian
# This is a fixed matrix, so we compute it once at module load time

def _build_heat_L():
    N = 100
    L = np.zeros((N, N))
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            L[idx, idx] = -4.0
            if i > 0:
                L[idx, (i - 1) * 10 + j] = 1.0
            if i < 9:
                L[idx, (i + 1) * 10 + j] = 1.0
            if j > 0:
                L[idx, i * 10 + (j - 1)] = 1.0
            if j < 9:
                L[idx, i * 10 + (j + 1)] = 1.0
    return L

_L = _build_heat_L()
# Use eigh for symmetric matrix for better numerical stability
_L_eigvals, _L_eigvecs = np.linalg.eigh(_L)

def _rk4_fallback(f, t_span, y0, steps=2000):
    h = (t_span[1] - t_span[0]) / steps
    t, y = t_span[0], np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

def _solve_linear_ode_eig(A, y0, dt):
    """Solve y' = A*y using eigendecomposition. More stable than expm for moderate N."""
    # A is symmetric (Laplacian), so use eigh for better numerical stability
    try:
        eigvals, eigvecs = scipy.linalg.eigh(A)
        # y_end = V @ diag(exp(eigvals * dt)) @ V^{-1} @ y0
        # Since V is orthogonal: V^{-1} = V.T
        coeffs = eigvecs.T @ y0
        scaled_coeffs = coeffs * np.exp(eigvals * dt)
        y_end = eigvecs @ scaled_coeffs
        return y_end
    except Exception:
        # Fallback to expm if eigh fails
        expA = scipy.linalg.expm(A * dt)
        return expA @ y0

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Handle trivial case
    if np.linalg.norm(y0) < 1e-15:
        return y0.copy()

    start_time = time.time()
    deadline = start_time + 280  # 300s limit, leave 20s margin
    
    # Strategy 1: Analytic solution using known 10x10 heat equation structure
    # The 5-point stencil Laplacian is fixed; verify with minimal f-calls.
    if N == 100:
        try:
            # Verify: check if f(t0, y0) ≈ L @ y0 (1 f-call)
            # We assume the form y' = k * L * y. 
            # If k=1, then f(t0, y0) = L @ y0.
            # If k != 1, we can estimate k.
            
            v_actual = f(t0, y0)
            v_pred = _L @ y0
            
            # Estimate k using projection: k = <v_actual, v_pred> / <v_pred, v_pred>
            # This handles the case where the heat equation is scaled (diffusivity != 1)
            dot_vp = np.dot(v_actual, v_pred)
            dot_pp = np.dot(v_pred, v_pred)
            
            if dot_pp < 1e-30:
                # y0 is in the null space or very small
                return y0.copy()
                
            k = dot_vp / dot_pp
            
            # Check residual to confirm the model
            residual = v_actual - k * v_pred
            err = np.linalg.norm(residual)
            denom = 1e-12 + np.linalg.norm(v_actual)
            
            if err / denom < 1e-8:
                # Confirmed: y' = k * L * y. Solve analytically with 0 additional f-calls.
                # Use precomputed eigen decomposition of L
                # y(t) = exp(k * L * dt) @ y0
                # = V @ diag(exp(k * eigvals * dt)) @ V.T @ y0
                coeffs = _L_eigvecs.T @ y0
                scaled_coeffs = coeffs * np.exp(k * _L_eigvals * dt)
                y_end = _L_eigvecs @ scaled_coeffs
                return y_end
            else:
                # Not the standard scaled Laplacian. 
                # Try to identify the operator with 2 more probes? 
                # Or just fall back to general methods.
                # For robustness, let's try a second probe to see if it's linear but different matrix.
                # However, the task specifies "fixed 10x10 two-dimensional heat equation".
                # If it's not k*L, it might be a different discretization or non-linear.
                # Given the contract, it's almost certainly k*L.
                # Let's try a random probe to check linearity and structure.
                rng = np.random.default_rng(42)
                y_rand = rng.standard_normal(N)
                v_actual2 = f(t0, y_rand)
                v_pred2 = _L @ y_rand
                dot_vp2 = np.dot(v_actual2, v_pred2)
                dot_pp2 = np.dot(v_pred2, v_pred2)
                if dot_pp2 > 1e-30:
                    k2 = dot_vp2 / dot_pp2
                    residual2 = v_actual2 - k2 * v_pred2
                    err2 = np.linalg.norm(residual2)
                    denom2 = 1e-12 + np.linalg.norm(v_actual2)
                    
                    if err2 / denom2 < 1e-8 and abs(k - k2) < 1e-6:
                         # Consistent scaling k
                         coeffs = _L_eigvecs.T @ y0
                         scaled_coeffs = coeffs * np.exp(k * _L_eigvals * dt)
                         y_end = _L_eigvecs @ scaled_coeffs
                         return y_end
                    elif err2 / denom2 < 1e-8:
                         # Different k? Unlikely for autonomous linear ODE, but possible if f depends on t?
                         # Contract says y' = f(t, y). If autonomous, k is constant.
                         # If non-autonomous, analytic solution is harder.
                         # But heat equation is typically autonomous.
                         pass
        except Exception:
            pass

    # Strategy 2: General linear operator identification (fallback for non-standard linear ODEs)
    if N <= 200:
        try:
            A = np.zeros((N, N))
            for i in range(N):
                e_i = np.zeros(N)
                e_i[i] = 1.0
                A[:, i] = f(t0, e_i)
            
            # Quick verification with 1 probe
            rng = np.random.default_rng(123)
            y_rand = rng.standard_normal(N)
            v_actual = f(t0, y_rand)
            v_pred = A @ y_rand
            err = np.linalg.norm(v_actual - v_pred)
            denom = 1e-12 + np.linalg.norm(v_pred)
            
            if err / denom < 1e-8:
                y_end = _solve_linear_ode_eig(A, y0, dt)
                return y_end
        except Exception:
            pass

    # Strategy 3: SciPy Implicit Solvers
    if time.time() > deadline:
        return _rk4_fallback(f, t_span, y0, 5000)
        
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    if time.time() > deadline:
        return _rk4_fallback(f, t_span, y0, 5000)

    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Strategy 4: Fallback RK4
    return _rk4_fallback(f, t_span, y0, 5000)
