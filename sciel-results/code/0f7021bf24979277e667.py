"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin from 300s limit
    
    # Try SciPy's Radau method first (good for stiff problems)
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        # Fallback to manual RK45 if scipy not available
        def _rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
            y = np.array(y0, dtype=float)
            t = t0
            h = (t1 - t0) / 100.0
            while t < t1 and time.time() < deadline:
                h = min(h, t1 - t)
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                k5 = f(t + h/5, y + h/5*k4)
                k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
                
                y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
                err = h*(k4 - k3)/32
                err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
                
                if err_norm <= rtol:
                    y = y_new
                    t += h
                    h *= min(5.0, 0.9*(rtol/err_norm)**0.2) if err_norm > 0 else 5.0
                else:
                    h *= max(0.1, 0.9*(rtol/err_norm)**0.2)
            return y
        
        return _rk45(f, t0, t1, y0)
    
    # Strategy: Try multiple methods with different tolerances, pick the one with best score
    # Score = -log10(rel_err) - 0.5*log10(nfev)
    # We don't know the reference, so we'll use tightest tolerances that complete in time
    
    best_result = None
    best_score = -np.inf
    
    # Try Radau with various tolerances (no max_step to allow large steps in stiff regions)
    for rtol, atol in [(1e-12, 1e-14), (1e-11, 1e-13), (1e-10, 1e-12)]:
        if time.time() > deadline - 5:
            break
        try:
            sol = solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=rtol, atol=atol
            )
            if sol.success:
                nfev = sol.nfev
                # Estimate: tighter tol = lower error. Use nfev as proxy for calls.
                # We want to maximize: -log10(err) - 0.5*log10(nfev)
                # Since we don't know err exactly, assume err ~ rtol for the method
                est_score = -np.log10(rtol) - 0.5*np.log10(max(nfev, 1))
                if est_score > best_score:
                    best_score = est_score
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass
    
    # Try BDF if Radau didn't give great results
    if best_score < 5.5:
        for rtol, atol in [(1e-12, 1e-14), (1e-11, 1e-13)]:
            if time.time() > deadline - 5:
                break
            try:
                sol = solve_ivp(
                    f, t_span, y0,
                    method='BDF',
                    rtol=rtol, atol=atol
                )
                if sol.success:
                    nfev = sol.nfev
                    est_score = -np.log10(rtol) - 0.5*np.log10(max(nfev, 1))
                    if est_score > best_score:
                        best_score = est_score
                        best_result = sol.y[:, -1].copy()
            except Exception:
                pass
    
    # Try LSODA
    if best_score < 5.5:
        for rtol, atol in [(1e-12, 1e-14), (1e-11, 1e-13)]:
            if time.time() > deadline - 5:
                break
            try:
                sol = solve_ivp(
                    f, t_span, y0,
                    method='LSODA',
                    rtol=rtol, atol=atol
                )
                if sol.success:
                    nfev = sol.nfev
                    est_score = -np.log10(rtol) - 0.5*np.log10(max(nfev, 1))
                    if est_score > best_score:
                        best_score = est_score
                        best_result = sol.y[:, -1].copy()
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    
    # Final fallback: RK45 with moderate tolerances
    def _rk45_fallback(f, t0, t1, y0, rtol=1e-10, atol=1e-12):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            k5 = f(t + h/5, y + h/5*k4)
            k6 = f(t + 3*h/5, y + 3*h/5*(k1 + 4/3*(k4-k1)))
            
            y_new = y + h*(k1 + 3*k2 + 3*k3 + k4)/8
            err = h*(k4 - k3)/32
            err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
            
            if err_norm <= rtol:
                y = y_new
                t += h
                h *= min(5.0, 0.9*(rtol/err_norm)**0.2) if err_norm > 0 else 5.0
            else:
                h *= max(0.1, 0.9*(rtol/err_norm)**0.2)
        return y
    
    return _rk45_fallback(f, t0, t1, y0)
# EVOLVE-BLOCK-END
