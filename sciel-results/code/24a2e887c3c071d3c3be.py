import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap(h):
    """Compute max over shifts of mean(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Build Fourier basis: DC + first K harmonics (real and imaginary parts)
    K = 10  # number of harmonics -> 2K+1 = 21 basis functions
    n_basis = 2 * K + 1
    idx = np.arange(N)
    basis = np.zeros((n_basis, N))
    basis[0, :] = 1.0 / np.sqrt(N)  # DC component
    for k in range(1, K + 1):
        basis[2*k - 1, :] = np.cos(2 * np.pi * k * idx / N) / np.sqrt(N)
        basis[2*k, :] = np.sin(2 * np.pi * k * idx / N) / np.sqrt(N)

    def coeffs_to_h(c):
        """Map coefficient vector to h, then project to [0,1] with sum N/2."""
        h = c @ basis
        # Project to [0,1]
        h = np.clip(h, 0, 1)
        # Enforce sum constraint: adjust to make sum exactly N/2
        diff = N / 2.0 - h.sum()
        if diff > 1e-12:
            # Need to add mass: add to elements below 1
            room = 1.0 - h
            total_room = room.sum()
            if total_room > 1e-12:
                h += room * (diff / total_room)
            else:
                # All at 1, can't add more (shouldn't happen)
                pass
        elif diff < -1e-12:
            # Need to remove mass: remove from elements above 0
            space = h.copy()
            total_space = space.sum()
            if total_space > 1e-12:
                h -= space * (-diff / total_space)
            else:
                pass
        # Final clip
        h = np.clip(h, 0, 1)
        # Final sum correction (tiny)
        residual = N / 2.0 - h.sum()
        if abs(residual) > 1e-14:
            # Distribute residual evenly among elements not at boundary
            interior = (h > 1e-6) & (h < 1 - 1e-6)
            if interior.any():
                h[interior] += residual / interior.sum()
                h = np.clip(h, 0, 1)
        return h

    def logsumexp_overlap(c, temperature=50.0):
        """Smoothed max overlap using log-sum-exp over all shifts."""
        h = coeffs_to_h(c)
        inv_h = 1.0 - h
        # Compute all N overlap values
        overlaps = np.zeros(N)
        for s in range(N):
            overlaps[s] = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        # Log-sum-exp for smooth max
        m = overlaps.max()
        lse = m + np.log(np.sum(np.exp((overlaps - m) / temperature)) * temperature / N)
        # Actually, standard logsumexp: log(sum(exp(x_i/T))) * T gives smooth max
        # Let's use: T * log(sum(exp(overlaps/T))) - T*log(N)
        # This approximates max(overlaps) as T -> 0
        lse_val = temperature * (np.log(np.sum(np.exp((overlaps - m) / temperature))) + m / temperature - np.log(N))
        return lse_val

    def objective_and_grad(c):
        """Objective (smoothed max overlap) and its gradient w.r.t. coefficients."""
        h = coeffs_to_h(c)
        inv_h = 1.0 - h
        # Compute all overlaps
        overlaps = np.zeros(N)
        for s in range(N):
            overlaps[s] = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        
        T = 50.0  # temperature for log-sum-exp
        m = overlaps.max()
        exps = np.exp((overlaps - m) / T)
        weights = exps / exps.sum()  # softmax weights
        
        # Objective
        obj = T * (np.log(np.sum(exps)) + (m - T * np.log(N)) / T)
        # Simplify: obj = T * log(sum(exp((ov-m)/T))) + m - T*log(N)
        obj = T * np.log(np.sum(exps)) + m - T * np.log(N)
        
        # Gradient: d(obj)/d(h_i)
        # obj = T * log(sum_s exp(ov_s/T)) + m - T*log(N)
        # d(obj)/d(h_i) = sum_s weights_s * d(ov_s)/d(h_i) + d(m)/d(h_i)
        # For simplicity, ignore d(m)/d(h_i) (it's a selection function)
        # d(ov_s)/d(h_i) = inv_h[(i+s)%N] - h[(i-s)%N] * ... 
        # Actually ov_s = sum_i h[i] * inv_h[(i+s)%N]
        # d(ov_s)/d(h_i) = inv_h[(i+s)%N]
        # But h also appears in inv_h = 1-h, so:
        # ov_s = sum_i h[i] * (1 - h[(i+s)%N])
        # d(ov_s)/d(h_i) = (1 - h[(i+s)%N]) - h[(i-s)%N]  [from the second term where i appears in h[(i'+s)%N] with i'+s = i, i.e., i' = i-s]
        
        grad_h = np.zeros(N)
        for s in range(N):
            w = weights[s]
            # d(ov_s)/d(h_i) = (1 - h[(i+s)%N]) - h[(i-s)%N]
            d_ov = (1.0 - h) - np.roll(h, s)  # element i: (1-h[i]) - h[i-s] ... wait
            # Let me re-derive: ov_s = sum_i h[i]*(1-h[(i+s)%N])
            # d/dh_i [h[i]*(1-h[(i+s)%N])] = (1-h[(i+s)%N])
            # d/dh_i [h[j]*(1-h[(j+s)%N])] for j != i: 
            #   = -h[j] * d/dh_i[h[(j+s)%N]] = -h[j] if (j+s)%N == i, i.e., j = (i-s)%N
            # So d(ov_s)/d(h_i) = (1-h[(i+s)%N]) - h[(i-s)%N]
            d_ov_s = (1.0 - np.roll(h, -s)) - np.roll(h, s)
            grad_h += w * d_ov_s
        
        # Chain rule: grad_c = basis.T @ grad_h
        grad_c = basis.T @ grad_h
        
        return obj, grad_c

    def projected_gradient_descent(c0, lr=0.01, n_steps=5000, tol=1e-10):
        """Projected gradient descent in coefficient space."""
        c = c0.copy()
        best_c = c.copy()
        best_obj = objective_and_grad(c)[0]
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            obj, grad = objective_and_grad(c)
            if obj < best_obj:
                best_obj = obj
                best_c = c.copy()
            
            # Adaptive learning rate
            g_norm = np.linalg.norm(grad)
            if g_norm < tol:
                break
            
            # Gradient step
            c_new = c - lr * grad
            
            # Simple box constraint on coefficients (keep them bounded)
            c_new = np.clip(c_new, -5, 5)
            
            # Check if this is better (using actual objective)
            obj_new = objective_and_grad(c_new)[0]
            if obj_new < obj:
                c = c_new
            else:
                # Halve learning rate
                lr *= 0.5
                c_new = c - lr * grad
                c_new = np.clip(c_new, -5, 5)
                obj_new = objective_and_grad(c_new)[0]
                if obj_new < obj:
                    c = c_new
                else:
                    lr *= 0.5
                    if lr < 1e-12:
                        break
        
        return best_c, best_obj

    # Initial best: constant 0.5
    h_const = np.full(N, 0.5)
    best_h = h_const.copy()
    best_val = _score(h_const)  # Should be 0.25 (max overlap of 0.5 with 0.5)

    # Multiple random restarts in coefficient space
    np.random.seed(42)
    n_restarts = 8
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initial coefficients (small perturbation from zero)
        c0 = np.random.randn(n_basis) * 0.5
        # Set DC component to give mean 0.5: basis[0] = 1/sqrt(N), so c[0] = 0.5*sqrt(N)
        c0[0] = 0.5 * np.sqrt(N)
        
        # Run projected gradient descent
        c_opt, obj_opt = projected_gradient_descent(c0, lr=0.05, n_steps=3000)
        
        # Evaluate the actual (non-smoothed) objective
        h_opt = coeffs_to_h(c_opt)
        val_opt = _score(h_opt)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
        
        # Also try with different temperatures / more aggressive exploration
        if time.time() > deadline:
            break
        c0b = np.random.randn(n_basis) * 1.0
        c0b[0] = 0.5 * np.sqrt(N)
        c_opt2, obj_opt2 = projected_gradient_descent(c0b, lr=0.03, n_steps=2000)
        h_opt2 = coeffs_to_h(c_opt2)
        val_opt2 = _score(h_opt2)
        if val_opt2 < best_val:
            best_val = val_opt2
            best_h = h_opt2.copy()
    
    # Final refinement: use scipy SLSQP on the full h vector starting from best_h
    if time.time() < deadline - 20:
        def slsqp_obj(h_flat):
            h = h_flat.reshape(N)
            return _max_overlap(h)
        
        def slsqp_grad(h_flat):
            h = h_flat.reshape(N)
            inv_h = 1.0 - h
            overlaps = np.zeros(N)
            for s in range(N):
                overlaps[s] = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
            # Use the argmax shift for gradient (subgradient)
            s_star = np.argmax(overlaps)
            # d(ov_s)/d(h_i) = (1-h[(i+s)%N]) - h[(i-s)%N]
            grad_h = (1.0 - np.roll(h, -s_star)) - np.roll(h, s_star)
            return grad_h.reshape(-1)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: h.sum() - N/2.0, 'jac': lambda h: np.ones(N)}
        ]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(slsqp_obj, best_h.copy(), jac=slsqp_grad,
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.success or res.fun < best_val:
                h_sl = np.clip(res.x.reshape(N), 0, 1)
                # Fix sum
                diff = N/2.0 - h_sl.sum()
                if abs(diff) > 1e-10:
                    interior = (h_sl > 1e-6) & (h_sl < 1 - 1e-6)
                    if interior.any():
                        h_sl[interior] += diff / interior.sum()
                        h_sl = np.clip(h_sl, 0, 1)
                val_sl = _score(h_sl)
                if val_sl < best_val:
                    best_val = val_sl
                    best_h = h_sl
        except Exception:
            pass
    
    # Final validation and cleanup
    best_h = np.clip(best_h, 0, 1)
    # Ensure exact sum
    diff = N / 2.0 - best_h.sum()
    if abs(diff) > 1e-12:
        interior = (best_h > 1e-8) & (best_h < 1 - 1e-8)
        if interior.any():
            best_h[interior] += diff / interior.sum()
            best_h = np.clip(best_h, 0, 1)
        else:
            # Fallback: uniform adjustment
            best_h += diff / N
            best_h = np.clip(best_h, 0, 1)
    
    # Final safety check
    if abs(best_h.sum() - N/2.0) > 1e-6:
        # Reset to constant if something went wrong
        best_h = np.full(N, 0.5)
    
    assert abs(best_h.sum() - N/2.0) < 1e-6
    assert best_h.min() >= -1e-8 and best_h.max() <= 1.0 + 1e-8
    
    return best_h
# EVOLVE-BLOCK-END
