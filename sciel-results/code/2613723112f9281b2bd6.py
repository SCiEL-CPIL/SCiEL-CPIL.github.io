"""Root finder for 3 fixed two-dimensional systems using RBF-surrogate + Broyden polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses an RBF-surrogate-guided basin finder followed by a Secant-Step Hybrid Broyden polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    def time_ok():
        return (time.time() - start_time) < deadline

    x = np.array(x0, dtype=float).copy()
    n = 2

    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = float(np.linalg.norm(Fx))

    if not np.all(np.isfinite(Fx)):
        for _ in range(5):
            x = x + np.random.normal(0, 0.1, n)
            Fx = np.array(F(x), dtype=float)
            if np.all(np.isfinite(Fx)):
                break
        norm_Fx = float(np.linalg.norm(Fx))
        if not np.all(np.isfinite(Fx)):
            return x

    if norm_Fx <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = norm_Fx

    # --- RBF Surrogate Basin Finder ---
    # We will build a small RBF network to approximate ||F(x)||^2 and minimize it.
    # This helps identify a good basin if x0 is far from the root.
    
    # Collect initial points for the surrogate
    points = [x.copy()]
    values = [norm_Fx**2]
    
    # Add a few perturbed points to give the surrogate some structure
    for _ in range(4):
        xp = x + np.random.normal(0, 0.1, n)
        Fp = np.array(F(xp), dtype=float)
        if np.all(np.isfinite(Fp)):
            np_Fp = float(np.linalg.norm(Fp))
            points.append(xp)
            values.append(np_Fp**2)
            if np_Fp < best_norm:
                best_x = xp.copy()
                best_norm = np_Fp

    # RBF parameters
    rbf_centers = np.array(points)
    rbf_values = np.array(values)
    # Heuristic width: based on spread of points
    if len(points) > 1:
        dists = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dists.append(np.linalg.norm(rbf_centers[i] - rbf_centers[j]))
        sigma = np.mean(dists) * 0.5 if dists else 0.1
        if sigma < 1e-4:
            sigma = 0.1
    else:
        sigma = 0.1

    def rbf_residual(xq):
        """Approximate ||F(xq)||^2 using RBF."""
        # Phi(xq) = sum_i exp(-||xq - c_i||^2 / (2*sigma^2)) * v_i
        # This is a crude approximation, but good enough for basin identification.
        res = 0.0
        for i in range(len(rbf_centers)):
            d2 = np.dot(xq - rbf_centers[i], xq - rbf_centers[i])
            res += np.exp(-d2 / (2 * sigma**2)) * rbf_values[i]
        return res

    def rbf_grad(xq):
        """Gradient of the RBF residual."""
        g = np.zeros(n)
        for i in range(len(rbf_centers)):
            d2 = np.dot(xq - rbf_centers[i], xq - rbf_centers[i])
            weight = np.exp(-d2 / (2 * sigma**2)) * rbf_values[i]
            # d/dx exp(-d2/2s2) = exp(...) * (-1/s2) * (x - c_i)
            g += weight * (-1.0 / (sigma**2)) * (xq - rbf_centers[i])
        return g

    # Minimize the surrogate to find a better starting point
    x_surr = x.copy()
    # Simple gradient descent on the surrogate
    for _ in range(20):
        if not time_ok():
            break
        g = rbf_grad(x_surr)
        # Adaptive step size
        step = 0.1
        # Check if step improves surrogate
        x_new_surr = x_surr - step * g
        if rbf_residual(x_new_surr) < rbf_residual(x_surr):
            x_surr = x_new_surr
        else:
            step *= 0.5
            x_new_surr = x_surr - step * g
            if rbf_residual(x_new_surr) < rbf_residual(x_surr):
                x_surr = x_new_surr
            else:
                break

    # If the surrogate suggests a better point (lower residual), use it
    # We don't trust the surrogate's absolute value, but relative improvement is useful.
    # However, we must be careful not to move to a region where F is undefined or worse.
    # Let's evaluate F at x_surr.
    Fx_surr = np.array(F(x_surr), dtype=float)
    if np.all(np.isfinite(Fx_surr)):
        norm_surr = float(np.linalg.norm(Fx_surr))
        if norm_surr < best_norm:
            best_x = x_surr.copy()
            best_norm = norm_surr
            x = x_surr.copy()
            Fx = Fx_surr
            norm_Fx = norm_surr

    # --- Broyden Polish ---
    # Now use the standard Broyden method from the best point found.
    
    def compute_jacobian_fwd(x, Fx, h=1e-5):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            if not np.all(np.isfinite(Fp)):
                return None
            J[:, j] = (Fp - Fx) / h
        return J

    def compute_jacobian_central(x, h=1e-5):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            Fm = np.array(F(x - e), dtype=float)
            if not np.all(np.isfinite(Fp)) or not np.all(np.isfinite(Fm)):
                return None
            J[:, j] = (Fp - Fm) / (2 * h)
        return J

    # Initialize Jacobian
    J = compute_jacobian_fwd(x, Fx, 1e-5)
    if J is None:
        J = compute_jacobian_central(x, 1e-5)
    if J is None:
        return best_x

    max_iter_broyden = 40
    reanchor_every = 5

    for it in range(max_iter_broyden):
        if not time_ok():
            break

        # Re-anchor
        if it > 0 and it % reanchor_every == 0:
            J_new = compute_jacobian_central(x, 1e-6)
            if J_new is not None:
                J = J_new

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Line search
        alpha = 1.0
        max_ls = 10
        ls_success = False
        for _ in range(max_ls):
            x_new = x + alpha * dx
            Fx_new = np.array(F(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            norm_Fx_new = float(np.linalg.norm(Fx_new))
            
            if norm_Fx_new <= 1e-8:
                return x_new

            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new

            # Sufficient decrease
            if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                s = x_new - x
                y = Fx_new - Fx
                s_dot_s = float(np.dot(s, s))
                if s_dot_s > 1e-30:
                    J += np.outer(y - J @ s, s) / s_dot_s
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                ls_success = True
                break
            
            alpha *= 0.5

        if not ls_success:
            break

        if norm_Fx <= 1e-8:
            return x

    # --- Final Polish ---
    # If we are close, do a few high-precision Newton steps
    if best_norm < 1e-4:
        x = best_x
        Fx = np.array(F(x), dtype=float)
        norm_Fx = float(np.linalg.norm(Fx))
        for _ in range(10):
            if not time_ok():
                break
            if norm_Fx <= 1e-8:
                return x
            
            J = compute_jacobian_central(x, 1e-7)
            if J is None:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            x_new = x + dx
            Fx_new = np.array(F(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            norm_Fx_new = float(np.linalg.norm(Fx_new))
            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new
            
            if norm_Fx_new < norm_Fx:
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
            else:
                break

    return best_x
# EVOLVE-BLOCK-END
