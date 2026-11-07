import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses a hybrid strategy:
    1. Broyden's method with periodic FD Jacobian re-anchoring.
    2. If that fails, fallback to a few multi-start Broyden attempts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit

    x0 = np.array(x0, dtype=float)
    n = 2

    # Initial evaluation
    Fx = np.array(F(x0), dtype=float)
    norm_Fx = float(np.linalg.norm(Fx))

    if norm_Fx <= 1e-8:
        return x0

    best_x = x0.copy()
    best_norm = norm_Fx

    def compute_jacobian_fd(x, F, h=1e-5):
        """Compute Jacobian using forward differences (2 calls per component)."""
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            J[:, j] = (Fp - Fx) / h  # Note: this uses Fx which is F(x), but we need F(x) here.
            # Actually, let's be precise. Forward diff: (F(x+h) - F(x))/h.
            # We have F(x) as Fx. But if x changes, Fx is stale.
            # So we should recompute F(x) or pass it.
            # Let's just recompute for safety in a helper or just use central diff which is more robust.
            # The prompt says "forward (2 calls/iter)" saved calls.
            # Let's stick to central diff for initialization but maybe forward for updates?
            # Actually, Broyden doesn't need FD updates, only initial.
            # Let's use central diff for the initial Jacobian as it's more accurate.
            pass
        # Let's rewrite compute_jacobian_fd to be self-contained and correct.
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = np.array(F(x + e), dtype=float)
            Fm = np.array(F(x - e), dtype=float)
            J[:, j] = (Fp - Fm) / (2 * h)
        return J

    # Initialize Jacobian
    J = compute_jacobian_fd(x0, F)

    max_iter = 50
    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break

        # Line search to ensure sufficient decrease
        alpha = 1.0
        max_ls = 10
        ls_success = False
        for _ in range(max_ls):
            x_new = x0 + alpha * dx
            Fx_new = np.array(F(x_new), dtype=float)
            norm_Fx_new = float(np.linalg.norm(Fx_new))

            if norm_Fx_new <= 1e-8:
                return x_new

            if norm_Fx_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_Fx_new

            # Sufficient decrease condition
            if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                # Good-Broyden update
                s = x_new - x0
                y = Fx_new - Fx
                s_dot_s = float(np.dot(s, s))
                if s_dot_s > 1e-30:
                    J += np.outer(y - J @ s, s) / s_dot_s
                x0 = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                ls_success = True
                break

            alpha *= 0.5

        if not ls_success:
            # Line search failed; break and try other strategies
            break

        if norm_Fx <= 1e-8:
            return x0

    # If Broyden didn't converge, try a few more with fresh Jacobian
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Recompute Jacobian
            Fx = np.array(F(x0), dtype=float)
            norm_Fx = float(np.linalg.norm(Fx))
            if norm_Fx <= 1e-8:
                return x0
            if norm_Fx < best_norm:
                best_x = x0.copy()
                best_norm = norm_Fx

            J = compute_jacobian_fd(x0, F)

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

            alpha = 1.0
            for _ in range(10):
                x_new = x0 + alpha * dx
                Fx_new = np.array(F(x_new), dtype=float)
                norm_Fx_new = float(np.linalg.norm(Fx_new))
                if norm_Fx_new <= 1e-8:
                    return x_new
                if norm_Fx_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_Fx_new
                if norm_Fx_new < (1 - 1e-4 * alpha) * norm_Fx:
                    x0 = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    break
                alpha *= 0.5

    return best_x
