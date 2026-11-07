"""Root finder for 3 fixed 2-D systems: damped Newton + Broyden polish + grid scan."""
import numpy as np
import time


def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 115.0
    call_count = [0]
    CALL_CAP = 490

    def time_ok():
        return (time.time() - start_time) < time_limit

    def budget_ok():
        return call_count[0] < CALL_CAP and time_ok()

    def safe_eval_F(x):
        try:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                call_count[0] += 1
                return None
            result = np.asarray(F(x_arr), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None

    def finite_diff_J(x, Fx, h=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J

    def invertible(J):
        return abs(np.linalg.det(J)) > 1e-10

    def damped_newton(x_start, J, max_iter=40):
        x = np.asarray(x_start, dtype=float).copy()
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        n = np.linalg.norm(Fx)
        best_x, best_Fx, best_n = x.copy(), Fx.copy(), n
        if n <= 1e-8:
            return x, Fx, True
        for _ in range(max_iter):
            if not budget_ok():
                break
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            d = np.linalg.norm(dx)
            if not np.isfinite(d) or d < 1e-16:
                break
            step = 1.0
            for _ in range(24):
                x_new = x + step * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    break
                n_new = np.linalg.norm(Fx_new)
                if n_new < best_n:
                    best_x, best_Fx, best_n = x_new.copy(), Fx_new.copy(), n_new
                if n_new <= 1e-8:
                    return x_new, Fx_new, True
                if n_new < n:
                    x, Fx, n = x_new, Fx_new, n_new
                    break
                step *= 0.5
            else:
                break
        return best_x, best_Fx, best_n <= 1e-8

    def broyden(x_start, J, max_iter=30):
        x = np.asarray(x_start, dtype=float).copy()
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        n = np.linalg.norm(Fx)
        best_x, best_Fx, best_n = x.copy(), Fx.copy(), n
        if n <= 1e-8:
            return x, Fx, True
        for _ in range(max_iter):
            if not budget_ok():
                break
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            d = np.linalg.norm(dx)
            if not np.isfinite(d) or d < 1e-16:
                break
            if d > 10.0:
                dx *= 10.0 / d
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                x_new = x + 0.5 * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    break
            n_new = np.linalg.norm(Fx_new)
            if n_new < best_n:
                best_x, best_Fx, best_n = x_new.copy(), Fx_new.copy(), n_new
            if n_new <= 1e-8:
                return x_new, Fx_new, True
            s = x_new - x
            y = Fx_new - Fx
            ss = float(np.dot(s, s))
            if ss > 1e-16:
                J = J + np.outer(y - J @ s, s) / ss
            x, Fx = x_new, Fx_new
        return best_x, best_Fx, best_n <= 1e-8

    def try_point(x_start, J, max_iter):
        x, Fx, ok = broyden(x_start, J.copy(), max_iter)
        if ok:
            return x, Fx, True
        x2, Fx2, ok2 = damped_newton(x_start, J.copy(), max_iter)
        if ok2:
            return x2, Fx2, True
        n1 = np.linalg.norm(Fx) if Fx is not None else np.inf
        n2 = np.linalg.norm(Fx2) if Fx2 is not None else np.inf
        if n2 < n1:
            return x2, Fx2, False
        return x, Fx, False

    x0_arr = np.asarray(x0, dtype=float)
    scale = max(1.0, np.linalg.norm(x0_arr))

    x, Fx, ok = try_point(x0_arr, None, 40)
    if ok:
        return x
    best_x, best_Fx, best_n = x.copy(), Fx.copy(), np.linalg.norm(Fx) if Fx is not None else np.inf

    # One finite-difference Jacobian, reused across all restarts
    J = finite_diff_J(x0_arr, safe_eval_F(x0_arr))
    if J is None or not invertible(J):
        J = finite_diff_J(x0_arr, safe_eval_F(x0_arr), h=1e-4)
    if J is None or not invertible(J):
        J = np.eye(2)

    # Global grid scan (reuses J) to escape bad basins
    for ax in range(-3, 4):
        if not budget_ok():
            break
        for ay in range(-3, 4):
            if not budget_ok():
                break
            pt = np.array([ax, ay], dtype=float) * scale
            xg, Fg, okg = try_point(pt, J, 15)
            if okg:
                return xg
            if Fg is not None:
                ng = np.linalg.norm(Fg)
                if ng < best_n:
                    best_n, best_x, best_Fx = ng, xg.copy(), Fg.copy()

    # Polish the best point with a fresh local Jacobian
    if budget_ok() and best_Fx is not None and best_n > 1e-8:
        Jf = finite_diff_J(best_x, best_Fx)
        if Jf is None or not invertible(Jf):
            Jf = J
        xp, Fp, okp = broyden(best_x, Jf.copy(), 25)
        if okp:
            return xp
        if Fp is not None:
            np_ = np.linalg.norm(Fp)
            if np_ < best_n:
                best_x, best_Fx, best_n = xp.copy(), Fp.copy(), np_
        if budget_ok() and best_n > 1e-8:
            xq, Fq, okq = damped_newton(best_x, Jf.copy(), 25)
            if okq:
                return xq
            if Fq is not None:
                nq = np.linalg.norm(Fq)
                if nq < best_n:
                    best_x, best_Fx, best_n = xq.copy(), Fq.copy(), nq

    if best_Fx is not None and np.linalg.norm(best_Fx) <= 1e-8:
        return best_x
    return best_x
