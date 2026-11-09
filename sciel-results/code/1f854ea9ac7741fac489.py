import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.time_limit = 320.0

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()

        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            self.calls += 1
            return func(np.clip(x, -5.0, 5.0))

        n = self.dim
        best_x = np.zeros(n)
        best_fx = float('inf')

        rng = np.random.default_rng(123)
        starts = [np.zeros(n)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, n))

        # Phase 1: Multi-start Powell to locate the valley
        for x0 in starts:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
            remaining = self.budget - self.calls
            if remaining < 10:
                break
            try:
                max_iter = min(150, remaining // (n + 2))
                if max_iter < 1:
                    break
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'xatol': 1e-7}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x.copy()
            except StopIteration:
                break
            except Exception:
                continue

        # Phase 2: BFGS-style polish with finite-difference gradient
        remaining = self.budget - self.calls
        if remaining > 20 and best_x is not None and time.time() - self.start_time < self.time_limit - 10:
            try:
                x = best_x.copy()
                fx = best_fx
                # Build initial Hessian inverse approximation
                H = np.eye(n)
                h = 1e-6
                max_polish_iters = min(30, remaining // (2 * n + 1))
                for it in range(max_polish_iters):
                    if self.calls >= self.budget:
                        break
                    if time.time() - self.start_time > self.time_limit - 2:
                        break
                    # Finite-difference gradient
                    grad = np.zeros(n)
                    for i in range(n):
                        if self.calls >= self.budget:
                            break
                        if time.time() - self.start_time > self.time_limit - 2:
                            break
                        xp = x.copy()
                        xm = x.copy()
                        xp[i] += h
                        xm[i] -= h
                        xp = np.clip(xp, -5.0, 5.0)
                        xm = np.clip(xm, -5.0, 5.0)
                        fp = eval_func(xp)
                        fm = eval_func(xm)
                        grad[i] = (fp - fm) / (2 * h)
                    if self.calls >= self.budget:
                        break
                    # Check gradient magnitude
                    if np.linalg.norm(grad, ord=np.inf) < 1e-10:
                        break
                    # BFGS search direction: d = -H * grad
                    d = -H @ grad
                    # Line search (backtracking)
                    alpha = 1.0
                    for ls_iter in range(30):
                        if self.calls >= self.budget:
                            break
                        if time.time() - self.start_time > self.time_limit - 2:
                            break
                        x_new = np.clip(x + alpha * d, -5.0, 5.0)
                        f_new = eval_func(x_new)
                        if f_new < fx - 1e-4 * alpha * np.dot(grad, d):
                            break
                        alpha *= 0.5
                    else:
                        break
                    # Update Hessian inverse (BFGS update)
                    s = x_new - x
                    g_new = np.zeros(n)
                    for i in range(n):
                        if self.calls >= self.budget:
                            break
                        if time.time() - self.start_time > self.time_limit - 2:
                            break
                        xp = x_new.copy()
                        xm = x_new.copy()
                        xp[i] += h
                        xm[i] -= h
                        xp = np.clip(xp, -5.0, 5.0)
                        xm = np.clip(xm, -5.0, 5.0)
                        fp = eval_func(xp)
                        fm = eval_func(xm)
                        g_new[i] = (fp - fm) / (2 * h)
                    if self.calls >= self.budget:
                        break
                    y = g_new - grad
                    sy = np.dot(s, y)
                    if sy > 1e-12:
                        Hy = H @ y
                        H = H + (s @ s.T) / sy - (Hy @ Hy.T) / (sy * sy)
                    x = x_new
                    fx = f_new
                    grad = g_new
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        # Phase 3: Final tight Powell polish if budget remains
        remaining = self.budget - self.calls
        if remaining > 10 and best_x is not None and time.time() - self.start_time < self.time_limit - 5:
            try:
                max_iter = min(40, remaining // (n + 1))
                if max_iter >= 1:
                    res = minimize(
                        eval_func,
                        best_x,
                        method='Powell',
                        options={'maxiter': max_iter, 'ftol': 1e-14, 'xatol': 1e-12}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        if best_x is None or not np.all(np.isfinite(best_x)):
            best_x = np.zeros(n)
            if self.calls < self.budget:
                best_fx = eval_func(best_x)
            else:
                best_fx = 0.0

        return best_x, best_fx
