"""Combined optimizer for BBOB f12 (Bent cigar) with surrogate-guided (1+1)-ES."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Start at origin (common for BBOB)
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Keep a sliding window of recent evaluations for the quadratic model
        history_x = [best_x.copy()]
        history_f = [best_f]
        max_hist = 30

        # Adaptive step size for (1+1)-ES
        sigma = 2.0
        sigma_cap = 15.0
        growth = 1.25
        decay = 0.85

        # Phase transition: after ~250 calls, switch to polish mode
        phase1_end = 250

        for i in range(budget - 1):
            if calls >= budget or time.time() > deadline:
                break

            # Update history with current best
            history_x.append(best_x.copy())
            history_f.append(best_f)
            if len(history_x) > max_hist:
                history_x.pop(0)
                history_f.pop(0)

            # Phase transition
            if i >= phase1_end:
                sigma = 0.05
                growth = 1.1
                decay = 0.7
                sigma_cap = 1.0

            # Try to use surrogate model if we have enough data
            used_surrogate = False
            n_hist = len(history_x)
            n_params = 1 + dim + dim * (dim + 1) // 2  # 21 for dim=5

            if n_hist >= n_params and i > 20:
                X = np.array(history_x)
                F = np.array(history_f)

                # Build design matrix for quadratic features
                M = np.ones((n_hist, n_params))
                M[:, 1:dim + 1] = X
                col = dim + 1
                for ii in range(dim):
                    for jj in range(ii, dim):
                        M[:, col] = X[:, ii] * X[:, jj]
                        col += 1

                # Solve least squares
                try:
                    coeffs, _, rank, _ = np.linalg.lstsq(M, F, rcond=None)

                    if rank >= n_params - 1:
                        # Extract gradient and Hessian
                        c1 = coeffs[1:1 + dim]

                        # Reconstruct C from coeffs
                        C = np.zeros((dim, dim))
                        col = dim + 1
                        for ii in range(dim):
                            for jj in range(ii, dim):
                                val = coeffs[col]
                                C[ii, jj] = val
                                if ii != jj:
                                    C[jj, ii] = val
                                col += 1

                        # Check if Hessian is positive definite (local minimum)
                        eigvals = np.linalg.eigvalsh(C)

                        if eigvals[0] > 1e-6:  # Positive definite -> local min
                            # Solve C * delta = -c1/2 for the step
                            # Actually, grad f = c1 + 2 C x = 0 => C x = -c1/2
                            # So x_new = -C^{-1} c1 / 2
                            try:
                                x_pred = -np.linalg.solve(C, c1) / 2.0
                                # Clip to box
                                x_pred = np.clip(x_pred, -5, 5)

                                # Only take the step if it's not too far
                                dist = np.linalg.norm(x_pred - best_x)
                                if dist < sigma * 5:
                                    fy = func(x_pred)
                                    calls += 1

                                    # Add to history
                                    history_x.append(x_pred.copy())
                                    history_f.append(fy)
                                    if len(history_x) > max_hist:
                                        history_x.pop(0)
                                        history_f.pop(0)

                                    if fy < best_f:
                                        best_x = x_pred.copy()
                                        best_f = fy
                                        used_surrogate = True
                                        sigma *= 0.8  # reduce sigma on success
                                    else:
                                        sigma *= 0.9
                                else:
                                    # Step too far, use gradient descent fallback
                                    grad = c1 + 2 * C @ best_x
                                    grad_norm = np.linalg.norm(grad)
                                    if grad_norm > 1e-12:
                                        y = best_x - 0.5 * grad / grad_norm * sigma
                                        y = np.clip(y, -5, 5)
                                        fy = func(y)
                                        calls += 1

                                        history_x.append(y.copy())
                                        history_f.append(fy)
                                        if len(history_x) > max_hist:
                                            history_x.pop(0)
                                            history_f.pop(0)

                                        if fy < best_f:
                                            best_x = y.copy()
                                            best_f = fy
                                            used_surrogate = True
                                            sigma *= 0.8
                                        else:
                                            sigma *= 0.9
                            except np.linalg.LinAlgError:
                                pass
                        else:
                            # Not positive definite, use gradient descent
                            grad = c1 + 2 * C @ best_x
                            grad_norm = np.linalg.norm(grad)
                            if grad_norm > 1e-12:
                                y = best_x - 0.5 * grad / grad_norm * sigma
                                y = np.clip(y, -5, 5)
                                fy = func(y)
                                calls += 1

                                history_x.append(y.copy())
                                history_f.append(fy)
                                if len(history_x) > max_hist:
                                    history_x.pop(0)
                                    history_f.pop(0)

                                if fy < best_f:
                                    best_x = y.copy()
                                    best_f = fy
                                    used_surrogate = True
                                    sigma *= 0.8
                                else:
                                    sigma *= 0.9
                except Exception:
                    pass

            if not used_surrogate:
                # Standard (1+1)-ES step
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                # Add to history
                history_x.append(y.copy())
                history_f.append(fy)
                if len(history_x) > max_hist:
                    history_x.pop(0)
                    history_f.pop(0)

                if fy < best_f:
                    best_x = y.copy()
                    best_f = fy
                    sigma *= growth
                else:
                    sigma *= decay

            sigma = np.clip(sigma, 1e-6, sigma_cap)

        return best_x, best_f
