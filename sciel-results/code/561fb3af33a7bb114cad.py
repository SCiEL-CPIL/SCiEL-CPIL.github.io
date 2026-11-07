"""Surrogate-assisted optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Evaluate a Sobol space-filling sequence (128 points) to build a
           coarse model of the landscape. This gives strong early AOCC progress
           by quickly locating the best basin among the 21 peaks.
  Phase 2: Iterative surrogate-guided refinement. Build an IDW-based surrogate
           from all evaluations, predict the minimum location, and evaluate
           there. Repeat until budget is exhausted or deadline.
  Phase 3: L-BFGS-B local refinement from the best surrogate point and a few
           diverse starting points to squeeze out the final convergence.
  A wall-clock deadline guard (320 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Surrogate-assisted optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return float(func(x))

        best = {"x": None, "f": np.inf}

        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = np.array(x, dtype=float)

        rng = np.random.default_rng(12345)

        # Storage for all evaluations (for surrogate)
        X_eval = []
        F_eval = []

        def record(x, f):
            X_eval.append(np.array(x, dtype=float))
            F_eval.append(float(f))
            update_best(x, f)

        # ---------------- Phase 1: Sobol space-filling initialization ----------------
        # Use scipy's Sobol sequence for good space-filling properties
        from scipy.stats import qmc

        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        # 128 points for initial coarse model
        n_sobol = 128
        sobol_points = sobol.random(n_sobol)
        # Map [0,1]^n to [-5,5]^n
        sobol_points = sobol_points * 10.0 - 5.0

        for i in range(n_sobol):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break
            try:
                fi = fwrap(sobol_points[i])
                record(sobol_points[i], fi)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0
                record(x0, best["f"])
            except:
                return (None, np.inf)

        # ---------------- Phase 2: Surrogate-guided iterative refinement ----------------
        # Use IDW (inverse distance weighted) surrogate to guide next evaluations
        # The surrogate predicts f(x) and we also use an uncertainty measure
        # to balance exploitation vs exploration.

        def idw_predict(x, X, F, power=2.0, eps=1e-6):
            """IDW interpolation with uncertainty estimate."""
            d = np.linalg.norm(X - x, axis=1)
            d = np.maximum(d, eps)
            w = 1.0 / (d ** power)
            pred = np.sum(w * F) / np.sum(w)
            # Uncertainty: variance of weighted values
            var = np.sum(w * (F - pred) ** 2) / np.sum(w)
            # Scale uncertainty by average distance (farther = more uncertain)
            avg_d = np.sum(w * d) / np.sum(w)
            unc = np.sqrt(var) + 0.5 * avg_d * (np.max(F) - np.min(F)) / (np.max(F) - np.min(F) + 1e-8)
            return pred, unc

        # Number of surrogate-guided evaluations
        surrogate_budget = min(200, budget - 150)
        n_surrogate_steps = 0

        while n_surrogate_steps < surrogate_budget:
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break

            if len(X_eval) < 3:
                # Not enough data for surrogate; use random
                x_new = rng.uniform(-5.0, 5.0, size=n)
            else:
                X_arr = np.array(X_eval)
                F_arr = np.array(F_eval)
                f_range = np.max(F_arr) - np.min(F_arr) + 1e-8

                # Generate candidate points: mix of exploitation and exploration
                candidates = []

                # Exploitation: perturb best point
                for scale in [0.1, 0.3, 0.5]:
                    cand = best["x"] + scale * rng.standard_normal(n)
                    cand = np.clip(cand, -5.0, 5.0)
                    candidates.append(cand)

                # Exploration: random points
                for _ in range(5):
                    cand = rng.uniform(-5.0, 5.0, size=n)
                    candidates.append(cand)

                # Surrogate-guided: find point with lowest predicted value
                # or high uncertainty
                grid_n = 20  # per dimension for surrogate search
                grid = np.linspace(-5.0, 5.0, grid_n)
                # Use a coarser grid in 5D (20^5 = 3.2M is too much)
                # Instead, use random sampling for surrogate minimization
                n_surrogate_candidates = 500
                surrogate_cands = rng.uniform(-5.0, 5.0, size=(n_surrogate_candidates, n))

                preds = np.zeros(n_surrogate_candidates)
                uncs = np.zeros(n_surrogate_candidates)
                for k in range(n_surrogate_candidates):
                    p, u = idw_predict(surrogate_cands[k], X_arr, F_arr)
                    preds[k] = p
                    uncs[k] = u

                # Acquisitive function: minimize predicted value + uncertainty
                # Use expected improvement-like criterion
                f_min = np.min(F_arr)
                # EI-like: -pred + beta * unc
                beta = 0.5
                acq = -preds + beta * uncs
                best_idx = np.argmin(acq)
                x_surrogate = surrogate_cands[best_idx]
                candidates.append(x_surrogate)

                # Also try minimizing the surrogate directly
                best_pred_idx = np.argmin(preds)
                x_min_pred = surrogate_cands[best_pred_idx]
                candidates.append(x_min_pred)

                # Select the candidate with the best acquisition value
                # Evaluate all candidates via surrogate and pick the best
                best_acq_val = np.inf
                x_new = candidates[0]
                for cand in candidates:
                    if len(X_arr) >= 3:
                        p, u = idw_predict(cand, X_arr, F_arr)
                        acq_val = -p + beta * u
                    else:
                        acq_val = 0
                    if acq_val < best_acq_val:
                        best_acq_val = acq_val
                        x_new = cand

            x_new = np.clip(x_new, -5.0, 5.0)

            try:
                f_new = fwrap(x_new)
                record(x_new, f_new)
            except (TimeoutError, ValueError):
                break

            n_surrogate_steps += 1

        # ---------------- Phase 3: L-BFGS-B local refinement ----------------
        bounds = [(-5.0, 5.0) for _ in range(n)]

        # Try L-BFGS-B from the best point
        if best["x"] is not None:
            try:
                res = minimize(fwrap, best["x"].copy(), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 100, 'maxfun': 80})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
                    # Record the final point
                    X_eval.append(best["x"].copy())
                    F_eval.append(best["f"])
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Try a few more diverse starting points if budget remains
        remaining = budget - calls["count"]
        n_extra_starts = min(5, max(0, remaining // 15))

        for _ in range(n_extra_starts):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break

            # Pick starting point: near best or random
            r = rng.random()
            if r < 0.4:
                start_x = best["x"] + rng.standard_normal(n) * 0.3
            elif r < 0.7:
                start_x = best["x"] + rng.standard_normal(n) * 1.0
            else:
                start_x = rng.uniform(-5.0, 5.0, size=n)
            start_x = np.clip(start_x, -5.0, 5.0)

            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 50, 'maxfun': 30})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
                    X_eval.append(best["x"].copy())
                    F_eval.append(best["f"])
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0
            except:
                pass

        return best["x"], best["f"]
