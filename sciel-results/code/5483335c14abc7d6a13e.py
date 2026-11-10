"""Bent-cigar (f12) black-box optimizer with adaptive principal-axis search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Adaptive principal-axis (1+1) search for anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _initial_points(self, rng, func):
        # A small, structured start: origin + box corners + a few randoms.
        pts = [np.zeros(self.dim)]
        corners = np.array(np.meshgrid([-5.0, 5.0] * self.dim), dtype=float).reshape(-1, self.dim)
        pts.extend(corners)
        pts.extend(rng.uniform(-5, 5, size=(8, self.dim)))
        vals = np.empty(len(pts))
        for i, p in enumerate(pts):
            vals[i] = func(np.clip(p, -5, 5))
        return np.array(pts), vals

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin under the 330 s wall-clock kill
        rng = np.random.default_rng(0)

        pts, vals = self._initial_points(rng, func)
        i = int(np.argmin(vals))
        x = pts[i].copy()
        fx = float(vals[i])
        n_calls = len(pts)

        # Adaptive covariance from accepted (improving) steps.
        cov = np.eye(dim)
        n_acc = 0
        ref_sigma = 1.0
        sigma = 1.0
        last_fit = 0

        for step in range(1, budget):
            if time.time() > deadline:
                break
            if n_calls >= budget:
                break

            # Re-fit the covariance periodically to re-align with anisotropy.
            if n_acc >= 2 and step - last_fit >= 8:
                last_fit = step
                try:
                    w = cov + 1e-8 * np.eye(dim)
                    evals, evecs = np.linalg.eigh(w)
                    evals = np.clip(evals, 1e-10, None)
                    cov = (evecs * np.sqrt(evals)) @ evecs.T
                    eig = np.linalg.eigvalsh(cov)
                    span = max(float(np.max(eig)) / max(float(np.min(eig)), 1e-10), 1.0)
                    ref_sigma = 2.0 / (1.0 + 0.05 * np.log10(span))
                except Exception:
                    cov = np.eye(dim)
                    ref_sigma = 1.0

            # Sample in the principal-axis basis.
            try:
                L = np.linalg.cholesky(cov + 1e-10 * np.eye(dim))
            except Exception:
                L = np.eye(dim)
            z = rng.standard_normal(dim)
            y = x + sigma * (L @ z)
            y = np.clip(y, -5, 5)
            fy = float(func(y))
            n_calls += 1

            if fy < fx:
                d = y - x
                # Rank-1 covariance update (improvement direction).
                cov = cov + 0.1 * (np.outer(d, d) / ref_sigma)
                n_acc += 1
                x, fx = y, fy
                # Step-size control: grow on success, shrink on failure.
                sigma = min(sigma * 1.25, 6.0)
            else:
                sigma = max(sigma * 0.8, 1e-3)

        return x, fx
# EVOLVE-BLOCK-END
