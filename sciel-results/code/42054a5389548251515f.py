"""Optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Ridge-aware stochastic optimizer.

    Strategy:
      1. Probe to estimate the flat (ridge) direction of the sharp-ridge function.
      2. Lock the ridge direction; step along it with an adaptive 1-D walk.
      3. Damped random walk in the orthogonal subspace to reduce cross-ridge error.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0          # safety margin before the 300 s kill
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        def now():
            return time.time()

        # ---- Phase 0: initial point ----
        x = rng.uniform(-4.0, 4.0, d)
        fx = func(x)
        calls = 1
        best_x, best_f = x.copy(), fx

        # ---- Phase 1: estimate the ridge (flat) direction ----
        # A sharp ridge is flat along one direction and steep elsewhere.
        # We probe random directions and pick the one with the smallest
        # second-order change (most flat) to define the ridge axis.
        # Use a small number of probes; cost is bounded.
        probe_calls = 0
        max_probe = min(40, budget // 6)
        best_flat_dir = None
        best_flat_score = np.inf
        r = 0.3
        for _ in range(max_probe):
            v = rng.standard_normal(d)
            nrm = np.linalg.norm(v)
            if nrm < 1e-9:
                continue
            v = v / nrm
            xp = np.clip(x + r * v, -5.0, 5.0)
            xm = np.clip(x - r * v, -5.0, 5.0)
            fpp = func(xp)
            fmm = func(xm)
            probe_calls += 2
            # second-order (flatness) measure: |f(x+v)+f(x-v)-2f(x)|
            second = abs(fpp + fmm - 2.0 * fx)
            if second < best_flat_score:
                best_flat_score = second
                best_flat_dir = v
            if now() > deadline - 60.0:
                break
            if budget - calls - probe_calls < 50:
                break

        if best_flat_dir is None:
            # fallback: a random direction
            best_flat_dir = rng.standard_normal(d)
            best_flat_dir /= np.linalg.norm(best_flat_dir)

        # Build an orthonormal basis with the ridge direction as first axis.
        ridge = best_flat_dir.copy()
        # Complete to a basis.
        basis = [ridge]
        for _ in range(d - 1):
            v = rng.standard_normal(d)
            for b in basis:
                v = v - np.dot(v, b) * b
            nrm = np.linalg.norm(v)
            if nrm < 1e-9:
                continue
            v = v / nrm
            basis.append(v)
        while len(basis) < d:
            v = rng.standard_normal(d)
            for b in basis:
                v = v - np.dot(v, b) * b
            nrm = np.linalg.norm(v)
            if nrm < 1e-9:
                continue
            v = v / nrm
            basis.append(v)
        basis = np.array(basis[:d])

        # ---- Phase 2: ridge-aware search ----
        # Coordinates: x = center + s*ridge + sum_i a_i * basis[i]
        # We track s (ridge coordinate) and the orthogonal offset vector.
        s = 0.0
        orth = np.zeros(d - 1)
        sigma_ridge = 1.0
        sigma_orth = 0.5
        ridge_success = 0
        orth_success = 0
        R = 10  # window for 1/5 rule

        def rebuild(s, orth):
            return np.clip(x + s * ridge + basis[1:] @ orth, -5.0, 5.0)

        # Re-center so current x is the origin of the local frame.
        # s=0, orth=0 currently maps to x (since we defined frame at x).
        for it in range(budget - calls - probe_calls):
            if now() > deadline:
                break
            # --- Ridge step (1-D adaptive walk) ---
            if now() > deadline - 5.0:
                break
            ds = sigma_ridge * rng.standard_normal()
            new_s = s + ds
            cand = rebuild(new_s, orth)
            fcand = func(cand)
            calls += 1
            if fcand < fx:
                s = new_s
                fx = fcand
                if fx < best_f:
                    best_f, best_x = fx, cand.copy()
                ridge_success += 1
            # --- Orthogonal step (damped random walk) ---
            if now() > deadline - 5.0:
                break
            dorth = sigma_orth * rng.standard_normal(d - 1)
            new_orth = np.clip(orth + dorth, -5.0, 5.0)
            cand2 = rebuild(s, new_orth)
            fcand2 = func(cand2)
            calls += 1
            if fcand2 < fx:
                orth = new_orth
                fx = fcand2
                if fx < best_f:
                    best_f, best_x = fx, cand2.copy()
                orth_success += 1

            # --- Adaptive step sizes via 1/5 rule ---
            if (it + 1) % R == 0:
                if ridge_success >= R // 5 + 1:
                    sigma_ridge *= 1.3
                else:
                    sigma_ridge *= 0.7
                ridge_success = 0
                if orth_success >= R // 5 + 1:
                    sigma_orth *= 1.2
                else:
                    sigma_orth *= 0.6
                orth_success = 0

                # Shrink orthogonal step as we get closer (steep region)
                if fx < best_f + 1e-3:
                    sigma_orth *= 0.9

            # If very close to optimum, do a fine local refinement
            if best_f < fx + 1e-6 and now() > t0 + 200.0:
                # switch to pure fine search
                sigma_ridge = min(sigma_ridge, 0.05)
                sigma_orth = min(sigma_orth, 0.05)

        return best_x, best_f
# EVOLVE-BLOCK-END
