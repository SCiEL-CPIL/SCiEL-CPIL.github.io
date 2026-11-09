"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """
    Optimizer for BBOB f18 (Schaffers F7 ill-conditioned).

    Strategy:
      Phase 1: Random sampling to locate the basin.
      Phase 2: (1+1)-ES with a 1/5 success-rule step-size adaptation and
               a covariance matrix estimated from recent successful steps
               to adapt to the ill-conditioning.
      Phase 3: Final local refinement.

    Entry point and return interface are kept exactly as required.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # ---------------- Phase 1: Random sampling ----------------
        n_samples = min(50, max(10, total_budget // 10))
        for _ in range(n_samples):
            if time.time() > deadline - 2.0 or calls >= total_budget:
                break
            x = rng.uniform(-5, 5, dim)
            try:
                f = func(x)
            except Exception:
                continue
            calls += 1
            record(x, f)

        # ---------------- Phase 2: (1+1)-ES with 1/5 rule and covariance adaptation ----------------
        # Parameters for 1/5 rule
        success_threshold = 0.2  # 1/5
        sigma = 1.0
        C = np.eye(dim)  # Covariance matrix
        B, D = np.eye(dim), np.ones(dim)
        
        # Window for covariance estimation
        window_size = 20
        successful_steps = []
        
        # Main loop
        while calls < total_budget:
            if time.time() > deadline - 5.0:
                break

            # Generate offspring
            z = rng.standard_normal(dim)
            y = best_x + sigma * (B @ (D * z))
            y = np.clip(y, -5, 5)
            
            try:
                f_y = func(y)
            except Exception:
                f_y = float('inf')
            
            calls += 1
            
            # Accept or reject
            if f_y < best_f:
                # Success
                step = (y - best_x) / sigma
                successful_steps.append(step)
                if len(successful_steps) > window_size:
                    successful_steps.pop(0)
                
                # Update best
                best_x = y.copy()
                best_f = f_y
                
                # Adapt sigma (increase)
                sigma *= 1.05  # 1/5 rule: if success rate > 0.2, increase sigma
                
                # Update covariance matrix
                if len(successful_steps) >= 5:
                    # Compute covariance of successful steps
                    steps_arr = np.array(successful_steps)
                    # Center the steps
                    mean_step = np.mean(steps_arr, axis=0)
                    centered_steps = steps_arr - mean_step
                    C_new = np.cov(centered_steps, rowvar=False)
                    if C_new.size == 0 or not np.all(np.isfinite(C_new)):
                        C_new = np.eye(dim)
                    else:
                        # Ensure PSD and positive trace
                        C_new = (C_new + C_new.T) / 2
                        ev, evv = np.linalg.eigh(C_new)
                        ev = np.maximum(ev, 1e-8)
                        C_new = (evv * ev) @ evv.T
                        C_new = (C_new + C_new.T) / 2
                        
                        # Blend with old covariance to avoid instability
                        C = 0.8 * C + 0.2 * C_new
                        B, D = _eigh_sqrt(C)
                        
            else:
                # Failure
                # Adapt sigma (decrease)
                sigma *= 0.95  # 1/5 rule: if success rate < 0.2, decrease sigma

            # Safety checks
            sigma = float(np.clip(sigma, 1e-8, 10.0))
            if not np.all(np.isfinite(best_x)):
                best_x = np.zeros(dim)
                best_f = float('inf')
                sigma = 1.0
                C = np.eye(dim)
                B, D = np.eye(dim), np.ones(dim)
                successful_steps = []

        # ---------------- Phase 3: Local refinement ----------------
        if calls < total_budget - 5 and time.time() < deadline - 5:
            remaining = total_budget - calls
            base_scales = [0.1, 0.05, 0.02, 0.01, 0.005]
            for _ in range(min(remaining, 20)):
                if time.time() > deadline - 2:
                    break
                for scale in base_scales:
                    if calls >= total_budget:
                        break
                    x_try = best_x + scale * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    try:
                        f_try = func(x_try)
                    except Exception:
                        calls += 1
                        continue
                    calls += 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()

        return best_x, best_f


def _eigh_sqrt(C):
    ev, evv = np.linalg.eigh(C)
    ev = np.maximum(ev, 1e-12)
    return evv, np.sqrt(ev)
# EVOLVE-BLOCK-END
