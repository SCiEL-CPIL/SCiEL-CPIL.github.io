"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def halton_sequence(n, dim):
    """Generate n points in [0,1)^dim using Halton sequences."""
    points = np.zeros((n, dim))
    for i in range(1, n + 1):
        for d in range(dim):
            # Find the (d+1)-th prime
            prime = _prime(d + 1)
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                f /= prime
                r += f * (i_val % prime)
                i_val //= prime
            points[i - 1, d] = r
    return points


def _prime(n):
    """Return the n-th prime number (1-indexed)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    if n <= len(primes):
        return primes[n - 1]
    # Fallback: simple prime search
    candidate = primes[-1] + 2
    while True:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            return candidate
        candidate += 2


class Optimizer:
    """Multi-resolution Halton grid scan with coordinate descent refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        def timed_call(x):
            if time.time() > deadline:
                return None
            return func(x)

        call_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Multi-resolution Halton scan
        # Use a fraction of budget for global exploration
        scan_budget = int(budget * 0.4)  # 40% for scanning
        remaining = budget - scan_budget

        # Generate Halton points mapped to [-5, 5]^dim
        halton_pts = halton_sequence(scan_budget, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_pts = halton_pts * 10.0 - 5.0

        # Add center point
        center = np.zeros(dim)
        f_center = func(center)
        call_count += 1
        update_best(center, f_center)

        # Evaluate Halton points
        for pt in halton_pts:
            if call_count >= budget or time.time() > deadline:
                break
            f_val = func(pt)
            call_count += 1
            update_best(pt, f_val)

        # Phase 2: Coordinate descent refinement from best point
        # Use remaining budget for local search
        x = best_x.copy()
        f = best_f

        # Adaptive step sizes for coordinate descent
        step_sizes = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in step_sizes:
            if call_count >= budget or time.time() > deadline:
                break
            
            improved = True
            while improved and call_count < budget and time.time() <= deadline:
                improved = False
                for d in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    
                    # Try both directions
                    for direction in [1, -1]:
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        
                        if np.allclose(x_new, x):
                            continue
                        
                        f_new = func(x_new)
                        call_count += 1
                        
                        if f_new < f:
                            x = x_new
                            f = f_new
                            update_best(x, f)
                            improved = True
                            break  # Restart coordinate loop after improvement

        # Phase 3: If we still have budget, do a finer local search
        if call_count < budget and time.time() < deadline:
            # Local random perturbation around best
            rng = np.random.default_rng(123)
            local_radius = 0.1
            while call_count < budget and time.time() < deadline:
                x_perturb = best_x + rng.normal(0, local_radius, dim)
                x_perturb = np.clip(x_perturb, -5, 5)
                f_perturb = func(x_perturb)
                call_count += 1
                if f_perturb < best_f:
                    best_f = f_perturb
                    best_x = x_perturb.copy()
                    local_radius *= 0.7  # Shrink radius on improvement

        return best_x, best_f
# EVOLVE-BLOCK-END
