"""Optimizer for BBOB-style Sphere f1 with shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # Phase 1: Exploit Sphere structure to find optimum exactly
        # For Sphere: f(x) = ||R(x - x*)||^2 + offset = (x-x*)^T R^T R (x-x*) + offset
        # If R is orthogonal (rotation), then R^T R = I, so f(x) = ||x - x*||^2 + offset
        # We can recover x* using 2d+1 evaluations: f(0), f(e_i * s), f(-e_i * s) for each i
        
        # Actually, for a general quadratic f(x) = (x-c)^T A (x-c) + f0 with A positive definite,
        # we need more evaluations. But for Sphere with rotation, A = I (if rotation is orthogonal).
        # Let's assume A = I first and verify.
        
        # Strategy: Use 2d+1 evaluations to estimate the center of the sphere
        # f(x) = ||x - c||^2 + f0 = x^T x - 2 c^T x + c^T c + f0
        # From f(0) = c^T c + f0
        # f(s * e_i) = s^2 - 2 s c_i + c^T c + f0
        # f(-s * e_i) = s^2 + 2 s c_i + c^T c + f0
        # Adding: f(s*e_i) + f(-s*e_i) = 2 s^2 + 2(c^T c + f0)
        # Subtracting: f(-s*e_i) - f(s*e_i) = 4 s c_i => c_i = (f(-s*e_i) - f(s*e_i)) / (4s)
        
        # Choose s to be within bounds but large enough for accuracy
        # x* is in [-4, 4]^5, so use s = 4.0 to span most of the range
        
        s = 4.0
        f0 = func(np.zeros(dim))
        
        c = np.zeros(dim)
        for i in range(dim):
            x_pos = np.zeros(dim)
            x_neg = np.zeros(dim)
            x_pos[i] = s
            x_neg[i] = -s
            f_pos = func(x_pos)
            f_neg = func(x_neg)
            c[i] = (f_neg - f_pos) / (4.0 * s)
        
        # Now we have an estimate of the center c
        # Verify by checking if f(c) is close to f0 - ||c||^2 (for pure sphere)
        f_c = func(c)
        expected_f_c = f0 - np.sum(c**2)
        
        # The actual function might have a different scaling or offset
        # Let's also estimate the quadratic coefficient
        # f(x) = a * ||x - c||^2 + f_opt where a > 0
        # From f(0) = a * ||c||^2 + f_opt
        # f(c) = f_opt
        # So a = (f(0) - f(c)) / ||c||^2 if c != 0
        
        # But we don't need 'a' explicitly for finding the minimum, just c
        # However, if the rotation is not identity, the Hessian is not I
        # Let's do a quick check: evaluate along a few directions to see if it's isotropic
        
        # For now, assume it's a standard sphere and use c as the starting point
        # Then polish with (1+1)-ES
        
        x = c.copy()
        fx = f_c
        
        # Check if we're within bounds
        x = np.clip(x, -5, 5)
        if not np.allclose(x, c):
            fx = func(x)
        
        calls_made = 1 + 2 * dim + 1  # f0 + 2*dim + f_c = 1 + 10 + 1 = 12
        
        # Phase 2: Refine center using additional finite differences
        # Use a smaller step to refine each coordinate
        for _ in range(2):  # 2 refinement rounds
            if time.time() > deadline:
                break
            s2 = 0.5
            for i in range(dim):
                x_pos = x.copy()
                x_neg = x.copy()
                x_pos[i] += s2
                x_neg[i] -= s2
                x_pos = np.clip(x_pos, -5, 5)
                x_neg = np.clip(x_neg, -5, 5)
                f_pos = func(x_pos)
                f_neg = func(x_neg)
                calls_made += 2
                # For sphere: gradient_i = 2(x_i - c_i), so from finite diff:
                # (f_pos - f_neg) / (2*s2) ≈ 2(x_i - c_i)
                # c_i ≈ x_i - (f_pos - f_neg) / (4*s2)
                grad_est = (f_pos - f_neg) / (2.0 * s2)
                c[i] = x[i] - grad_est / 2.0
            # Update x to refined center
            x_new = np.clip(c, -5, 5)
            if not np.allclose(x_new, x):
                fx = func(x_new)
                calls_made += 1
                x = x_new
            else:
                # Still need to count the center eval if we moved
                pass
        
        # Phase 3: Polish with (1+1)-ES starting from refined center
        sigma = 0.1
        rng = np.random.default_rng(42)
        
        for i in range(budget - calls_made):
            if time.time() > deadline:
                break
            
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            # Adaptive sigma: if sigma gets too small or too large, adjust
            if sigma < 1e-8:
                sigma = 1e-8
            if sigma > 2.0:
                sigma = 2.0
        
        return x, fx
