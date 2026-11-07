"""Derivative-free trust-region optimizer with local quadratic surrogates for BBOB f18.

Strategy:
1. Start with a coarse global search using a trust-region method that builds
   local quadratic models from a small set of evaluations.
2. Adapt the trust-region radius based on model prediction accuracy.
3. Use a simplex-like pattern for sampling points around the current center.
4. Include a final fine-tuning phase with very small steps.
"""
import time
import numpy as np


class Optimizer:
    """Trust-region optimizer with local quadratic surrogates."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 310.0  # Safety margin before 330s kill
        calls = [0]
        n = self.dim

        def time_ok():
            return (time.time() - t0) < WALL

        def safe_eval(x):
            """One recorded, exception-safe, budget/time-guarded evaluation."""
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        # Global best tracking
        best_x = np.zeros(n)
        best_f = np.inf

        # Trust-region parameters
        delta = 2.0  # Initial trust-region radius
        delta_min = 1e-6
        delta_max = 4.5  # Keep within box bounds

        # Center of trust region
        center = np.zeros(n)

        # Number of points to sample in the trust region
        # We'll use a simplex-like pattern: n+1 points plus the center
        # For n=5, that's 6 points per iteration

        # Track model quality
        model_residual = 0.0

        # Main loop
        while calls[0] < self.budget and time_ok():
            # Generate candidate points around center
            # Use a scaled simplex pattern
            points = [center.copy()]
            
            # Generate n directions (simplex vertices)
            for i in range(n):
                direction = np.zeros(n)
                direction[i] = 1.0
                # Alternate positive and negative for better coverage
                sign = 1.0 if (i % 2 == 0) else -1.0
                points.append(center + sign * delta * direction)
            
            # Also add some diagonal directions for better exploration
            for i in range(min(n, 3)):
                direction = np.zeros(n)
                direction[i] = 1.0
                direction[(i + 1) % n] = 1.0
                points.append(center + delta * 0.7 * direction)
                points.append(center - delta * 0.7 * direction)

            # Evaluate all points
            values = []
            valid_points = []
            for p in points:
                # Clip to box
                p_clipped = np.clip(p, -5.0, 5.0)
                v = safe_eval(p_clipped)
                if v is None:
                    break
                values.append(v)
                valid_points.append(p_clipped)
            
            if not values:
                break

            values = np.array(values)
            valid_points = np.array(valid_points)

            # Find best in this batch
            idx_best = int(np.argmin(values))
            if values[idx_best] < best_f:
                best_f = float(values[idx_best])
                best_x = valid_points[idx_best].copy()

            # Build a local quadratic model
            # Fit: f(x) ≈ f0 + g^T (x - x0) + 0.5 (x - x0)^T H (x - x0)
            # We have m points, need to fit parameters
            # Use least squares if we have enough points
            
            m = len(values)
            if m >= 3:  # Need at least 3 points for a meaningful fit
                # Center the data
                x0 = center
                X = valid_points - x0  # m x n
                
                # Features: [1, x_1, ..., x_n, x_1^2, x_1*x_2, ..., x_n^2]
                # But for simplicity, use a linear model first and see if it helps
                # Actually, let's use a simple linear approximation:
                # f(x) ≈ f0 + g^T (x - x0)
                
                # Fit linear model: values ≈ f0 + X @ g
                # Use the center value as f0
                f0 = values[np.argmin(np.sum(X**2, axis=1))]  # Value at center
                X_centered = X - X[np.argmin(np.sum(X**2, axis=1))]
                y_centered = values - f0
                
                # Solve least squares: X_centered @ g ≈ y_centered
                if len(X_centered) >= n:
                    try:
                        g, _, _, _ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
                        # Predicted step: -g / ||g|| * delta
                        g_norm = np.linalg.norm(g)
                        if g_norm > 1e-10:
                            step = -g / g_norm * delta
                            new_center = center + step
                            new_center = np.clip(new_center, -5.0, 5.0)
                            
                            # Check if the model predicts improvement
                            predicted_value = f0 + g @ step
                            actual_best = np.min(values)
                            
                            # If model predicts improvement, accept the step
                            if predicted_value < f0:
                                center = new_center
                                # Keep delta the same or increase slightly
                                delta = min(delta * 1.1, delta_max)
                            else:
                                # Model doesn't predict improvement, shrink delta
                                delta = max(delta * 0.7, delta_min)
                        else:
                            # Gradient is zero, we might be at a minimum
                            delta = max(delta * 0.5, delta_min)
                    except np.linalg.LinAlgError:
                        delta = max(delta * 0.7, delta_min)
                else:
                    delta = max(delta * 0.7, delta_min)
            else:
                delta = max(delta * 0.7, delta_min)

            # If delta is very small, do a fine-tuning phase
            if delta < 1e-4:
                # Fine-tuning: small random steps
                while calls[0] < self.budget and time_ok():
                    step = 0.01 * np.random.randn(n)
                    new_x = np.clip(center + step, -5.0, 5.0)
                    v = safe_eval(new_x)
                    if v is None:
                        break
                    if v < best_f:
                        best_f = float(v)
                        best_x = new_x.copy()
                        center = new_x.copy()
                    else:
                        # No improvement, try a smaller step
                        pass
                break

        return best_x, float(best_f)
