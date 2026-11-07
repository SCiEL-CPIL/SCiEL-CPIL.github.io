"""Linear function optimizer: exact gradient recovery via simplex probing."""
import numpy as np
import time


class Optimizer:
    """Linear function optimizer: exact gradient recovery via simplex probing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            calls_made += 1
            try:
                return func(x)
            except Exception:
                return None

        def time_ok():
            return time.time() < deadline

        # Step 1: Evaluate at origin (call 1) to get intercept b
        origin = np.zeros(dim)
        f_origin = safe_call(origin)
        if f_origin is None:
            # Fallback: random search
            rng = np.random.default_rng(42)
            for _ in range(budget - calls_made):
                if not time_ok():
                    break
                x = rng.uniform(-5, 5, dim)
                fx = safe_call(x)
                if fx is not None and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            return best_x, best_f

        best_f = f_origin
        best_x = origin.copy()

        # Step 2: Exact gradient recovery via axis probes (calls 2 through dim+1)
        # For linear f(x) = g^T x + c, probing x = 5*e_i gives g_i * 5 + c
        # So g_i = (f(5*e_i) - f(0)) / 5
        gradient = np.zeros(dim)
        probe_success = True
        for i in range(dim):
            if not time_ok() or calls_made >= budget:
                probe_success = False
                break
            probe = np.zeros(dim)
            probe[i] = 5.0
            f_probe = safe_call(probe)
            if f_probe is None:
                probe_success = False
                break
            if f_probe < best_f:
                best_f = f_probe
                best_x = probe.copy()
            gradient[i] = (f_probe - f_origin) / 5.0

        # Step 3: If gradient was successfully recovered, compute optimal vertex
        if probe_success and calls_made < budget and time_ok():
            # Optimal vertex: x_i = -5 if g_i > 0, +5 if g_i < 0, 0 if g_i == 0
            optimal_vertex = np.zeros(dim)
            for i in range(dim):
                if gradient[i] > 1e-10:
                    optimal_vertex[i] = -5.0
                elif gradient[i] < -1e-10:
                    optimal_vertex[i] = 5.0
                else:
                    optimal_vertex[i] = 0.0

            # Evaluate the optimal vertex
            f_opt = safe_call(optimal_vertex)
            if f_opt is not None:
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = optimal_vertex.copy()
            else:
                # If the call failed, use the vertex anyway as our best guess
                best_x = optimal_vertex.copy()

        # Step 4: If we have remaining budget, verify with a few nearby vertices
        # or do a quick local search to ensure we're at the true minimum
        if calls_made < budget and time_ok():
            # Check the opposite vertex too (in case of numerical issues)
            opposite = -best_x.copy()
            if not np.array_equal(opposite, best_x):
                f_opp = safe_call(opposite)
                if f_opp is not None and f_opp < best_f:
                    best_f = f_opp
                    best_x = opposite.copy()

        return best_x, best_f
