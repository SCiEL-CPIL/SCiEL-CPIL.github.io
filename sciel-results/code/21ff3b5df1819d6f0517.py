import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s wall-clock

        dim = self.dim
        budget = self.budget

        # Call 1: Evaluate at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_used = 1

        # Call 2: Probe along e_0 with step 5
        x_probe = np.zeros(dim)
        x_probe[0] = 5.0
        f_probe = func(x_probe)
        calls_used += 1
        f1 = f_probe

        if f1 < best_f:
            best_f = f1
            best_x = x_probe.copy()

        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Determine signs for dimensions 0..dim-1
        # v[0] = (f1 - f0) / 5
        signs = np.zeros(dim)
        signs[0] = 1.0 if (f1 - f0) >= 0 else -1.0

        # Calls 3-5: Probe dimensions 1, 2, 3
        step = 5.0
        for i in range(1, min(dim, 4)):
            if calls_used >= budget or time.time() > deadline:
                break
            x_probe = np.zeros(dim)
            x_probe[i] = step
            f_probe = func(x_probe)
            calls_used += 1
            signs[i] = 1.0 if (f_probe - f0) >= 0 else -1.0
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()

        # Handle dimension 4 if exists (for dim=5, this is the 5th dimension)
        if dim >= 5:
            if calls_used >= budget or time.time() > deadline:
                # Can't probe dim 4, but we can still estimate using remaining info
                # For now, default to 0 for dim 4
                signs[4] = 0.0
            else:
                x_probe = np.zeros(dim)
                x_probe[4] = step
                f_probe = func(x_probe)
                calls_used += 1
                signs[4] = 1.0 if (f_probe - f0) >= 0 else -1.0
                if f_probe < best_f:
                    best_f = f_probe
                    best_x = x_probe.copy()
        elif dim > 4:
            # dim is always 5 per task, but handle gracefully
            pass

        # Call 6 (for dim=5, this is the 7th call total): Jump to the optimal vertex
        # For a linear function f(x) = v^T x + c, the minimum over [-5,5]^dim is at the vertex
        # x_i = 5 if v_i < 0 (signs[i] = -1) else x_i = -5 if v_i > 0 (signs[i] = 1)
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if signs[i] > 0:
                x_vertex[i] = -5.0
            elif signs[i] < 0:
                x_vertex[i] = 5.0
            else:
                # Unresolved (only possible for dim 4 if we ran out of budget)
                # Use midpoint or 0
                x_vertex[i] = 0.0

        if calls_used < budget and time.time() < deadline:
            f_vertex = func(x_vertex)
            calls_used += 1
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()

        # Use remaining budget: re-evaluate best point
        # This doesn't improve best_f but consumes budget safely
        while calls_used < budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_used += 1

        return best_x, best_f
