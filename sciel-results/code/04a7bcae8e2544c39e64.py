"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead derivative-free local search on BBOB f12."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        # 1) Generate a diverse set of candidate starting points
        # We use 5 different starting points to increase the chance of finding the minimum.
        starts = []
        starts.append(rng.uniform(-5, 5, dim))
        starts.append(np.zeros(dim))
        starts.append(rng.uniform(-2, 2, dim))
        starts.append(rng.uniform(2, 4, dim))
        starts.append(rng.uniform(-4, -2, dim))

        # 2) Evaluate starts
        best_x = None
        best_f = float('inf')
        valid_starts = []
        calls_made = 0

        for s in starts:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            fs = func(s)
            calls_made += 1
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
            valid_starts.append((s, fs))

        if best_x is None:
            return None, None

        # 3) Nelder-Mead
        # We will spend the remaining budget on Nelder-Mead.
        # Standard NM for dim=5 requires dim+1 = 6 vertices.
        # Initial simplex construction.
        x0 = best_x
        f0 = best_f
        alpha = 1.0
        beta = 0.5
        gamma = 2.0
        sigma = 0.5
        delta = 1.0 # Initial step size

        # Construct simplex
        simplex = [x0]
        simplexf = [f0]
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            simplex.append(v)
            fv = func(v)
            calls_made += 1
            simplex.append(v) # Wait, I'm appending to list, need to manage structure carefully
            # Let's re-structure simplex as list of (x, f)
            # Actually, let's just keep two lists: xs, fs
            # Or a list of tuples.
            
        # Let's restart the simplex construction cleanly
        xs = [x0]
        fs = [f0]
        for i in range(dim):
            v = x0.copy()
            v[i] += delta
            fv = func(v)
            calls_made += 1
            xs.append(v)
            fs.append(fv)
            
        # Sort by fitness
        # We need to maintain xs and fs sorted.
        # For dim=5, sorting 6 items is cheap.
        
        while calls_made < budget and time.time() < deadline:
            # Sort
            order = np.argsort(fs)
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Centroid of all but the worst
            centroid = np.mean(xs[:-1], axis=0)
            worst_x = xs[-1]
            worst_f = fs[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_x)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_made += 1
            
            if fr < fs[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst_x)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_made += 1
                if fe < fr:
                    xs[-1] = xe
                    fs[-1] = fe
                else:
                    xs[-1] = xr
                    fs[-1] = fr
            elif fr < fs[-2]:
                # Accept reflection
                xs[-1] = xr
                fs[-1] = fr
            else:
                # Contraction
                # Inside contraction
                xc = centroid + beta * (worst_x - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_made += 1
                if fc < worst_f:
                    xs[-1] = xc
                    fs[-1] = fc
                else:
                    # Shrink
                    # Move all points towards best
                    for i in range(1, len(xs)):
                        xs[i] = xs[0] + sigma * (xs[i] - xs[0])
                        fs[i] = func(xs[i])
                        calls_made += 1
                        if calls_made >= budget or time.time() >= deadline:
                            break
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    # Re-sort after shrink
                    order = np.argsort(fs)
                    xs = [xs[i] for i in order]
                    fs = [fs[i] for i in order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
