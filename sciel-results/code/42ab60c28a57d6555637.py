"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with structurally-informed starting points for Katsuura."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        dim = self.dim
        budget = self.budget

        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_f, best_x
            n = dim
            # Build initial simplex with adaptive steps
            simplex = [np.array(x0, dtype=float, copy=True)]
            for i in range(n):
                xi = np.array(x0, dtype=float, copy=True)
                step = 0.3 if abs(xi[i]) < 0.5 else 0.15 * max(abs(xi[i]), 0.5)
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    return

            alpha, gamma, rho = 1.0, 2.0, -0.5

            while calls_made < budget and time.time() < deadline - 5:
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if max(fvals) - min(fvals) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                try:
                    fr = counted_func(xr)
                except RuntimeError:
                    return

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    try:
                        fe = counted_func(xe)
                    except RuntimeError:
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    try:
                        fc = counted_func(xc)
                    except RuntimeError:
                        return
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink: move all points toward best
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            try:
                                fvals[i] = counted_func(simplex[i])
                            except RuntimeError:
                                return

        # Generate structurally-informed starting points
        # Katsuura function has periodic structure with minima near specific patterns
        # The polynomial part suggests x=1, cosine parts have period related to pi
        start_points = []
        
        # Point near the known structure of Katsuura (minima tend to be near specific values)
        # Katsuura: sum over i of (1 - cos(24*pi*x_i - 12*pi))^2 + (x_i - 1)^2/100
        # Plus a product term: 10 * sum((x_i - 1)^2) for i < n-1
        # Minima are near x_i = 1 + k/12 for various k, but shifted by hidden transform
        
        # Add points near 1 (the nominal minimum before transformation)
        start_points.append(np.ones(dim))
        
        # Add points near other likely minimum locations
        for k in [-1, 0, 1]:
            pt = np.ones(dim) * (1 + k / 12.0)
            start_points.append(np.clip(pt, -5, 5))
        
        # Add some random points for diversity
        for _ in range(10):
            start_points.append(rng.uniform(-4, 4, dim))
        
        # Add points near 0 (in case transformation shifts things)
        start_points.append(np.zeros(dim))
        
        # Shuffle and deduplicate
        rng.shuffle(start_points)
        seen = set()
        unique_starts = []
        for p in start_points:
            key = tuple(np.round(p, 3))
            if key not in seen:
                seen.add(key)
                unique_starts.append(p)
        
        # Phase 1: Quick evaluation of all starting points
        for x in unique_starts:
            if calls_made >= budget or time.time() > deadline - 10:
                break
            try:
                counted_func(x)
            except RuntimeError:
                break
        
        # Phase 2: Run Nelder-Mead from the best few starting points
        # Sort starting points by their function values
        scored_starts = []
        for x in unique_starts:
            if best_x is not None:
                # We already evaluated these, but let's use the best ones
                pass
        
        # Use the best point found so far as primary start
        if best_x is not None:
            nelder_mead(best_x, budget - calls_made)
        
        # Phase 3: If budget remains, try from other promising starts
        if calls_made < budget - 20 and time.time() < deadline - 20:
            # Find the second-best starting point
            if len(unique_starts) > 1:
                # Re-evaluate a few to find good secondary starts
                candidates = []
                for x in unique_starts[:5]:
                    if np.allclose(x, best_x, atol=0.1):
                        continue
                    try:
                        f = counted_func(x)
                        candidates.append((f, x))
                    except RuntimeError:
                        break
                
                if candidates:
                    candidates.sort(key=lambda t: t[0])
                    for f, x in candidates[:3]:
                        if calls_made >= budget - 10 or time.time() > deadline - 10:
                            break
                        nelder_mead(x, budget - calls_made)
        
        # Phase 4: Final high-precision polish from the best point
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline - 10:
            # Use a smaller initial simplex for the polish
            n = dim
            simplex = [np.array(best_x, dtype=float, copy=True)]
            for i in range(n):
                xi = np.array(best_x, dtype=float, copy=True)
                step = 0.02
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    break

            if len(fvals) == n + 1:
                alpha, gamma, rho = 1.0, 2.0, -0.5
                while calls_made < budget and time.time() < deadline - 2:
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    if max(fvals) - min(fvals) < 1e-14:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    try:
                        fr = counted_func(xr)
                    except RuntimeError:
                        break

                    if fr < fvals[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        try:
                            fe = counted_func(xe)
                        except RuntimeError:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        if fr < fvals[-1]:
                            xc = centroid + 0.5 * (xr - centroid)
                        else:
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        try:
                            fc = counted_func(xc)
                        except RuntimeError:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                try:
                                    fvals[i] = counted_func(simplex[i])
                                except RuntimeError:
                                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
