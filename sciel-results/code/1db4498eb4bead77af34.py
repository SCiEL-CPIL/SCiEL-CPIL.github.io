"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Strategy: Structure-exploiting iterative coordinate-wise grid search + Nelder-Mead polish.
    
    Rastrigin f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global min at x=0 (value 0), local minima at integers (value 10*n).
    Period 2 in each coordinate.
    
    Approach:
    1. Minimal initial sampling to find approximate basin.
    2. Iterative coarse-to-fine coordinate-wise grid search (multiple passes).
    3. Extended Nelder-Mead polish with high-precision final pass.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, calls_left, deadline, max_iter=300):
        """Nelder-Mead polish with adaptive parameters."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        if calls >= calls_left or time.time() > deadline:
            return best_x, best_f, calls

        # Build initial simplex with smaller step for better convergence
        simplex = [x.copy()]
        fvals = [fx]
        step = 0.05
        for i in range(d):
            if calls >= calls_left or time.time() > deadline:
                break
            xi = x.copy()
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        if len(simplex) < 2:
            return best_x, best_f, calls

        alpha, gamma, rho, shrink = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_iter):
            if calls >= calls_left or time.time() > deadline:
                break
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()

            if calls >= calls_left or time.time() > deadline:
                break

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                if calls >= calls_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                if calls >= calls_left or time.time() > deadline:
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    break
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + shrink * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls < calls_left and time.time() <= deadline:
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        else:
                            break
                    if calls >= calls_left or time.time() > deadline:
                        break

        return best_x, best_f, calls

    def _iterative_coord_search(self, func, x0, calls_left, deadline, num_passes=3):
        """
        Iterative coordinate-wise grid search with decreasing step sizes.
        Multiple passes allow refinement as other coordinates improve.
        """
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Define step sizes for each pass (coarse to fine)
        pass_steps = [
            [1.0, 0.5, 0.25],    # Pass 1: coarse
            [0.125, 0.0625],     # Pass 2: medium
            [0.03125, 0.015625]  # Pass 3: fine
        ]

        for pass_idx in range(min(num_passes, len(pass_steps))):
            if calls >= calls_left or time.time() > deadline:
                break
            steps = pass_steps[pass_idx]
            for coord in range(d):
                if calls >= calls_left or time.time() > deadline:
                    break
                base = x[coord]
                for step in steps:
                    for direction in [-1, 1]:
                        if calls >= calls_left or time.time() > deadline:
                            break
                        offset = direction * step
                        x_trial = x.copy()
                        x_trial[coord] = base + offset
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls += 1
                        if f_trial < best_f:
                            best_f = f_trial
                            best_x = x_trial.copy()
                            x = x_trial.copy()
                            base = x[coord]

        return best_x, best_f, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill
        calls = 0

        # Phase 1: Minimal initial sampling - just origin and a few strategic points
        initial_pts = [
            np.zeros(dim),
            np.ones(dim) * 0.5,
            np.ones(dim) * (-0.5),
            np.array([1.0, -1.0, 0.0, 1.0, -1.0][:dim]),
            np.array([-1.0, 1.0, 0.0, -1.0, 1.0][:dim]),
        ]
        
        best_f = np.inf
        best_x = np.zeros(dim)
        
        for pt in initial_pts:
            if calls >= budget or time.time() > deadline:
                break
            x = np.clip(pt, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Iterative coordinate-wise grid search (main workhorse)
        # Allocate ~60% of budget to this phase
        refine_budget = min(int(budget * 0.6), budget - calls)
        if refine_budget >= 30 and time.time() < deadline:
            x_ref, f_ref, c_ref = self._iterative_coord_search(
                func, best_x, refine_budget, deadline, num_passes=3
            )
            calls += c_ref
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()

        # Phase 3: Nelder-Mead polish - allocate remaining budget
        polish_budget = budget - calls
        if polish_budget >= 20 and time.time() < deadline:
            # Use most of the remaining budget for NM
            nm_budget = min(polish_budget, budget - calls)
            x_pol, f_pol, c_pol = self._nm_polish(
                func, best_x, nm_budget, deadline, max_iter=500
            )
            calls += c_pol
            if f_pol < best_f:
                best_f = f_pol
                best_x = x_pol.copy()

        # Phase 4: Final high-precision polish if budget remains
        if calls < budget and time.time() < deadline and (budget - calls) >= 10:
            final_budget = budget - calls
            x_final, f_final, c_final = self._nm_polish(
                func, best_x, final_budget, deadline, max_iter=200
            )
            calls += c_final
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
