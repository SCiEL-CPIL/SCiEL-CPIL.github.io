"""Portfolio optimizer for BBOB-style f7 (Step ellipsoidal), dim=5, budget=500."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: broad samples + axis scans + small CMA-ES + coordinate polish,
    sharing a call budget and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        B = self.budget
        d = self.dim
        LO, HI = -5.0, 5.0

        used = 0

        def f(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), LO, HI)
            used += 1
            return float(func(x))

        # ---- Phase 0: Initial Sampling -----------------------------------
        # Center, axis extremes, and a few random points
        candidates = [np.zeros(d)]
        for i in range(d):
            p1 = np.zeros(d); p1[i] = -4.0; candidates.append(p1)
            p2 = np.zeros(d); p2[i] = 4.0; candidates.append(p2)
        
        rng_init = np.random.default_rng(42)
        for _ in range(5):
            candidates.append(rng_init.uniform(-4, 4, d))
            
        best_x = None
        best_f = np.inf
        for x in candidates:
            if used >= B or time.time() > deadline:
                break
            v = f(x)
            if v < best_f:
                best_f, best_x = v, x.copy()
        
        # ---- Phase 1: Nelder-Mead Multi-Start ----------------------------
        def nelder_mead(start_x, max_iter=150, tol=1e-10):
            nonlocal used, best_f, best_x
            if used >= B or time.time() > deadline:
                return
            
            # Initialize Simplex
            delta = 0.5
            simplex = np.empty((d + 1, d))
            simplex[0] = start_x
            for i in range(d):
                p = start_x.copy()
                p[i] += delta
                simplex[i + 1] = np.clip(p, LO, HI)
            
            fvals = np.array([f(simplex[i]) for i in range(d + 1)])
            
            # Update global best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()
            
            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if used >= B or time.time() > deadline:
                    break
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < tol:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, LO, HI)
                fr = f(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, LO, HI)
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    
                    xc = np.clip(xc, LO, HI)
                    fc = f(xc)
                    
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], LO, HI)
                            fvals[i] = f(simplex[i])
                
                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()
            
            return best_x, best_f

        # Run NM 1
        nelder_mead(best_x, max_iter=150)
        
        # Run NM 2 (Restart with perturbation)
        if used < B and time.time() < deadline:
            rng_restart = np.random.default_rng(123)
            start2 = np.clip(best_x + 0.5 * rng_restart.standard_normal(d), LO, HI)
            nelder_mead(start2, max_iter=100)

        # ---- Phase 2: Coordinate Polish ----------------------------------
        if used < B and time.time() < deadline:
            for i in range(d):
                if used >= B or time.time() > deadline:
                    break
                lo = max(LO, best_x[i] - 1.0)
                hi = min(HI, best_x[i] + 1.0)
                for t in np.linspace(lo, hi, 7):
                    if used >= B or time.time() > deadline:
                        break
                    cand = best_x.copy(); cand[i] = t
                    v = f(cand)
                    if v < best_f:
                        best_f, best_x = v, cand.copy()

        # ---- Phase 3: Final Tiny Perturbation Polish ---------------------
        rng2 = np.random.default_rng(999)
        fails = 0
        while used < B and time.time() < deadline:
            y = np.clip(best_x + 0.1 * rng2.standard_normal(d), LO, HI)
            fy = f(y)
            if fy < best_f:
                best_f, best_x = fy, y.copy()
                fails = 0
            else:
                fails += 1
                if fails >= 8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
