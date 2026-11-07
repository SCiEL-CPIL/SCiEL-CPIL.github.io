"""Grid-based coarse-to-fine search with local refinement for Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine grid scan with local Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.deadline = None

    def _eval(self, func, x):
        """Evaluate func(x), tracking call count and deadline."""
        if self.calls >= self.budget:
            return np.inf
        if self.deadline is not None and time.time() > self.deadline:
            return np.inf
        x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
        fx = float(func(x))
        self.calls += 1
        return fx

    def _grid_scan(self, func, lo, hi, n_per_dim, top_k=3):
        """
        Evaluate grid points in [lo, hi]^dim with n_per_dim points per dimension.
        Returns list of (fx, x) sorted by fx, up to top_k best.
        """
        if self.calls >= self.budget:
            return []
        
        # Number of grid points
        n_total = n_per_dim ** self.dim
        
        # If too many points, reduce
        while n_total > max(1, self.budget - self.calls - 50) and n_per_dim > 2:
            n_per_dim -= 1
            n_total = n_per_dim ** self.dim
        
        if n_per_dim < 2:
            n_per_dim = 2
            n_total = n_per_dim ** self.dim
        
        # Generate grid
        coords = np.linspace(lo, hi, n_per_dim)
        results = []
        
        # Use itertools.product for grid enumeration
        import itertools
        for idx in itertools.product(range(n_per_dim), repeat=self.dim):
            if self.calls >= self.budget:
                break
            if self.deadline is not None and time.time() > self.deadline:
                break
            x = np.array([coords[i] for i in idx], dtype=np.float64)
            fx = self._eval(func, x)
            if np.isfinite(fx):
                results.append((fx, x))
        
        results.sort(key=lambda t: t[0])
        return results[:top_k]

    def _nelder_mead(self, func, x0, max_iter=50):
        """
        Simple Nelder-Mead local optimizer.
        Returns best (fx, x) found.
        """
        dim = self.dim
        # Initial simplex
        simplex = [np.array(x0, dtype=np.float64)]
        for i in range(dim):
            xi = np.array(x0, dtype=np.float64).copy()
            step = 0.5 if xi[i] < 4.5 else -0.5
            xi[i] += step
            simplex.append(xi)
        
        # Evaluate initial simplex
        fx_list = [self._eval(func, x) for x in simplex]
        
        alpha, gamma, rho = 1.0, 2.0, 0.5
        best_fx = min(fx_list)
        best_x = simplex[np.argmin(fx_list)].copy()
        
        for _ in range(max_iter):
            if self.calls >= self.budget:
                break
            if self.deadline is not None and time.time() > self.deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_x = simplex[-1]
            worst_fx = fx_list[-1]
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_x)
            fxr = self._eval(func, xr)
            
            if fxr < fx_list[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst_x)
                fxe = self._eval(func, xe)
                if fxe < fxr:
                    simplex[-1] = xe
                    fx_list[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fxr
            elif fxr < fx_list[-2]:
                simplex[-1] = xr
                fx_list[-1] = fxr
            else:
                # Contraction
                xc = centroid + rho * (worst_x - centroid)
                fxc = self._eval(func, xc)
                if fxc < worst_fx:
                    simplex[-1] = xc
                    fx_list[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fx_list[i] = self._eval(func, simplex[i])
            
            # Update best
            for i in range(len(fx_list)):
                if fx_list[i] < best_fx:
                    best_fx = fx_list[i]
                    best_x = simplex[i].copy()
        
        return best_fx, best_x

    def __call__(self, func):
        self.calls = 0
        self.deadline = time.time() + 280  # Safety margin before 300s limit
        
        dim = self.dim
        lo, hi = -5.0, 5.0
        
        # Phase 1: Coarse grid scan
        # For dim=5, use 3^5 = 243 points initially
        n_coarse = 3
        coarse_results = self._grid_scan(func, lo, hi, n_coarse, top_k=5)
        
        if not coarse_results:
            # Fallback: random search
            rng = np.random.default_rng(42)
            x = rng.uniform(lo, hi, dim)
            fx = self._eval(func, x)
            return x, fx
        
        # Phase 2: Fine grid around best coarse points
        best_coarse_fx, best_coarse_x = coarse_results[0]
        
        # Determine region size based on grid spacing
        grid_spacing = (hi - lo) / (n_coarse - 1)
        
        # Fine grid: 4 points per dim in a region of size grid_spacing around best
        region_size = grid_spacing
        n_fine = 4
        fine_lo = np.maximum(best_coarse_x - region_size / 2, lo)
        fine_hi = np.minimum(best_coarse_x + region_size / 2, hi)
        
        fine_results = self._grid_scan(func, fine_lo, fine_hi, n_fine, top_k=3)
        
        if fine_results:
            best_fine_fx, best_fine_x = fine_results[0]
        else:
            best_fine_fx, best_fine_x = best_coarse_fx, best_coarse_x
        
        # Phase 3: Local Nelder-Mead refinement from best points
        candidates = []
        if fine_results:
            candidates.extend(fine_results[:2])
        candidates.extend(coarse_results[:2])
        
        best_fx = best_fine_fx
        best_x = best_fine_x
        
        for fx, x in candidates:
            if self.calls >= self.budget - 10:
                break
            if self.deadline is not None and time.time() > self.deadline - 10:
                break
            nm_fx, nm_x = self._nelder_mead(func, x, max_iter=30)
            if nm_fx < best_fx:
                best_fx = nm_fx
                best_x = nm_x
        
        # Phase 4: If budget remains, do another round of fine search
        if self.calls < self.budget - 5:
            region_size2 = region_size / 2
            fine_lo2 = np.maximum(best_x - region_size2, lo)
            fine_hi2 = np.minimum(best_x + region_size2, hi)
            fine_results2 = self._grid_scan(func, fine_lo2, fine_hi2, 4, top_k=1)
            if fine_results2:
                fx2, x2 = fine_results2[0]
                nm_fx2, nm_x2 = self._nelder_mead(func, x2, max_iter=20)
                if nm_fx2 < best_fx:
                    best_fx = nm_fx2
                    best_x = nm_x2
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
