"""Optimizer using direct search pattern search with fixed directions and binary step-size search."""
import numpy as np
import time

class Optimizer:
    """Direct search: fixed orthogonal + diagonal directions, binary step-size search."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._build_directions()
    
    def _build_directions(self):
        """Build fixed set of search directions: axes + diagonals."""
        d = self.dim
        dirs = []
        # Axis directions (positive and negative)
        for i in range(d):
            v = np.zeros(d)
            v[i] = 1.0
            dirs.append(v.copy())
            dirs.append(-v.copy())
        # Diagonal directions (all +1, all -1, and some mixed)
        v = np.ones(d)
        dirs.append(v / np.linalg.norm(v))
        dirs.append(-v / np.linalg.norm(v))
        # Some random-ish fixed diagonals for better coverage
        for k in range(min(d, 3)):
            v = np.zeros(d)
            v[k] = 1.0
            v[(k+1) % d] = 1.0
            v[(k+2) % d] = -1.0
            dirs.append(v / np.linalg.norm(v))
        self.dirs = np.array(dirs)
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _pattern_search_step(self, func, x, fx, step_size, total_calls, deadline):
        """
        Try each direction with binary search on step size.
        Returns updated x, fx, step_size, total_calls.
        """
        d = self.dim
        improved = False
        
        for dir_vec in self.dirs:
            if total_calls >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            # Binary search for best step in this direction
            # Try a few step sizes: large, medium, small
            best_step = 0.0
            best_f = fx
            best_x = x.copy()
            
            # Try positive direction first
            for trial in range(3):
                step = step_size * (0.5 ** trial)
                if step < 1e-10:
                    break
                x_try = self._clip(x + step * dir_vec)
                if np.allclose(x_try, x, atol=1e-12):
                    continue
                f_try = func(x_try)
                total_calls += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    best_step = step
            
            # Try negative direction
            for trial in range(3):
                step = step_size * (0.5 ** trial)
                if step < 1e-10:
                    break
                x_try = self._clip(x - step * dir_vec)
                if np.allclose(x_try, x, atol=1e-12):
                    continue
                f_try = func(x_try)
                total_calls += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    best_step = step
            
            if best_step > 0:
                x = best_x
                fx = best_f
                improved = True
                # Keep step size or slightly increase if we made good progress
                if best_step == step_size:
                    step_size = min(step_size * 1.2, 10.0)
                else:
                    step_size = max(step_size * 0.9, 1e-6)
        
        if not improved:
            step_size = max(step_size * 0.5, 1e-6)
        
        return x, fx, step_size, total_calls
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Sparse random initialization to find good starting regions
        n_initial = min(15, budget // 6)
        initial_points = []
        initial_values = []
        
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        total_calls += 1
        initial_points.append(x0.copy())
        initial_values.append(f0)
        best_x, best_f = x0.copy(), f0
        
        for i in range(n_initial - 1):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            xi = rng.uniform(-4.5, 4.5, size=dim)
            fi = func(xi)
            total_calls += 1
            initial_points.append(xi.copy())
            initial_values.append(fi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Phase 2: Pattern search from multiple diverse starting points
        # Select top-k starting points
        k = min(6, len(initial_points))
        order = np.argsort(initial_values)
        selected_starts = [initial_points[order[i]].copy() for i in range(k)]
        
        # Add a couple more diverse random points
        while len(selected_starts) < 8 and total_calls < budget - 20:
            x_new = rng.uniform(-4.0, 4.0, size=dim)
            too_close = False
            for s in selected_starts:
                if np.linalg.norm(x_new - s) < 1.0:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x_new)
        
        # Budget allocation: give more budget to better starting points
        # But ensure each gets at least some budget
        remaining_budget = budget - total_calls
        calls_per_run = max(10, remaining_budget // len(selected_starts))
        
        for start_idx, x_start in enumerate(selected_starts):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            
            remaining = budget - total_calls
            max_calls_this = min(calls_per_run, remaining)
            
            if max_calls_this < 5:
                continue
            
            # Check if we already have this point's value
            fx = None
            for j, p in enumerate(initial_points):
                if np.allclose(x_start, p, atol=1e-10):
                    fx = initial_values[j]
                    break
            
            if fx is None:
                fx = func(x_start)
                total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x_start.copy()
            
            x = x_start.copy()
            step_size = 2.0  # Initial step size for pattern search
            
            # Local pattern search loop
            while total_calls < budget:
                if time.time() > deadline - 5.0:
                    break
                
                x, fx, step_size, total_calls = self._pattern_search_step(
                    func, x, fx, step_size, total_calls, deadline
                )
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # If step size becomes very small, we've converged locally
                if step_size < 1e-5:
                    break
        
        return best_x, best_f
