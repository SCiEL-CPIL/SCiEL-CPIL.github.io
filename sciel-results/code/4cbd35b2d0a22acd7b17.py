"""Multi-start Nelder-Mead with dynamic budget allocation for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer:
    1. Launches 10 independent Nelder-Mead runs from diverse starting points.
    2. Dynamically allocates evaluation budget based on convergence rates.
    3. Polishes the best result with tighter tolerances.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.shape != (n,):
                x_arr = x_arr.reshape(-1)
                if x_arr.size != n:
                    raise ValueError("Wrong dimension")
            f_val = float(func(x_arr))
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        n_starts = 10
        start_points = []
        
        # Include origin and some structured points
        start_points.append(np.zeros(n))
        start_points.append(rng.uniform(-4, 4, n))
        start_points.append(rng.uniform(-4, 4, n))
        start_points.append(np.array([4.0] * n))
        start_points.append(np.array([-4.0] * n))
        start_points.append(rng.uniform(-3, 3, n))
        start_points.append(rng.uniform(-3, 3, n))
        start_points.append(rng.uniform(-2, 2, n))
        start_points.append(rng.uniform(-5, 5, n))
        start_points.append(rng.uniform(-5, 5, n))
        
        # Clip to box
        start_points = [np.clip(p, -5, 5) for p in start_points]
        
        # Track state for each start
        states = []
        for i, sp in enumerate(start_points):
            states.append({
                'x0': sp,
                'calls_used': 0,
                'best_f': np.inf,
                'best_x': sp.copy(),
                'last_f': np.inf,
                'improvement_rate': 0.0,
                'active': True
            })
        
        # Allocate budget: 70% for multi-start, 30% for polishing
        multi_start_budget = int(self.budget * 0.70)
        polish_budget = self.budget - multi_start_budget
        
        # Phase 1: Multi-start Nelder-Mead with dynamic allocation
        # Each iteration, allocate budget to the most promising starts
        while calls < multi_start_budget and time.time() < deadline:
            # Check if any active starts remain
            active_states = [s for s in states if s['active']]
            if not active_states:
                break
            
            # Calculate priority based on improvement rate and current best
            for s in states:
                if not s['active']:
                    continue
                # Estimate convergence rate: how much improvement per call
                if s['calls_used'] > 0:
                    s['improvement_rate'] = (s['last_f'] - s['best_f']) / max(1, s['calls_used'])
                else:
                    s['improvement_rate'] = 0.0
            
            # Sort active states by promise (lower best_f, higher improvement rate)
            active_states.sort(key=lambda s: (s['best_f'], -s['improvement_rate']))
            
            # Allocate next batch of calls: give more to top performers
            # Use a simple strategy: round-robin with bias toward top performers
            batch_size = min(10, multi_start_budget - calls)
            
            # Give top 3 starts 2x the calls of others
            if len(active_states) >= 3:
                allocation = [2, 2, 2] + [1] * (len(active_states) - 3)
            else:
                allocation = [2] * len(active_states)
            
            # Normalize allocation to fit batch_size
            total_alloc = sum(allocation)
            if total_alloc == 0:
                break
            allocation = [max(1, int(a * batch_size / total_alloc)) for a in allocation]
            # Adjust to fit exactly
            while sum(allocation) > batch_size:
                # Reduce from the end
                for i in range(len(allocation) - 1, -1, -1):
                    if allocation[i] > 1:
                        allocation[i] -= 1
                        if sum(allocation) == batch_size:
                            break
            while sum(allocation) < batch_size and active_states:
                # Add to the best performer
                allocation[0] += 1
            
            # Run Nelder-Mead for each active state with allocated calls
            for idx, s in enumerate(active_states):
                if idx >= len(allocation):
                    break
                if calls >= multi_start_budget or time.time() > deadline:
                    break
                if not s['active']:
                    continue
                
                alloc = allocation[idx]
                if alloc <= 0:
                    continue
                
                # Set max iterations for this run
                max_iter = max(1, alloc // (n + 3))
                
                try:
                    result = scipy.optimize.minimize(
                        safe_func,
                        s['best_x'],
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-4,
                            'fatol': 1e-4,
                            'adaptive': True
                        }
                    )
                    
                    # Update state
                    s['calls_used'] += alloc
                    s['last_f'] = result.fun
                    if result.fun < s['best_f']:
                        s['best_f'] = result.fun
                        s['best_x'] = result.x.copy()
                    
                    # Mark as inactive if it converged or made no progress
                    if result.fun >= s['last_f'] - 1e-6:
                        s['active'] = False
                    elif s['calls_used'] > 50:
                        s['active'] = False
                        
                except (TimeoutError, StopIteration):
                    s['active'] = False
                    break
                except Exception:
                    s['active'] = False
        
        # Phase 2: Polish the best result
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            if remaining_calls > 5:
                # Use L-BFGS-B if available, else Nelder-Mead
                max_iter = max(5, remaining_calls // (n + 3))
                
                # First try with moderate tolerance
                try:
                    result = scipy.optimize.minimize(
                        safe_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-6,
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
                
                # Second polish with tighter tolerance if budget remains
                if calls < self.budget - 3 and time.time() < deadline:
                    remaining = self.budget - calls
                    if remaining > 3:
                        max_iter2 = max(3, remaining // (n + 3))
                        try:
                            result2 = scipy.optimize.minimize(
                                safe_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxiter': max_iter2,
                                    'xatol': 1e-8,
                                    'fatol': 1e-8,
                                    'adaptive': True
                                }
                            )
                        except (TimeoutError, StopIteration, Exception):
                            pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
