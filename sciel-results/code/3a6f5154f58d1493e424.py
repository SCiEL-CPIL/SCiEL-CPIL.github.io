    def __call__(self, func):
        rng = np.random.default_rng(0)
        calls = 0
        best_x = None
        best_f = float('inf')
        
        # Deadline guard variables
        start_time = time.time()
        deadline = 290.0  # Safety margin

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return float('inf')
            if calls >= self.budget:
                return float('inf')
            
            calls += 1
            try:
                val = func(x)
                if not np.isfinite(val):
                    val = float('inf')
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Scout
        # Evaluate nominal minimum (1,1,1,1,1) and origin (0,0,0,0,0)
        # And a few random points to handle large shifts
        scout_points = [
            np.ones(self.dim),
            np.zeros(self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim)
        ]
        
        for x in scout_points:
            if calls >= self.budget or time.time() > deadline:
                break
            wrapped_func(np.clip(x, -5, 5))

        # Phase 2: Local Search
        if best_x is not None and calls < self.budget and time.time() < deadline:
            # Use Powell from the best scout point
            # Allocate most of the remaining budget to this
            maxfev = self.budget - calls
            try:
                minimize(
                    wrapped_func, 
                    best_x, 
                    method='Powell', 
                    options={'maxfev': maxfev, 'xtol': 1e-10, 'ftol': 1e-10}
                )
            except Exception:
                pass

        # Phase 3: If there is significant budget left (Powell stopped early), try a restart
        if best_x is not None and calls < self.budget - 10 and time.time() < deadline:
            # Perturb best point
            start_x = best_x + 0.5 * rng.standard_normal(self.dim)
            start_x = np.clip(start_x, -5, 5)
            maxfev = self.budget - calls
            try:
                minimize(
                    wrapped_func, 
                    start_x, 
                    method='Powell', 
                    options={'maxfev': maxfev, 'xtol': 1e-10, 'ftol': 1e-10}
                )
            except Exception:
                pass

        return best_x, best_f
