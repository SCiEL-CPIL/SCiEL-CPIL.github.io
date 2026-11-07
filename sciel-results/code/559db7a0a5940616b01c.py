    def __call__(self, func):
        t0 = time.time()
        self._deadline = t0 + 280.0
        dim = self.dim
        calls = [0]
        budget = self.budget

        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate integer lattice points within [-4,4]^5
        # These are candidates for the true minimum of Rastrigin (before rotation/shift)
        # After rotation and shift, they won't be exact minima, but nearby points will be
        # This gives us a strong early baseline
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    for l in range(-4, 5):
                        for m in range(-4, 5):
                            lattice_points.append(np.array([i, j, k, l, m], dtype=float))

        # Too many points (9^5 = 59049), so sample strategically
        # First evaluate center and axis-aligned neighbors
        priority_points = [np.zeros(dim)]
        # Add points at distance 1 from origin in each direction
        for i in range(dim):
            for sign in (-1, 1):
                p = np.zeros(dim)
                p[i] = sign
                priority_points.append(p)

        # Evaluate these first
        for p in priority_points:
            if calls[0] >= budget:
                break
            f = safe_func(p)
            record(p, f)

        # Phase 2: Based on best so far, do local CMA-ES refinement
        # The true minimum is within [-4,4]^5, so if we found a good point,
        # refine locally with small sigma
        if calls[0] < budget - 20:
            # Try a few local refinements around the best point
            for trial in range(3):
                if calls[0] >= budget - 10 or time.time() > self._deadline - 20:
                    break
                # Start with decreasing sigma
                sigma0 = 1.0 / (trial + 1)
                xc, fc = self._local_cmaes(safe_func, best_x, sigma0, calls, max_gens=15, pop_size=6)
                record(xc, fc)

        # Phase 3: If we still have budget, try a broader search from best
        if calls[0] < budget - 30 and time.time() < self._deadline - 30:
            # Do a medium-scale CMA-ES centered on best
            sigma0 = 2.0
            xc, fc = self._local_cmaes(safe_func, best_x, sigma0, calls, max_gens=25, pop_size=10)
            record(xc, fc)

        # Phase 4: Final fine-tuning
        if calls[0] < budget - 5 and time.time() < self._deadline - 10:
            sigma0 = 0.1
            xc, fc = self._local_cmaes(safe_func, best_x, sigma0, calls, max_gens=10, pop_size=4)
            record(xc, fc)

        return best_x, best_f
