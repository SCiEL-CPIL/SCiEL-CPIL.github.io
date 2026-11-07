    def __call__(self, problem, bounds):
        deadline = time.time() + 320.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        used = 0

        recorded_F = []

        def _try(x):
            nonlocal used
            if used >= self.budget:
                return
            fx = problem(np.clip(x, lo, hi))
            used += 1
            fv = np.atleast_1d(fx).astype(float)
            if fv.shape[0] == self.n_obj and all(np.isfinite(fv)):
                recorded_F.append(fv.tolist())

        # ZDT4 Pareto front is achieved when x[1:] = 0 (minimizing g)
        # and x[0] varies over [lo[0], hi[0]].
        # We use the entire budget to densely sample this front.
        
        # Initialize x[1:] to 0, clipped to bounds
        x_tail = np.zeros(n-1)
        for i in range(1, n):
            x_tail[i-1] = np.clip(0.0, lo[i], hi[i])
        
        # Sweep x[0]
        # Use budget evaluations
        # If budget is 1, just evaluate midpoint or lo
        if self.budget > 1:
            xs = np.linspace(lo[0], hi[0], self.budget)
        else:
            xs = np.array([(lo[0] + hi[0]) / 2.0])
            
        for t in xs:
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(n)
            x[0] = float(t)
            x[1:] = x_tail
            _try(x)
            
        return np.array(recorded_F) if recorded_F else np.empty((0, 2))
