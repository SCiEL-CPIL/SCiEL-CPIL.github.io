    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        d = self.dim
        total_budget = self.budget

        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return func(x)

        rng = np.random.default_rng(123)

        best_x = np.zeros(d)
        best_f = np.inf

        n_restarts = 5
        budget_per_restart = 80
        
        for _ in range(n_restarts):
            if calls[0] >= total_budget:
                break
            if time.time() > deadline - 10:
                break
                
            x0 = rng.uniform(-4, 4, d)
            
            res_x, res_f = self._run_cmaes(wrapped, x0, budget_per_restart, deadline, calls, rng)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        remaining_budget = total_budget - calls[0]
        if remaining_budget > 10 and time.time() < deadline - 5:
            res_x, res_f = self._run_cmaes(wrapped, best_x, remaining_budget, deadline, calls, rng, sigma0=0.5)
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()

        return best_x, best_f
