    def __call__(self, func):
        self.start_time = time.time()
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = float('inf')
        
        def wrapped_func(x):
            if time.time() - self.start_time > 290:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            
            self.calls += 1
            fx = func(x)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        try:
            minimize(
                wrapped_func, 
                self.best_x, 
                method='CMA', 
                options={
                    'maxiter': 200,
                    'tolfun': 1e-14,
                    'sigma0': 1.0,
                    'verbose': 0
                }
            )
        except (TimeoutError, ValueError):
            pass
        except Exception:
            pass

        return self.best_x, self.best_f
