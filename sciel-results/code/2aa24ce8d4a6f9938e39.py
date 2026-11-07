class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        max_time = 325.0
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        phase_len = 20
        
        while calls_made < self.budget:
            if time.time() > max_time:
                break
            
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 1.0
            successes = 0
            window_size = 10
            window_count = 0
            
            for i in range(phase_len - 1):
                if calls_made >= self.budget or time.time() > max_time:
                    break
                
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass # x, fx unchanged
                
                window_count += 1
                if window_count == window_size:
                    # 1/5 rule
                    if successes > window_size / 5.0:
                        sigma *= 1.5
                    else:
                        sigma *= 0.87
                    successes = 0
                    window_count = 0

        return best_x, best_f
