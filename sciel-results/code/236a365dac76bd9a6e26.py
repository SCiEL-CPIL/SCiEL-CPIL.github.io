def run_11_es(func, x_start, budget, dim, sigma_init=2.0):
    x = x_start.copy()
    f_x = func(x)
    sigma = sigma_init
    successes = 0
    total = 0
    window = 10
    recent_successes = 0
    recent_total = 0
    
    for _ in range(budget):
        # Adapt sigma
        if recent_total >= window:
            rate = recent_successes / recent_total
            if rate > 0.2:
                sigma *= 1.1
            elif rate < 0.2:
                sigma *= 0.9
            recent_successes = 0
            recent_total = 0
            
        # Generate neighbor
        z = np.random.randn(dim)
        y = x + sigma * z
        y = np.clip(y, -5, 5)
        
        f_y = func(y)
        
        recent_total += 1
        if f_y < f_x:
            x = y
            f_x = f_y
            successes += 1
            recent_successes += 1
            
    return x, f_x
