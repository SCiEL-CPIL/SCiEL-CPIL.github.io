    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    steps_per_temp = 200
    
    while T > T_min:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                return best_h
            # ... move logic ...
        T *= cooling_rate
