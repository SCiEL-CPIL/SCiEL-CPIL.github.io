        # Initial exploration: evaluate 5 random points
        candidates = rng.uniform(-3, 3, (5, dim))
        for i in range(5):
            fx_cand = func(candidates[i])
            calls += 1
            if fx_cand < fx:
                x = candidates[i]
                fx = fx_cand
                best_x = x.copy()
                best_fx = fx
        sigma = 1.0
