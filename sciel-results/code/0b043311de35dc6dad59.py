<<<<<<< SEARCH
    # Random restarts
    rng = random.Random(123)
    max_restarts = 300

    for _ in range(max_restarts):
        if check_deadline():
            break
        # Randomized order: sort by volume with random perturbation
        shuffled = list(indexed)
        shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
        n = len(shuffled)
        # Random swaps
        for _ in range(n // 2):
            i = rng.randint(0, n - 2)
            shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        order_r = [t[0] for t in shuffled]

        if grid is not None:
            import numpy as np
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res_r = pack_order(order_r, boxes, W, H, D, g)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r
=======
    # Random restarts
    rng = random.Random(123)
    max_restarts = 500

    for r in range(max_restarts):
        if check_deadline():
            break
        # Multiple randomization strategies
        strategy = r % 5
        shuffled = list(indexed)
        n = len(shuffled)

        if strategy == 0:
            # Volume sort with random swaps
            shuffled.sort(key=lambda t: -t[1]*t[2]*t[3])
            for _ in range(n // 2):
                i = rng.randint(0, n - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        elif strategy == 1:
            # Max dimension sort with random swaps
            shuffled.sort(key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
            for _ in range(n // 3):
                i = rng.randint(0, n - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        elif strategy == 2:
            # Surface area sort with random swaps
            shuffled.sort(key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))
            for _ in range(n // 2):
                i = rng.randint(0, n - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        elif strategy == 3:
            # Base area sort with random swaps
            shuffled.sort(key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))
            for _ in range(n // 2):
                i = rng.randint(0, n - 2)
                shuffled[i], shuffled[i+1] = shuffled[i+1], shuffled[i]
        else:
            # Random shuffle
            rng.shuffle(shuffled)

        order_r = [t[0] for t in shuffled]

        if grid is not None:
            import numpy as np
            g = np.zeros((W, H, D), dtype=bool)
        else:
            g = None
        res_r = pack_order(order_r, boxes, W, H, D, g)
        vol_r = filled_volume(res_r)
        if vol_r > best_vol:
            best_vol = vol_r
            best_result = res_r
>>>>>>> REPLACE

