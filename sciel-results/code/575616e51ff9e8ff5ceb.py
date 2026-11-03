<<<<<<< SEARCH
        # Random init
        if time_left() < 10:
            break
        circles0 = random_init(w, h, seed=init_count)
        init_count += 1
        try_candidate(circles0, w, h, budget=12, best=best_value)
=======
        # Random init
        if time_left() < 10:
            break
        circles0 = random_init(w, h, seed=init_count)
        init_count += 1
        try_candidate(circles0, w, h, budget=12, best=best_value)

        # Farthest-point sampling init (diverse structure)
        if time_left() < 10:
            break
        circles0 = farthest_point_init(w, h, seed=init_count)
        init_count += 1
        try_candidate(circles0, w, h, budget=12, best=best_value)
>>>>>>> REPLACE

