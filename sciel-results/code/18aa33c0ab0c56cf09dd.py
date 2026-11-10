    while time.time() < _HARD_DEADLINE - 2.0:
        # Pick a move
        k = rng.randint(1, n-1)
        axis_vec = tuple(current_chain[k][i] - current_chain[k-1][i] for i in range(dimension))
        angle = rng.choice([90, 180, 270])
        
        new_chain = _pivot_move(current_chain, k, axis_vec, angle, dimension)
