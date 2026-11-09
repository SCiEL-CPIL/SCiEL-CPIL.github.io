<<<<<<< SEARCH
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2)
=======
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(123)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng2, max_iters=60)
>>>>>>> REPLACE
