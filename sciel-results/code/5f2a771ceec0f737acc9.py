<<<<<<< SEARCH
    # Phase 2: Multi-start with random orderings
    rng = random.Random(42)
    num_starts = 200  # we'll do as many as time allows
    
    for trial in range(num_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
=======
    # Phase 2: Multi-start with random orderings
    rng = random.Random(42)
    num_starts = 200  # we'll do as many as time allows
    
    for trial in range(num_starts):
        if time.time() - start_time > DEADLINE - 8:
            break
>>>>>>> REPLACE
