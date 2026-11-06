<<<<<<< SEARCH
        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=5000,
                                       time_limit=local_search_time * 0.6, rng=rng2)
=======
        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=10000,
                                       time_limit=local_search_time * 0.8, rng=rng2)
>>>>>>> REPLACE
