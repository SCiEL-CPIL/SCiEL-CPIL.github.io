<<<<<<< SEARCH
        T *= cooling
        if T < min_T:
            T = min_T
=======
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Simple reheat if stuck
        if len(current) == best_len and T < 5.0 and rng.random() < 0.05:
            T = 10.0
>>>>>>> REPLACE
