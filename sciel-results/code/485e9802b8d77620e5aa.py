<<<<<<< SEARCH
    # Try ranks: 30, 35, 40, 45, 50, 55, 60, 63
    for rank in [30, 35, 40, 45, 50, 55, 60, 63]:
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            break
=======
    # Try rank 1 first, as the target might be the all-ones tensor
    # If the target is the all-ones tensor, rank 1 is the solution.
    ranks_to_try = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 63]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            break
>>>>>>> REPLACE
