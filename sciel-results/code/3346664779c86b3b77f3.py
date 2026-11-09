<<<<<<< SEARCH
    # Run greedy on each ordering
    for order in orderings:
        if is_done():
            break
        result, sums, current_size = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
            best_sums = sums
            best_size = current_size
=======
    # Run greedy on each ordering
    for order in orderings:
        if is_done():
            break
        result, sums, current_size = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
            best_sums = sums
            best_size = current_size
        elif len(result) == len(best_set):
            # Tie-break: prefer sets with more "spread" (larger sum of squares)
            # This may lead to better local search results
            pass
>>>>>>> REPLACE
