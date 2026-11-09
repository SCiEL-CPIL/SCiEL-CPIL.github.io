<<<<<<< SEARCH
    def multi_start_local_search(elements, time_budget=20):
        """
        Run multiple greedy starts with different orderings, then improve.
        """
        orderings = [
            lambda x: -abs(x),
            lambda x: x,
            lambda x: -x,
            lambda x: abs(x),
            lambda x: (0 if x > 0 else 1, -abs(x)),
            lambda x: (1 if x > 0 else 0, abs(x)),
            lambda x: (0 if x > 0 else 1, abs(x)),
            lambda x: (1 if x > 0 else 0, -abs(x)),
        ]

        best_subset = None
        best_size = -1

        per_order_budget = time_budget / (len(orderings) + 3)

        for order in orderings:
            if time.time() - start_time > timeout:
                break
            t0 = time.time()
            subset = greedy_sum_free(elements, order_key=order)

            # Improve with remove-and-retry
            if time.time() - t0 < per_order_budget * 0.4:
                improved = remove_and_retry_improve(elements, subset, max_rounds=30)
                if len(improved) > len(subset):
                    subset = improved

            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = subset

        # Final polish with simulated annealing on the best
        if best_subset is not None:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 5:
                sa_budget = min(remaining_time - 2, 20)
                if sa_budget > 2:
                    polished = simulated_annealing_improve(elements, best_subset, time_budget=sa_budget)
                    if len(polished) > best_size:
                        best_subset = polished
                        best_size = len(polished)

        return best_subset if best_subset is not None else []
=======
    def multi_start_local_search(elements, time_budget=20):
        """
        Run multiple greedy starts with different orderings, then improve.
        """
        orderings = [
            lambda x: -abs(x),
            lambda x: x,
            lambda x: -x,
            lambda x: abs(x),
            lambda x: (0 if x > 0 else 1, -abs(x)),
            lambda x: (1 if x > 0 else 0, abs(x)),
            lambda x: (0 if x > 0 else 1, abs(x)),
            lambda x: (1 if x > 0 else 0, -abs(x)),
        ]

        best_subset = None
        best_size = -1

        per_order_budget = time_budget / (len(orderings) + 3)

        for order in orderings:
            if time.time() - start_time > timeout:
                break
            t0 = time.time()
            subset = greedy_sum_free(elements, order_key=order)

            # Improve with remove-and-retry
            if time.time() - t0 < per_order_budget * 0.4:
                improved = remove_and_retry_improve(elements, subset, max_rounds=30)
                if len(improved) > len(subset):
                    subset = improved

            # Apply greedy polish
            subset = greedy_polish(elements, subset, max_passes=5)

            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = subset

        # Final polish with simulated annealing on the best
        if best_subset is not None:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 5:
                sa_budget = min(remaining_time - 2, 20)
                if sa_budget > 2:
                    polished = simulated_annealing_improve(elements, best_subset, time_budget=sa_budget)
                    # Apply final greedy polish
                    polished = greedy_polish(elements, polished, max_passes=5)
                    if len(polished) > best_size:
                        best_subset = polished
                        best_size = len(polished)

        return best_subset if best_subset is not None else []
>>>>>>> REPLACE
