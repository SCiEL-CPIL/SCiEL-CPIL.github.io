    # After B&B
    if time.time() < deadline - 5.0:
        # Local search polish
        current = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any element
            for x in range(1, n + 1):
                if x not in current:
                    if can_add(x, current):
                        current.add(x)
                        improved = True
                        break
        if len(current) > len(best_set):
            best_set = current
