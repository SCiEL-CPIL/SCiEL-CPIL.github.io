    for combo in itertools.combinations(range(n), 6):
        if is_regular_subset(adj_masks, combo):
            return False, combo
