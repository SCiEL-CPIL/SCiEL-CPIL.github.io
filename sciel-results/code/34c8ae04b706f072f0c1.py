if len(to_remove) > 0:
    in_set[to_remove] = False
    # Update count_conflicts
    all_nbrs = np.concatenate([neighbors[i] for i in to_remove])
    np.add.at(count_conflicts, all_nbrs, -1)
