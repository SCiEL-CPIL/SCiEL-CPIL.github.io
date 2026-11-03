    uncovered_count = num_subsets
    # ...
    while uncovered_count > 0:
        # ...
        # after adding block:
        for sub_idx in block_to_subsets[best_block]:
            if uncovered[sub_idx]:
                uncovered[sub_idx] = False
                uncovered_count -= 1
