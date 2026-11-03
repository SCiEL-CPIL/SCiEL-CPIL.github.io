    old_set = set(all_blocks[old_idx])
    new_set = set(old_set)
    # Remove 1 element
    rem = rng.choice(list(new_set))
    new_set.remove(rem)
    # Add 1 random element
    add = rng.randrange(v)
    while add in new_set:
        add = rng.randrange(v)
    new_set.add(add)
    new_block = tuple(sorted(new_set))
    new_idx = block_to_idx.get(new_block, -1)
    if new_idx == -1:
        # Fallback to random
        new_idx = rng.randrange(n_blocks)
    