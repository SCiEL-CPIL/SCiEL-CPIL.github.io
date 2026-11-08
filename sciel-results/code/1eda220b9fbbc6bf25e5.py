    # Initial greedy solution to set a good lower bound
    random.shuffle(all_blocks)
    used_indices = set()
    greedy_packing = []
    for block_params, block_indices in all_blocks:
        if not (block_indices & used_indices):
            greedy_packing.append(block_params)
            used_indices |= block_indices
            
    best_num_blocks = len(greedy_packing)
    best_packing = greedy_packing[:]
