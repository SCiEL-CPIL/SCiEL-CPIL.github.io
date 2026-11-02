    # Convert best_blocks (indices) to actual blocks
    final_blocks = [list(all_k_subsets[i]) for i in best_blocks]
    return final_blocks
