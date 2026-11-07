    def dfs_match(idx, used_mask, current_gain, assignment):
        # Pruning: if current_gain + sum of top (m-idx) gains <= best_gain, prune
        # We need precomputed max gains for remaining bases.
        ...
    