<<<<<<< SEARCH
    def greedy_cover(seed, max_time):
        """Run greedy set cover with randomization for tie-breaking."""
        rng = np.random.RandomState(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        remaining = n_tsubs
        used = np.zeros(n_ksubs, dtype=bool)
        selected = []
        t_start = time.time()

        while remaining > 0 and time.time() < max_time:
            # Vectorized gain computation: for each unused block, count how many uncovered t-subs it covers
            gains = (cov_matrix & uncovered[None, :]).sum(axis=1)
            gains[used] = 0

            # Find max gain
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all blocks with max gain (or top-k for randomization)
            top_k = min(20, n_ksubs)
            # Use random tie-breaking: add small random noise to top candidates
            candidates = np.where(gains >= max_gain)[0]
            if len(candidates) > top_k:
                # Randomly pick from top candidates
                chosen_idx = candidates[rng.randint(len(candidates))]
            else:
                chosen_idx = candidates[rng.randint(len(candidates))]

            selected.append(chosen_idx)
            used[chosen_idx] = True
            uncovered = uncovered & (1 - cov_matrix[chosen_idx])
            remaining = uncovered.sum()

        return selected
=======
    def greedy_cover(seed, max_time):
        """Run greedy set cover with incremental gain updates and randomized tie-breaking."""
        rng = np.random.RandomState(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        remaining = n_tsubs
        used = np.zeros(n_ksubs, dtype=bool)
        selected = []
        t_start = time.time()
        
        # Initialize gains
        gains = cov_matrix.sum(axis=1).astype(np.int32)
        
        while remaining > 0 and time.time() < max_time:
            # Find candidates with positive gain among unused blocks
            gains[used] = 0
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomized selection: prefer max gain, but with some randomness
            if len(candidates) > 1:
                # Weighted random: higher gain blocks more likely, but allow some diversity
                # Use a softmax-like selection among top candidates
                top_n = min(10, len(candidates))
                top_candidates = candidates[:top_n] if len(candidates) <= top_n else candidates[rng.choice(len(candidates), size=top_n, replace=False)]
                chosen_idx = top_candidates[rng.randint(len(top_candidates))]
            else:
                chosen_idx = candidates[0]

            selected.append(chosen_idx)
            used[chosen_idx] = True
            
            # Update gains incrementally: for each unused block, subtract overlap with newly selected block
            new_block_cov = cov_matrix[chosen_idx]
            # Only update gains for unused blocks
            unused_mask = ~used
            overlap = (cov_matrix[unused_mask] & new_block_cov[None, :]).sum(axis=1)
            gains_indices = np.where(unused_mask)[0]
            gains[gains_indices] -= overlap
            
            uncovered = uncovered & (1 - new_block_cov)
            remaining = uncovered.sum()

        return selected
>>>>>>> REPLACE

