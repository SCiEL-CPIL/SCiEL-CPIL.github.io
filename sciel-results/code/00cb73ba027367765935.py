<<<<<<< SEARCH
    def greedy_cover(seed):
        random.seed(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        
        while not np.all(covered):
            # Calculate gain for each block
            # gain[i] = number of uncovered triples in block i
            # We can vectorize this:
            # uncovered_flags = ~covered
            # gains = np.sum(uncovered_flags[block_t_indices], axis=1)
            
            uncovered_flags = ~covered
            gains = np.sum(uncovered_flags[block_t_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            blocks_indices.append(best_idx)
            covered[block_t_indices[best_idx]] = True
            
        return blocks_indices
=======
    def greedy_cover(seed, deadline=None):
        random.seed(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        
        while not np.all(covered):
            if deadline is not None and time.time() > deadline:
                break
                
            # Calculate gain for each block
            # gain[i] = number of uncovered triples in block i
            # We can vectorize this:
            # uncovered_flags = ~covered
            # gains = np.sum(uncovered_flags[block_t_indices], axis=1)
            
            uncovered_flags = ~covered
            gains = np.sum(uncovered_flags[block_t_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            blocks_indices.append(best_idx)
            covered[block_t_indices[best_idx]] = True
            
        return blocks_indices
>>>>>>> REPLACE

