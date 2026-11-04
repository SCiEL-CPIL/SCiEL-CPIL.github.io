<<<<<<< SEARCH
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=np.int32)
        gains = np.array(M.sum(axis=1)).flatten().astype(np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                continue
            
            for s in newly_covered:
                uncovered[s] = 0
                for b in subset_to_blocks[s]:
                    gains[b] -= 1
                    
        return selected_blocks
=======
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=np.int32)
        gains = (M @ uncovered).flatten().astype(np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                continue
            
            for s in newly_covered:
                uncovered[s] = 0
            # Recompute gains using sparse matrix multiplication (vectorized)
            gains = (M @ uncovered).flatten().astype(np.int32)
                    
        return selected_blocks
>>>>>>> REPLACE

