    def greedy_construct(seed):
        """Fast greedy construction with vectorized gain."""
        rng = random.Random(seed)
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        selected_blocks = set()
        
        uncovered_count = num_t_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline - 20.0:
                break
                
            candidates = set()
            for _ in range(10):
                while True:
                    idx = rng.randint(0, num_t_subsets - 1)
                    if coverage[idx] == 0:
                        break
                for b_idx in t_subset_to_blocks[idx]:
                    if b_idx not in selected_blocks:
                        candidates.add(b_idx)
                        
            if not candidates:
                break
                
            cand_indices = np.array(list(candidates), dtype=np.int32)
            U = (coverage == 0)
            gains = incidence[cand_indices, :].dot(U)
            best_pos = np.argmax(gains)
            best_gain = gains[best_pos]
            
            if best_gain == 0:
                break
                
            best_block = cand_indices[best_pos]
            selected_blocks.add(best_block)
            coverage[block_t_indices[best_block]] += 1
            uncovered_count -= best_gain
            
        return list(selected_blocks)
