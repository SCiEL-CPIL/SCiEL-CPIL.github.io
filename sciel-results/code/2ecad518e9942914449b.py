    def greedy_cover():
        uncovered = full_mask
        selected_blocks = []
        remaining_blocks = set(range(n_blocks))
        
        while uncovered and remaining_blocks:
            best_gain = -1
            best_candidates = []
            
            for b_idx in remaining_blocks:
                gain = bin(block_masks[b_idx] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
                
            # Pick randomly from best candidates
            best_block_idx = rng.choice(best_candidates)
            
            selected_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            remaining_blocks.discard(best_block_idx)
            
        return selected_blocks
