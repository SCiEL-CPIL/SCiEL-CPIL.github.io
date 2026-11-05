    def run_beam_search(seed):
        rng = random.Random(seed)
        shuffled_blocks = block_data.copy()
        rng.shuffle(shuffled_blocks)
        
        beam = [(full_mask, [])]
        best_solution = None
        best_len = float('inf')
        
        step = 0
        while beam:
            # If we already have a solution with length <= step + 1, 
            # and we are expanding step + 1, we might still find a better one?
            # No, if we found a solution of length L, any new solution will have length > L?
            # Not necessarily, another branch might have found a solution of length L earlier?
            # We track best_len. If step + 1 >= best_len, we can stop?
            # Yes, because we are generating solutions of length step + 1.
            if step + 1 >= best_len:
                break
                
            next_beam = []
            for uncovered, blocks in beam:
                if uncovered == 0:
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_solution = blocks
                    continue
                    
                best_gain = -1
                best_block = None
                best_mask = None
                
                for block, mask in shuffled_blocks:
                    inter = mask & uncovered
                    if not inter:
                        continue
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                        best_mask = mask
                        
                if best_block is None:
                    continue
                    
                next_beam.append((uncovered & ~best_mask, blocks + [best_block]))
                
            # Deduplicate and Sort
            seen = set()
            unique_next_beam = []
            for state in next_beam:
                key = state[0]
                if key not in seen:
                    seen.add(key)
                    unique_next_beam.append(state)
                    
            unique_next_beam.sort(key=lambda x: (len(x[1]), x[0].bit_count()))
            beam = unique_next_beam[:beam_width]
            step += 1
            
            if time.time() > max_time:
                break
                
        return best_solution
