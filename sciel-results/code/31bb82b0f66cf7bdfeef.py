            # Reduction Phase using counts
            solution = list(current_blocks)
            # Initialize counts
            counts = [0] * num_t_subsets
            for b in solution:
                m = block_to_mask[b]
                # Extract indices from mask? 
                # It's easier to iterate through the 5-subsets of the block directly.
                for t_sub in itertools.combinations(b, t):
                    idx = subset_to_idx[t_sub]
                    counts[idx] += 1
            
            changed = True
            while changed:
                changed = False
                indices = list(range(len(solution)))
                random.shuffle(indices)
                for i in indices:
                    if i >= len(solution):
                        continue
                    b = solution[i]
                    # Check if all subsets in b have count > 1
                    redundant = True
                    for t_sub in itertools.combinations(b, t):
                        idx = subset_to_idx[t_sub]
                        if counts[idx] <= 1:
                            redundant = False
                            break
                    
                    if redundant:
                        # Remove b
                        solution.pop(i)
                        # Decrement counts
                        for t_sub in itertools.combinations(b, t):
                            idx = subset_to_idx[t_sub]
                            counts[idx] -= 1
                        changed = True
                        break # Restart loop
