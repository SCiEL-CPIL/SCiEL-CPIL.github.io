<<<<<<< SEARCH
    # If we have time, do ILS on the best solution
    if best_solution is not None and time.time() < deadline - 10:
        current_best = list(best_solution)
        
        # Perturb and Repair
        for _ in range(5): # 5 ILS iterations
            if time.time() > deadline - 5:
                break
            
            # Perturb: Remove 5 random blocks
            perturbed = list(current_best)
            if len(perturbed) > 10:
                remove_indices = random.sample(range(len(perturbed)), 5)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    perturbed.pop(idx)
            
            # Repair: Greedy to cover gaps
            # We need to know what is uncovered by 'perturbed'
            cover_count = np.zeros(num_tsubs, dtype=np.int32)
            for b in perturbed:
                for s in block_to_tsubs[b]:
                    cover_count[s] += 1
            
            uncovered_mask = cover_count == 0
            
            # Run a simple greedy to fill gaps
            # We can reuse the logic from greedy_construction but start with 'perturbed'
            # For simplicity, we'll just add blocks greedily
            selected_set = set(perturbed)
            
            # Calculate gains for all blocks relative to current uncovered
            gains = np.zeros(num_blocks, dtype=np.int32)
            for b_idx in range(num_blocks):
                if b_idx in selected_set:
                    continue
                gain = 0
                for s in block_to_tsubs[b_idx]:
                    if uncovered_mask[s]:
                        gain += 1
                gains[b_idx] = gain
            
            while uncovered_mask.any():
                if time.time() > deadline - 2:
                    break
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                # Pick one with highest scarcity
                cand_scarcity = block_scarcity[candidates]
                b = candidates[np.argmax(cand_scarcity)]
                
                selected_set.add(b)
                for s in block_to_tsubs[b]:
                    if uncovered_mask[s]:
                        uncovered_mask[s] = False
                        gains[tsub_to_blocks_np[s]] -= 1
            
            # Local Optimize the result
            result_blocks = list(selected_set)
            optimized = local_optimize(result_blocks)
            
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized
                current_best = optimized
=======
    # If we have time, do ILS on the best solution
    if best_solution is not None and time.time() < deadline - 10:
        current_best = list(best_solution)
        
        # Perturb and Repair
        for _ in range(8): # 8 ILS iterations
            if time.time() > deadline - 5:
                break
            
            # Perturb: Remove 5 random blocks
            perturbed = list(current_best)
            if len(perturbed) > 10:
                remove_indices = random.sample(range(len(perturbed)), 5)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    perturbed.pop(idx)
            
            # Repair: Greedy to cover gaps
            cover_count = np.zeros(num_tsubs, dtype=np.int32)
            for b in perturbed:
                for s in block_to_tsubs[b]:
                    cover_count[s] += 1
            
            uncovered_mask = cover_count == 0
            
            selected_set = set(perturbed)
            
            # Vectorized gain computation for repair
            gains = np.zeros(num_blocks, dtype=np.int32)
            for s_idx in np.where(uncovered_mask)[0]:
                gains[tsub_to_blocks_np[s_idx]] += 1
            
            while uncovered_mask.any():
                if time.time() > deadline - 2:
                    break
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                cand_scarcity = block_scarcity[candidates]
                b = candidates[np.argmax(cand_scarcity)]
                
                selected_set.add(b)
                for s in block_to_tsubs[b]:
                    if uncovered_mask[s]:
                        uncovered_mask[s] = False
                        gains[tsub_to_blocks_np[s]] -= 1
            
            # Local Optimize the result
            result_blocks = list(selected_set)
            optimized = local_optimize(result_blocks)
            
            if len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized
                current_best = optimized

    # Final polish pass: one more local optimization on the best solution
    if best_solution is not None and time.time() < deadline - 3:
        polished = local_optimize(best_solution)
        if len(polished) < best_len:
            best_len = len(polished)
            best_solution = polished
>>>>>>> REPLACE

