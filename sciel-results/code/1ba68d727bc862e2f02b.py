        while len(block) < k:
            best_gain = -1
            best_point = -1
            best_freq = -1
            
            # Precompute masks for (t-1)-subsets of current block
            block_comb_masks = []
            if t - 1 > 0:
                for comb in itertools.combinations(block, t - 1):
                    mask_comb = 0
                    for p in comb:
                        mask_comb |= (1 << p)
                    block_comb_masks.append(mask_comb)
            else:
                block_comb_masks = [0] # Empty mask

            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                for mask_comb in block_comb_masks:
                    mask_s = mask_comb | (1 << x)
                    if mask_s in uncovered:
                        gain += 1
                
                if gain > best_gain or (gain == best_gain and freq[x] > best_freq):
                    best_gain = gain
                    best_point = x
                    best_freq = freq[x]
            
            if best_point == -1:
                break
                
            block.append(best_point)
