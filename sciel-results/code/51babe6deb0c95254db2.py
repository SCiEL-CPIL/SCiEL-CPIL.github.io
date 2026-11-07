<<<<<<< SEARCH
    # Final greedy polish with vectorized candidate ranking
    if best_sel is not None and time.time() < deadline - 1.0:
        selected = list(best_sel)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = best_score
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 0.5:
                    break
                old = selected[idx]
                sel_wo = sel_mask & ~(1 << old)
                loss = (comp[old] & sel_wo).bit_count()
                # Vectorized: compute gain for all candidates not in sel_mask
                all_idx = np.arange(total, dtype=np.int64)
                # comp rows as bitmasks; we need bit_count of (comp[new] & sel_wo)
                # Use numpy: for each new, count bits in comp[new] & sel_wo
                # Precompute comp as numpy array of uint64? total=4096, fits in uint64
                # But we already have comp as Python ints. Let's use a fast loop with early exit.
                best_gain = -1
                best_new = None
                # Try candidates in order of popcount (likely to be comparable)
                for new in range(total):
                    if (sel_mask >> new) & 1:
                        continue
                    gain = (comp[new] & sel_wo).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new
                if best_new is not None and best_gain > loss:
                    selected[idx] = best_new
                    sel_mask = (sel_mask & ~(1 << old)) | (1 << best_new)
                    current += best_gain - loss
                    improved = True
                    break
        
        if current > best_score:
            best_score = current
            best_sel = selected
=======
    # Final greedy polish with vectorized candidate ranking
    if best_sel is not None and time.time() < deadline - 1.0:
        selected = list(best_sel)
        sel_mask = 0
        for x in selected:
            sel_mask |= (1 << x)
        current = best_score
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for idx in range(m):
                if time.time() >= deadline - 0.5:
                    break
                old = selected[idx]
                sel_wo = sel_mask & ~(1 << old)
                loss = (comp[old] & sel_wo).bit_count()
                # Vectorized: compute gain for all candidates not in sel_mask
                best_gain = -1
                best_new = None
                for new in range(total):
                    if (sel_mask >> new) & 1:
                        continue
                    gain = (comp[new] & sel_wo).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new
                if best_new is not None and best_gain > loss:
                    selected[idx] = best_new
                    sel_mask = (sel_mask & ~(1 << old)) | (1 << best_new)
                    current += best_gain - loss
                    improved = True
                    break
        
        if current > best_score:
            best_score = current
            best_sel = selected
>>>>>>> REPLACE

