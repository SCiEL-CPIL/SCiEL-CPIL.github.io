import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    RANGE_SIZE = N_HIGH - N_LOW + 1
    
    # Precompute indices for convolution extraction
    # conv = np.convolve(indicator, indicator, mode='full')
    # We need conv[128:257]
    
    def get_counts(indicator):
        """
        Compute unordered pair sum counts for [128, 256] given a 0/1 indicator vector.
        Returns an array of counts.
        """
        conv = np.convolve(indicator, indicator, mode='full')
        
        s_vals = np.arange(N_LOW, N_HIGH + 1, dtype=np.int32)
        conv_slice = conv[N_LOW:N_HIGH+1].copy()
        
        # For unordered pairs {a,b} with a<=b:
        # If a != b, it's counted twice in conv (a,b) and (b,a).
        # If a == b, it's counted once in conv (a,a).
        # Total unordered count C(s) = (conv[s] + I(s/2 in A if s even)) / 2
        
        even_mask = (s_vals % 2 == 0)
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Odd sums: a != b always. Count = conv[s] // 2
        counts[~even_mask] = conv_slice[~even_mask] // 2
        
        # Even sums: a can equal b.
        # Formula: (conv[s] + indicator[s//2]) // 2
        if np.any(even_mask):
            even_s = s_vals[even_mask]
            mids = even_s // 2
            # mids range from 64 to 128
            raw = conv_slice[even_mask].copy()
            # Add 1 where the midpoint is in A
            raw += indicator[mids]
            counts[even_mask] = raw // 2
            
        return counts

    def evaluate_set(A_set):
        """
        Evaluate a set A. Returns (peak, is_covered, counts_array).
        """
        if not A_set:
            return float('inf'), False, None
            
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A_set:
            indicator[a] = 1
            
        counts = get_counts(indicator)
        
        if np.any(counts == 0):
            return float('inf'), False, counts
            
        peak = int(np.max(counts))
        return peak, True, counts

    # Initial state: Full set [0, 128]
    current_A = set(range(MAX_VAL + 1))
    current_indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
    current_indicator[:] = 1
    
    current_peak, is_covered, current_counts = evaluate_set(current_A)
    
    best_peak = current_peak
    best_A = set(current_A)
    
    # Greedy Peeling Loop
    # We try to remove elements one by one.
    # To be efficient, we don't re-evaluate the whole set for every candidate removal
    # from scratch if possible, but given N=128, re-evaluation via convolution
    # is O(N^2) ~ 16k ops, which is very fast. 
    # 128 removals * 128 candidates = 16k evaluations * 16k ops = 256M ops.
    # This might be too slow in Python if not vectorized well.
    # Let's optimize: 
    # Instead of checking all 128 candidates at each step, we can be smarter.
    # But let's first try the direct approach with a time check.
    # Actually, 128 steps * 128 candidates = 16,384 evaluations.
    # Each eval is ~1us-5us? np.convolve of size 129 is fast.
    # 16k * 5us = 80ms. This is very fast.
    
    # However, we want to minimize the PEAK. 
    # Greedy strategy:
    # 1. Start with full set.
    # 2. At each step, for each x in A, calculate the new peak if x is removed.
    # 3. Choose the x that minimizes the new peak.
    # 4. If min new peak < current peak, remove x and update.
    # 5. If min new peak == current peak, remove x that allows most future flexibility? 
    #    Or just pick the one that minimizes the sum of squares or similar secondary metric?
    #    Or simply pick the one that keeps coverage and has the lowest "cost" in terms of 
    #    how many high-multiplicity sums it contributes to.
    #    Let's use: minimize (new_peak, sum(counts^2)) or just (new_peak, -len(contributing_to_peak))
    
    # To make it faster and more robust, let's do a "lookahead" of 1.
    # If multiple removals yield the same peak, pick the one that minimizes the 
    # number of representations of the peak sums? Or just random among ties?
    # Let's try: minimize (new_peak, max_count_in_new_counts) is just new_peak.
    # Secondary metric: minimize the sum of counts where count == new_peak.
    
    while time.time() < deadline - 5:
        if not is_covered:
            break
            
        best_new_peak = current_peak + 1 # If we can't improve, stop? 
        # Actually, we can keep removing if peak stays same, to make set smaller?
        # The task is to minimize peak. Smaller set isn't the goal directly, but 
        # often correlated. However, removing elements might increase peak.
        # So we only remove if peak does not increase.
        
        best_candidate = None
        best_secondary_metric = float('inf')
        
        candidates = list(current_A)
        
        # Shuffle to break ties randomly if needed, but we want deterministic-ish or best-first
        # Let's iterate through all
        
        for x in candidates:
            # Simulate removal of x
            new_indicator = current_indicator.copy()
            new_indicator[x] = 0
            
            # Check if coverage is maintained
            # Quick check: does x participate in any unique representation?
            # This is hard to check quickly without full conv.
            # Just compute counts.
            new_counts = get_counts(new_indicator)
            
            if np.any(new_counts == 0):
                continue
                
            new_peak = int(np.max(new_counts))
            
            if new_peak < current_peak:
                # This is a strict improvement
                best_candidate = x
                best_new_peak = new_peak
                # We could break here, but there might be multiple with same lower peak.
                # Let's find the one with best secondary metric.
                # Secondary: minimize sum of squares? Or just take first?
                # Let's take the one that minimizes the number of peak occurrences.
                num_peaks = np.sum(new_counts == new_peak)
                if best_candidate is None or num_peaks < best_secondary_metric:
                    best_secondary_metric = num_peaks
                    best_candidate = x
            elif new_peak == current_peak:
                # Neutral move: peak doesn't change.
                # We might want to do this to allow future improvements.
                # Or we might want to stop if we can't lower peak?
                # Usually, peeling stops when no removal lowers the peak.
                # But sometimes a neutral removal allows a subsequent lower peak.
                # Let's allow neutral removals only if we are stuck?
                # Or always allow if it reduces the "density"?
                # Let's try: allow neutral removals, but prioritize strict improvements.
                # If we find a strict improvement, we do it.
                # If not, we look for neutral ones.
                
                if best_new_peak > new_peak: # i.e. best_new_peak is still > current_peak
                    best_new_peak = new_peak
                    # Secondary metric for neutral: minimize sum of counts? 
                    # Or minimize the max count of the "second highest" sums?
                    # Let's use sum of counts as a proxy for total redundancy.
                    # Lower sum means less redundancy, might be better for future?
                    # Actually, high redundancy is good for coverage.
                    # Let's use: minimize the number of elements that are "critical" for coverage?
                    # Too complex. Let's just pick the one that minimizes the L2 norm of counts?
                    l2 = np.sum(new_counts**2)
                    if best_candidate is None or l2 < best_secondary_metric:
                        best_secondary_metric = l2
                        best_candidate = x
        
        # If we found a candidate that strictly improves peak
        if best_candidate is not None and best_new_peak < current_peak:
            current_A.remove(best_candidate)
            current_indicator[best_candidate] = 0
            current_peak = best_new_peak
            current_counts = get_counts(current_indicator)
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(current_A)
                
        else:
            # No strict improvement found.
            # Should we try neutral removals?
            # If we do neutral removals, we might get stuck in a local minimum of peak.
            # But we started from the global maximum redundancy.
            # Greedy peeling from full set usually finds a good local optimum for peak.
            # Let's try to find a neutral removal to see if it helps later.
            # However, if we just keep removing neutrals, we might break coverage eventually.
            # Let's limit neutral removals or stop here.
            
            # Let's try one round of neutral removals if we haven't improved in a while?
            # Actually, let's just stop if no strict improvement.
            # Why? Because if no single removal lowers the peak, it's likely a local minimum.
            # To escape, we'd need swaps.
            
            # Let's implement a simple swap phase if stuck.
            # Swap: remove x, add y.
            # This is more expensive.
            
            # Let's first try to see if we can do a neutral removal that keeps the set "balanced".
            # If we can't find a strict improvement, let's try to find a neutral one
            # that minimizes the peak count of the *second* highest sum?
            
            # For now, let's just break and move to swaps.
            break
            
    # If we stopped early due to no strict improvement, try swaps
    if time.time() < deadline - 5:
        # Simple local search with swaps
        # Current A is a local minimum for single removals.
        # Try: remove x, add y (y not in A).
        # To be fast, only try swaps that seem promising.
        # e.g., x is in many high-multiplicity sums, y is in few.
        
        current_peak, is_covered, current_counts = evaluate_set(current_A)
        
        if is_covered:
            # Identify "bad" elements: those contributing to peak sums.
            peak_sums = np.where(current_counts == current_peak)[0]
            peak_sum_values = N_LOW + peak_sums
            
            # For each peak sum, find contributing pairs.
            # This is complex. Let's just do random swaps with a time limit.
            
            stuck_counter = 0
            T = 1.0 # Temperature for simulated annealing on swaps?
            # Let's do a simple hill climbing with random restarts of the swap search.
            
            for _ in range(5000): # Max swap iterations
                if time.time() > deadline - 5:
                    break
                
                if not is_covered:
                    # Repair? Hard. Just break.
                    break
                    
                # Pick random x in A
                x = np.random.choice(list(current_A))
                
                # Pick random y not in A
                if len(current_A) == MAX_VAL + 1:
                    continue # Full set, can't add
                available = np.setdiff1d(np.arange(MAX_VAL + 1), list(current_A))
                if len(available) == 0:
                    continue
                y = np.random.choice(available)
                
                if x == y:
                    continue
                    
                # Try swap
                new_A = set(current_A)
                new_A.remove(x)
                new_A.add(y)
                
                new_indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
                for a in new_A:
                    new_indicator[a] = 1
                    
                new_counts = get_counts(new_indicator)
                
                if np.any(new_counts == 0):
                    continue
                    
                new_peak = int(np.max(new_counts))
                
                if new_peak < current_peak:
                    current_A = new_A
                    current_indicator = new_indicator
                    current_counts = new_counts
                    current_peak = new_peak
                    is_covered = True
                    stuck_counter = 0
                    
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(current_A)
                        
                elif new_peak == current_peak:
                    # Accept neutral moves with some probability to explore
                    if np.random.random() < 0.1:
                        current_A = new_A
                        current_indicator = new_indicator
                        current_counts = new_counts
                        # current_peak unchanged
                        stuck_counter += 1
                else:
                    # Reject worse
                    stuck_counter += 1
                    
                if stuck_counter > 100:
                    # Restart local search from best_A?
                    # Or just break and try a different strategy?
                    # Let's break this loop and try a different random seed/state?
                    # For now, just break to save time.
                    break

    # Final polish: try to remove more elements if possible (neutral or improving)
    # We might have stopped too early in the greedy phase.
    # Let's do one more pass of greedy peeling.
    if time.time() < deadline - 2:
        current_A = set(best_A)
        current_peak, is_covered, current_counts = evaluate_set(current_A)
        current_indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in current_A:
            current_indicator[a] = 1
            
        # Repeat peeling until no improvement
        while time.time() < deadline - 2:
            best_candidate = None
            best_new_peak = current_peak + 1
            best_secondary_metric = float('inf')
            
            for x in list(current_A):
                new_indicator = current_indicator.copy()
                new_indicator[x] = 0
                new_counts = get_counts(new_indicator)
                
                if np.any(new_counts == 0):
                    continue
                    
                new_peak = int(np.max(new_counts))
                
                if new_peak < current_peak:
                    num_peaks = np.sum(new_counts == new_peak)
                    if best_candidate is None or num_peaks < best_secondary_metric:
                        best_candidate = x
                        best_new_peak = new_peak
                        best_secondary_metric = num_peaks
                elif new_peak == current_peak:
                    # Only consider neutral if we haven't found a strict improvement
                    if best_new_peak > new_peak:
                        best_new_peak = new_peak
                        l2 = np.sum(new_counts**2)
                        if best_candidate is None or l2 < best_secondary_metric:
                            best_candidate = x
                            best_secondary_metric = l2
            
            if best_candidate is not None and best_new_peak < current_peak:
                current_A.remove(best_candidate)
                current_indicator[best_candidate] = 0
                current_peak = best_new_peak
                current_counts = get_counts(current_indicator)
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(current_A)
            else:
                break

    return list(best_A)
