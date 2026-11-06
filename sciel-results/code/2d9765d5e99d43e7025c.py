def construct():
    import time
    start_time = time.time()
    
    def get_new_sums(seq, x):
        """
        seq: sorted list of existing numbers
        x: new number
        Returns: list of sums of all consecutive blocks in new_seq that contain x
        """
        n = len(seq)
        # Find insertion index
        k = 0
        while k < n and seq[k] < x:
            k += 1
        
        new_sums = []
        # Iterate over all blocks containing x
        # Block starts at i, ends at j
        # In new_seq, x is at index k.
        # i can be 0..k
        # j can be k..n
        for i in range(k + 1):
            # Sum of seq[i..k-1]
            left_sum = 0
            for m in range(i, k):
                left_sum += seq[m]
            # Now extend to the right
            right_sum = left_sum + x
            new_sums.append(right_sum)
            for j in range(k + 1, n + 1):
                right_sum += seq[j - 1]
                new_sums.append(right_sum)
        
        return new_sums

    def is_valid(seq, sums, x):
        """
        Check if adding x to seq (with existing sums) results in distinct sums.
        """
        if x in seq:
            return False
            
        new_sums = get_new_sums(seq, x)
        
        # Check for duplicates within new_sums
        if len(set(new_sums)) != len(new_sums):
            return False
            
        # Check if any new sum is in existing sums
        # Note: existing sums do NOT contain sums with x, so no overlap with old blocks
        # Wait, old sums are for blocks NOT containing x.
        # New sums are for blocks CONTAINING x.
        # So we just need to check intersection with `sums`.
        if sums.intersection(set(new_sums)):
            return False
            
        return True

    def update_sums(sums, seq, x):
        """
        Return new sums set after adding x.
        """
        new_sums = get_new_sums(seq, x)
        return sums.union(new_sums)

    # Beam Search
    # State: (seq, sums)
    beam = [([], set())]
    best_seq = []
    
    max_steps = 15
    beam_width = 500
    
    for step in range(max_steps):
        candidates = []
        for seq, sums in beam:
            # Try all x
            # Heuristic: try x that are "spaced out"?
            # Just try all 1..256
            for x in range(256, 0, -1):
                if is_valid(seq, sums, x):
                    new_seq = list(seq)
                    # Insert x
                    # We can just sort later, but we need sorted for get_new_sums
                    # get_new_sums assumes seq is sorted.
                    # So we must insert x into sorted position.
                    k = 0
                    while k < len(new_seq) and new_seq[k] < x:
                        k += 1
                    new_seq.insert(k, x)
                    
                    new_sums = update_sums(sums, seq, x)
                    candidates.append((new_seq, new_sums))
                    
        if not candidates:
            break
            
        # Prune
        # All have same length (step+1)
        # Sort by some heuristic to keep diverse/promising ones
        # Heuristic: Sum of elements? Or Min gap?
        # Let's keep the ones with the smallest max sum? No, we want to fit in [1, 256].
        # Actually, all x are in [1, 256].
        # Let's just keep the first `beam_width` candidates.
        # To make it deterministic, we can sort by the sequence itself?
        # Or by the number of "free" slots?
        
        # Let's sort by the sum of the sequence (smaller sum might allow more numbers?)
        # Or just keep the first 500.
        candidates.sort(key=lambda item: sum(item[0]))
        beam = candidates[:beam_width]
        
        # Update best
        best_seq = beam[0][0]
        
        if time.time() - start_time > 55:
            break
            
    return best_seq
