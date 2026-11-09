import time
import sys

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    # Beam search parameters
    beam_width = 100
    max_depth = 25
    
    # Initial state: empty sequence
    # State: (sequence, set_of_sums, last_val)
    # To save memory/time, we can store sequence as tuple, sums as frozenset? 
    # No, sets are mutable and faster to update if we copy them. 
    # But copying sets of size ~100 is okay.
    
    # Optimization: Represent sums as a set of ints.
    
    current_beam = [((), set(), 0)]
    best_sequence = []
    best_len = 0
    
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            break
            
        next_beam_candidates = []
        
        for seq, sums, last_val in current_beam:
            # Generate next values
            # last_val is the last element of seq. 
            # If seq is empty, last_val is 0.
            start_val = last_val + 1 if last_val > 0 else 1
            
            for x in range(start_val, 257):
                # Check validity
                # New sums are sums of subarrays ending at x.
                # These are: x, x + seq[-1], x + seq[-1] + seq[-2], ...
                
                new_sums = []
                current_sum = x
                # Iterate backwards through seq to compute suffix sums ending at x
                # Note: seq is the prefix. The new element is x.
                # The subarrays ending at x are:
                # [x], [seq[-1], x], [seq[-2], seq[-1], x], ...
                
                # We need to check if any of these are in `sums`.
                # Also, the new sums themselves are distinct (strictly increasing as we add positive ints).
                
                valid = True
                temp_sum = x
                if temp_sum in sums:
                    valid = False
                else:
                    # Add previous elements
                    for a in reversed(seq):
                        temp_sum += a
                        if temp_sum in sums:
                            valid = False
                            break
                        # We don't need to store all new sums yet, just check validity
                        # But we need to add them to the new sums set if valid.
                
                if valid:
                    # Calculate the new sums set
                    # To avoid recomputing if we decide to keep this candidate, 
                    # we can compute the list of new sums.
                    
                    new_sums_list = [x]
                    temp_sum = x
                    for a in reversed(seq):
                        temp_sum += a
                        new_sums_list.append(temp_sum)
                    
                    # Check for duplicates within new_sums_list? 
                    # No, they are strictly increasing.
                    # Check for intersection with existing sums? 
                    # We already did that in the validity check.
                    
                    new_sums_set = sums.union(new_sums_list)
                    new_seq = seq + (x,)
                    
                    # Heuristic for ranking
                    # 1. Count available next steps
                    # This is expensive to do for every candidate.
                    # Let's use a cheaper heuristic first, or batch it.
                    
                    # Cheap heuristic: 
                    # Prefer smaller last value? 
                    # Or prefer values that create "sparse" sums?
                    
                    # Let's just collect all valid candidates and rank them later.
                    next_beam_candidates.append((new_seq, new_sums_set, x))
                    
        if not next_beam_candidates:
            break
            
        # Rank candidates
        # We want to keep the 'best' beam_width candidates.
        
        # Heuristic: 
        # We want to maximize future potential.
        # A sequence with a smaller last element has more potential next values.
        # A sequence with a smaller total sum might have less "crowding" in the sum space?
        
        # Let's try: Sort by last element (ascending).
        # This tends to keep the sequence "tight" at the lower end, which might allow more elements?
        # Actually, for distinct sums, spreading out might be better.
        
        # Let's try a look-ahead heuristic:
        # For a candidate (seq, sums, last_val), count how many y > last_val are valid.
        # This is O(256 * len(seq)) per candidate.
        # With 100 candidates in beam, and potentially 256*100=25600 candidates generated,
        # this might be too slow if done naively.
        
        # Optimization: 
        # We can approximate the "potential" by just looking at the last few elements or total sum.
        
        # Let's try sorting by:
        # 1. Number of used sums (smaller is better? no, length is fixed)
        # 2. Last value (smaller is better)
        # 3. Total sum (smaller is better)
        
        # Let's try sorting by last value.
        next_beam_candidates.sort(key=lambda item: item[2])
        
        # If we have many candidates, take the first beam_width
        current_beam = next_beam_candidates[:beam_width]
        
        # Update best
        if len(current_beam[0][0]) > best_len:
            best_len = len(current_beam[0][0])
            best_sequence = list(current_beam[0][0])
            
    return best_sequence
