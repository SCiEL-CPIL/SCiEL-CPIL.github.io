import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Beam search with DFS-like exploration
    # State: (sequence, sums_set, suffix_sums)
    # suffix_sums: list of sums of suffixes of sequence
    # e.g., seq=[1, 3], suffix_sums=[4, 3]
    # sums_set: set of all consecutive sums
    
    beam = [([], set(), [])]
    best_seq = []
    max_len = 0
    beam_width = 2000  # Increased beam width significantly
    
    while True:
        if time.time() > deadline:
            break
            
        next_beam = []
        
        for seq, sums_set, suffix_sums in beam:
            last_val = seq[-1] if seq else 0
            n = len(seq)
            
            # Try all x > last_val
            # We can limit x to 256
            for x in range(last_val + 1, 257):
                # Calculate new suffix sums
                # Check for collisions
                
                valid = True
                new_suffix_sums_list = []
                
                # Check x
                if x in sums_set:
                    valid = False
                else:
                    new_suffix_sums_list.append(x)
                    # Check x + s for s in suffix_sums
                    for s in suffix_sums:
                        ns = x + s
                        if ns in sums_set:
                            valid = False
                            break
                        new_suffix_sums_list.append(ns)
                
                if not valid:
                    continue
                
                # If valid, create new state
                new_seq = seq + [x]
                new_sums_set = sums_set.union(new_suffix_sums_list)
                
                next_beam.append((new_seq, new_sums_set, new_suffix_sums_list))
                
                # Optional: Early exit if we hit a very long sequence? No, keep exploring.
        
        if not next_beam:
            break
        
        # Sort next_beam to keep the best ones
        # Heuristic: prefer sequences with smaller last value (more room)
        # and smaller total sum.
        # Let's try sorting by (last_val, total_sum) instead of (total_sum, last_val)
        # Smaller last_val is more critical for extending the sequence.
        
        def heuristic(state):
            seq, sums, suffix = state
            total = suffix[0] if suffix else 0
            last = seq[-1] if seq else 0
            return (last, total)
        
        next_beam.sort(key=heuristic)
        
        if len(next_beam) > beam_width:
            next_beam = next_beam[:beam_width]
            
        # Update best
        # The best sequence in the beam (first after sorting) might not be the longest
        # but usually the longest ones are found deeper. 
        # We should check the length of the first element in the sorted beam,
        # but also track the global max.
        
        # Actually, since we sort by last_val, the first one has the smallest last_val.
        # It might not be the longest.
        # We should find the max length in the beam.
        
        max_len_in_beam = 0
        best_seq_in_beam = []
        for state in next_beam:
            l = len(state[0])
            if l > max_len_in_beam:
                max_len_in_beam = l
                best_seq_in_beam = state[0]
                
        if max_len_in_beam > max_len:
            max_len = max_len_in_beam
            best_seq = best_seq_in_beam
            
        beam = next_beam
        
    return best_seq
