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
    beam_width = 500  # Increased beam width for better exploration
    
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
        
        if not next_beam:
            break
        
        # Sort next_beam to keep the best ones
        # Heuristic: prefer sequences with smaller total sum and smaller last value
        # to leave more room for future elements
        
        def heuristic(state):
            seq, sums, suffix = state
            total = suffix[0] if suffix else 0
            last = seq[-1] if seq else 0
            return (total, last)
        
        next_beam.sort(key=heuristic)
        
        if len(next_beam) > beam_width:
            next_beam = next_beam[:beam_width]
            
        # Update best
        cur_len = len(next_beam[0][0])
        if cur_len > max_len:
            max_len = cur_len
            best_seq = next_beam[0][0]
            
        beam = next_beam
        
    return best_seq
