import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Beam search
    # State: (sequence, sums_set, suffix_sums)
    # suffix_sums: list of sums of suffixes of sequence
    # e.g., seq=[1, 3], suffix_sums=[4, 3]
    # sums_set: set of all consecutive sums
    
    beam = [([], set(), [])]
    best_seq = []
    max_len = 0
    beam_width = 200
    
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
                # new_suffix_sums = [x + s for s in suffix_sums] + [x]
                # Check for collisions
                # The new sums are exactly new_suffix_sums
                # Check if any of them are in sums_set
                
                # Optimization: 
                # The smallest new sum is x (if x is not in sums_set)
                # Actually, x + suffix_sums[-1] is x + last_val.
                # x is the smallest new sum.
                
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
                # Note: new_suffix_sums_list is the list of new suffix sums.
                # Wait, the definition of suffix_sums for the new sequence:
                # seq' = seq + [x]
                # suffixes of seq':
                # 1. x
                # 2. x + last(seq)
                # 3. x + last2(seq) + last(seq)
                # ...
                # This matches new_suffix_sums_list generated above?
                # Let's trace:
                # suffix_sums of seq: [sum(seq), ..., last(seq)]
                # new_suffix_sums_list:
                #   x
                #   x + last(seq)
                #   x + second_last(seq) + last(seq)
                #   ...
                #   x + sum(seq)
                # Yes, this is the list of suffix sums of the new sequence.
                
                next_beam.append((new_seq, new_sums_set, new_suffix_sums_list))
        
        if not next_beam:
            break
        
        # Sort next_beam to keep the best ones
        # Heuristic: 
        # 1. Longer length is better (but all in this layer have same length? 
        #    No, beam contains states of potentially different lengths?
        #    In standard beam search, we expand all states to next length.
        #    So all states in next_beam have length len(beam[0][0]) + 1.
        #    Wait, if we start with [], we get length 1.
        #    Then length 2.
        #    So yes, all states in a specific iteration have the same length.
        #    So we don't need to sort by length.
        
        # Heuristic for pruning:
        # We want to leave room for future elements.
        # Smaller total sum (suffix_sums[0]) is better?
        # Smaller last value is better?
        # Let's use (total_sum, last_val) as the key.
        
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
