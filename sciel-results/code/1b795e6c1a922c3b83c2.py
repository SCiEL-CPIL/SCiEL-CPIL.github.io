import time
import random

def construct():
    # Beam Search Approach
    # State: (sequence, set_of_sums)
    # We store sums as a set for O(1) lookup.
    
    def get_sums(seq):
        sums = set()
        total = 0
        for i in range(len(seq)):
            total = 0
            for j in range(i, len(seq)):
                total += seq[j]
                sums.add(total)
        return sums

    def try_extend(seq, sums):
        # Returns list of (new_seq, new_sums) for all valid extensions
        # Actually, just return the new_seq and let caller compute sums?
        # Computing sums from scratch is O(N^2). 
        # Incremental is O(N).
        # Let's do incremental.
        pass

    # Optimization: Store sums in the beam state.
    # Beam item: (seq, sums_set)
    
    beam = [([], set())]
    best_seq = []
    best_len = 0
    B = 200  # Beam width
    
    time_limit = time.time() + 50
    
    while time.time() < time_limit:
        next_beam = []
        max_len_found = 0
        
        for seq, sums in beam:
            last_val = seq[-1] if seq else 0
            # Try all x > last_val
            for x in range(last_val + 1, 257):
                # Check if x can be added
                # New tails: x, x+seq[-1], x+seq[-1]+seq[-2]...
                # We need to check if any of these are in `sums`
                # Also, the new tails themselves are distinct (strictly increasing)
                
                valid = True
                new_tails = []
                total = 0
                # Iterate backwards
                for a in reversed([x] + seq):
                    total += a
                    if total in sums:
                        valid = False
                        break
                    new_tails.append(total)
                
                if valid:
                    new_seq = seq + [x]
                    new_sums = sums.union(new_tails)
                    next_beam.append((new_seq, new_sums))
                    if len(new_seq) > max_len_found:
                        max_len_found = len(new_seq)
        
        if not next_beam:
            break
            
        # Filter to only max length
        # If max_len_found is 0, it means no extensions were found? 
        # No, next_beam is not empty, so max_len_found >= 1.
        
        # If we found sequences longer than our best, update best
        if max_len_found > best_len:
            # We should store the best one immediately? 
            # Or wait until we finish pruning?
            # The beam will contain these.
            pass
            
        # Prune next_beam to width B
        # Heuristic: Keep sequences with smallest last value.
        # Among those, maybe randomize to avoid bias?
        
        if len(next_beam) > B:
            # Sort by last value
            next_beam.sort(key=lambda item: item[0][-1])
            # Take first B
            next_beam = next_beam[:B]
            
        # Update best
        if max_len_found > best_len:
            best_len = max_len_found
            # Pick one from the pruned beam or the full set?
            # Any one of the max length is a valid candidate.
            # Let's pick the one with smallest last value from the pruned set.
            best_seq = next_beam[0][0]
            
        beam = next_beam
        
    return best_seq
