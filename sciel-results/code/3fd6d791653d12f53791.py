import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    def get_new_state(seq, sums, tails, x):
        # Returns new state if x is valid, else None
        # New tails: [x, x + tails[0], x + tails[1], ...]
        new_tails = [x]
        for t in tails:
            new_tails.append(x + t)
            
        # Check collision
        for s in new_tails:
            if s in sums:
                return None
                
        new_sums = set(sums)
        new_sums.update(new_tails)
        new_seq = seq + [x]
        return (new_seq, new_sums, new_tails)

    # Beam Search
    beam_width = 50
    # Initial states: start with each x in 1..10
    beams = []
    for x in range(1, 11):
        seq = [x]
        sums = {x}
        tails = [x]
        beams.append((seq, sums, tails))
        
    best_seq = []
    best_len = 0
    
    while time.time() < deadline:
        next_beams = []
        
        for state in beams:
            seq, sums, tails = state
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
            last_val = seq[-1]
            
            valid_extensions = []
            
            for x in range(last_val + 1, 257):
                # Check validity: x + t not in sums for all t in [0] + tails
                # Note: t=0 corresponds to x itself.
                is_valid = True
                if x in sums:
                    is_valid = False
                else:
                    for t in tails:
                        if x + t in sums:
                            is_valid = False
                            break
                if is_valid:
                    valid_extensions.append(x)
                    
            if not valid_extensions:
                continue
                
            # Select top children
            # Keep small and large values to maintain diversity
            valid_extensions.sort()
            # Take 3 smallest and 2 largest
            selected = valid_extensions[:3] + valid_extensions[-2:]
            selected = list(set(selected))
            
            for x in selected:
                new_state = get_new_state(seq, sums, tails, x)
                if new_state:
                    next_beams.append(new_state)
                    
        # Prune next_beams to beam_width
        # Sort by length desc, then by last value asc
        next_beams.sort(key=lambda s: (-len(s[0]), s[0][-1]))
        
        beams = next_beams[:beam_width]
        
        if not beams:
            break
            
    return best_seq
