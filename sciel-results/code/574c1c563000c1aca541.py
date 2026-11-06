"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    max_val = 256
    
    # Beam: (sequence, set_of_sums, last_value)
    beams = [((), set(), 0)]
    
    best_seq = []
    best_len = 0
    
    # We iterate until we can't extend any beam or we hit a reasonable limit
    for depth in range(30):
        next_beams = []
        
        for seq, sums, last in beams:
            start = last + 1
            if start > max_val:
                continue
                
            for x in range(start, max_val + 1):
                # Compute new sums ending at x
                new_sums = set()
                total = x
                new_sums.add(total)
                
                for a in reversed(seq):
                    total += a
                    new_sums.add(total)
                    
                if new_sums & sums:
                    continue
                
                new_seq = seq + (x,)
                new_sums_set = sums | new_sums
                next_beams.append((new_seq, new_sums_set, x))
                
        if not next_beams:
            break
            
        # Update best
        max_len_in_beams = max(len(b[0]) for b in next_beams)
        if max_len_in_beams > best_len:
            best_len = max_len_in_beams
            best_candidate = max(next_beams, key=lambda b: len(b[0]))
            best_seq = list(best_candidate[0])

        # Prune beams
        # Heuristic: Estimate max possible future length
        # potential = current_len + log2(256 / last)
        # Using bit_length for speed: 256 is 2^8. bit_length(256)=9.
        # log2(256/last) approx 9 - last.bit_length()
        
        def heuristic(b):
            seq, sums, last = b
            if last == 0:
                return 0
            # Avoid issues if last > max_val (shouldn't happen)
            if last > max_val:
                return 0
            # 256 is 1 << 8.
            # We want to maximize the number of further doublings.
            # 9 - last.bit_length() is a good proxy.
            return len(seq) + (9 - last.bit_length())
            
        next_beams.sort(key=heuristic, reverse=True)
        
        # Keep top K
        beam_width = 150
        beams = next_beams[:beam_width]
        
    return best_seq

# EVOLVE-BLOCK-END
