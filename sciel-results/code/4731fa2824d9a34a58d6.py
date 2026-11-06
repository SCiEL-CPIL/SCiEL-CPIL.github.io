"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Beam search: maintain multiple candidate sequences
    # Each beam is a tuple (sequence, set_of_sums, last_value)
    # We try to maximize the length of the sequence
    
    beam_width = 50
    max_val = 256
    
    # Initial beam: empty sequence
    beams = [((), frozenset(), 0)]
    
    for _ in range(20):  # Try to build up to length 20
        next_beams = []
        for seq, sums, last in beams:
            for x in range(last + 1, max_val + 1):
                # Compute all new consecutive block sums ending at x
                new_sums = set()
                total = x
                new_sums.add(total)
                for a in reversed(seq):
                    total += a
                    new_sums.add(total)
                
                if new_sums & sums:
                    continue
                
                next_beams.append((seq + (x,), sums | new_sums, x))
        
        if not next_beams:
            break
        
        # Keep only the best beam_width beams
        # Heuristic: prefer sequences with smaller last value (more room to grow)
        # and fewer total sums (less constrained)
        next_beams.sort(key=lambda b: (b[2], len(b[1])))
        beams = next_beams[:beam_width]
    
    # Find the longest sequence among all beams
    best = max(beams, key=lambda b: len(b[0]))
    return list(best[0])

# EVOLVE-BLOCK-END
