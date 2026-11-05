import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Beam Search Parameters
    BEAM_WIDTH = 200
    MAX_CANDIDATES_TO_TRY = 20  # Try next 20 numbers
    
    # Each beam is a tuple: (sequence, products_set)
    # We also track the best length found
    best_seq = []
    best_len = 0
    
    # Initial beam
    beams = [([], set())]
    
    while time.time() < deadline:
        new_beams = []
        
        for seq, prods in beams:
            if not seq:
                last_val = 1
            else:
                last_val = seq[-1]
                
            # Try candidates greater than last_val
            # We limit the range to keep computation fast
            # But we must ensure we don't miss long sequences
            # Starting from last_val + 1 up to last_val + MAX_CANDIDATES_TO_TRY
            # Also, we should consider jumping further if we are stuck?
            # For now, greedy expansion of next few numbers.
            
            for c in range(last_val + 1, min(last_val + MAX_CANDIDATES_TO_TRY + 1, 129)):
                # Check collision
                tails = [c]
                prod = c
                for a in reversed(seq):
                    prod *= a
                    tails.append(prod)
                
                has_collision = False
                for p in tails:
                    if p in prods:
                        has_collision = True
                        break
                
                if not has_collision:
                    new_seq = seq + [c]
                    new_prods = prods.union(tails)
                    new_beams.append((new_seq, new_prods))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                        
        if not new_beams:
            break
            
        # Prune beams
        # Sort by length (desc), then by last value (asc), then by max product (asc)
        # Using max product as a tie-breaker is heuristic.
        
        def beam_sort_key(b):
            seq, prods = b
            # We want longer sequences
            # If same length, smaller last value is better (more room)
            # If same last value, smaller max product might be better?
            # Actually, smaller max product means the numbers are smaller, 
            # which might lead to more collisions later.
            # Larger numbers have larger products, less likely to collide.
            # So maybe we should prefer LARGER last values?
            # No, if we pick large values, we run out of candidates quickly.
            # We want to maximize the COUNT.
            # So we should prefer sequences that can be extended further.
            # A sequence with a smaller last value has more potential candidates.
            # So sort by:
            # 1. Length (Desc)
            # 2. Last Value (Asc)
            return (-len(seq), seq[-1] if seq else 0)
            
        new_beams.sort(key=beam_sort_key)
        
        # Keep top BEAM_WIDTH
        beams = new_beams[:BEAM_WIDTH]
        
        # Early termination if we found a very long sequence?
        # Max possible is 127.
        # If best_len is close to 127, we can stop?
        # But we don't know the max.
        
    return best_seq
