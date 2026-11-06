"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _score_bitmask(mask, n=512):
    """Count distinct d>0 such that exists a with a, a+d, a+2d in mask."""
    # For each d, check if any a has all three bits set.
    # mask is int with bit i set iff (i+1) in set (we use 1-indexed positions 1..512)
    # Actually let's use bit position = value (1..512). So bit v is set if v in set.
    count = 0
    for d in range(1, n // 2 + 1):
        # a, a+d, a+2d all in [1,512] => a in [1, 512-2d]
        # We need: for some a, bits a, a+d, a+2d all set.
        # Equivalently: (mask & (mask >> d) & (mask >> (2*d))) has any bit set in positions 1..(512-2d)
        # But shifting right by d moves bit v to position v-d. So bit a in (mask >> d) means bit a+d in mask.
        # So (mask & (mask >> d) & (mask >> (2*d))) bit a set means a, a+d, a+2d all in mask.
        # We only care about a in 1..(512-2d), but if a > 512-2d, then a+2d > 512, so those bits wouldn't be set anyway in a valid mask.
        # However, to be safe, we can mask out positions > 512-2d, but since our mask only has bits 1..512 set, and a+2d > 512 means that bit is 0, the AND will be 0 for those a.
        combined = mask & (mask >> d) & (mask >> (2 * d))
        if combined:
            count += 1
    return count


def _mask_to_list(mask):
    """Convert bitmask to sorted list of values."""
    return [i + 1 for i in range(512) if (mask >> i) & 1]


def _list_to_mask(lst):
    """Convert list of values to bitmask."""
    m = 0
    for v in lst:
        m |= (1 << v)
    return m


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    rng = random.Random(42)

    # Initial seed: try a few different starting points
    # Start with a structured set: multiples of some step, or a random set
    # Let's start with a set that has some AP structure
    # Option 1: 1..32 (the baseline)
    # Option 2: A set based on a known good construction
    
    # Let's try a greedy construction first
    # Start with a small set and greedily add elements that maximize score
    
    best_score = -1
    best_mask = None
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            # Start with baseline
            mask = _list_to_mask(list(range(1, 33)))
        else:
            # Random 32-element subset
            vals = rng.sample(range(1, 513), 32)
            mask = _list_to_mask(vals)
        
        score = _score_bitmask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
        
        # Local search: hill climbing
        for iteration in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            # Generate candidate moves
            # Move type 1: swap one element in with one out
            current_vals = _mask_to_list(mask)
            in_set = set(current_vals)
            
            # Find elements not in set
            not_in = [v for v in range(1, 513) if v not in in_set]
            
            # Heuristic: for each missing element x, count how many 3-term APs it would complete
            # For x to complete an AP, it must be:
            # - the middle: a, x, a+2d where x-a = a+2d-x => a = 2x - (a+2d)... 
            #   Actually: a, x, b with x-a = b-x => b = 2x-a. So for each a in set with a < x, check if 2x-a in set.
            # - the first: x, x+d, x+2d where x+d and x+2d in set
            # - the last: x-2d, x-d, x where x-2d and x-d in set
            
            # This is expensive to compute for all x. Let's just try random swaps and accept if score improves.
            
            # Pick a random element to remove and a random element to add
            out_idx = rng.randrange(32)
            out_val = current_vals[out_idx]
            in_val = rng.choice(not_in)
            
            if in_val == out_val:
                continue
            
            new_mask = mask & ~(1 << out_val)
            new_mask |= (1 << in_val)
            
            new_score = _score_bitmask(new_mask)
            if new_score > score:
                mask = new_mask
                score = new_score
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            # Also try: remove one, add two (keep size 32: remove 1, add 1 is swap; or remove 2, add 2)
            # Let's also try double swaps sometimes
            if iteration % 10 == 0:
                out_idx2 = rng.randrange(32)
                if out_idx2 == out_idx:
                    out_idx2 = (out_idx2 + 1) % 32
                out_val2 = current_vals[out_idx2]
                in_val2 = rng.choice(not_in)
                if in_val2 == out_val or in_val2 == out_val2:
                    continue
                new_mask2 = mask & ~(1 << out_val) & ~(1 << out_val2)
                new_mask2 |= (1 << in_val) | (1 << in_val2)
                new_score2 = _score_bitmask(new_mask2)
                if new_score2 > score:
                    mask = new_mask2
                    score = new_score2
                    if score > best_score:
                        best_score = score
                        best_mask = mask
    
    return _mask_to_list(best_mask)
# EVOLVE-BLOCK-END
