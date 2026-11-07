import time
import random

def construct():
    deadline = time.time() + 85
    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    def check_divisibility(sums_bitset):
        # Iterate through possible sums
        # Max sum is 256 * 30 = 7680
        # We can just iterate 1 to 7680
        # But we can skip unset bits.
        # Python doesn't have a built-in "next set bit" for ints efficiently in a loop without bit ops.
        # However, `while sums_bitset:` and `sums_bitset &= sums_bitset - 1` clears lowest bit.
        # But we need the VALUE of the bit.
        
        # Let's just iterate 1..7680. It's fast enough in C (inside Python loop? No, Python loop is slow).
        # 7680 iterations * 10000 nodes = 76M ops. Too slow.
        
        # Better: Iterate only over set bits.
        # How to get value of set bit?
        # `val = (sums_bitset & -sums_bitset).bit_length() - 1`
        # This is O(1) roughly.
        
        temp = sums_bitset
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            
            # Check multiples of s
            # m = 2s, 3s, ...
            # Check if bit m is set in sums_bitset
            m = s * 2
            while m < 8000:
                if (sums_bitset >> m) & 1:
                    return False
                m += s
                
            temp &= temp - 1 # Clear lowest set bit
            
        return True

    def backtrack(current_elements, sums_bitset, start_idx, order):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_set = list(current_elements)
            
        for i in range(start_idx, len(order)):
            x = order[i]
            
            # Collision check
            if (sums_bitset >> x) & 1:
                continue
                
            shifted = sums_bitset << x
            if sums_bitset & shifted:
                continue
                
            new_sums_bitset = sums_bitset | shifted | (1 << x)
            
            # Divisibility check
            if check_divisibility(new_sums_bitset):
                backtrack(current_elements + [x], new_sums_bitset, i + 1, order)

    # Try descending order first (large numbers)
    order1 = sorted(candidates, reverse=True)
    backtrack([], 0, 0, order1)
    
    if time.time() < deadline - 5:
        # Try ascending
        order2 = sorted(candidates)
        backtrack([], 0, 0, order2)
        
    if time.time() < deadline - 5:
        # Randomized
        rng = random.Random(42)
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            order3 = candidates.copy()
            rng.shuffle(order3)
            backtrack([], 0, 0, order3)

    if not best_set:
        return [2]
        
    # Final verification
    sums = {0}
    for x in best_set:
        new_sums = {s + x for s in sums}
        sums |= new_sums
    sums.discard(0)
    
    valid = True
    sums_list = list(sums)
    for s in sums_list:
        for k in range(2, max(sums_list)//s + 1):
            if s * k in sums:
                valid = False
                break
        if not valid:
            break
            
    if valid:
        return [int(x) for x in best_set]
    else:
        return [2]
