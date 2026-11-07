import time
import random

def is_valid(s):
    """
    Check if the subset sums of s are disjoint across different cardinalities.
    Uses bitset DP for efficiency.
    """
    n = len(s)
    if n == 0:
        return True
    
    # bits[k] is a bitmask where bit 'sum' is 1 if 'sum' is achievable with exactly k elements
    # Max sum is roughly 256 * n
    # Python integers handle arbitrary length, so we don't need to pre-allocate a fixed size array
    bits = [0] * (n + 1)
    bits[0] = 1  # Sum 0 is achievable with 0 elements (bit 0 set)
    
    for x in s:
        # Update bits in reverse order to avoid using the same element multiple times
        # bits[k+1] gets new sums from bits[k] + x
        for k in range(n - 1, -1, -1):
            if bits[k]:
                bits[k + 1] |= bits[k] << x
        
    # Check for disjointness
    # If any sum is achievable with two different cardinalities, it's invalid
    # We can check if bits[i] & bits[j] is non-zero for any i != j
    # Optimization: If we combine all bits, any bit set in the combined mask that came from multiple sources is bad.
    # But simpler: just check pairwise for small n.
    # Or: 
    # seen_bits = 0
    # for k in range(n+1):
    #    if bits[k] & seen_bits: return False
    #    seen_bits |= bits[k]
    # This works because if a sum is in bits[k] and bits[j] (k!=j), then when we process the second one, 
    # the bit will be in seen_bits (from the first one).
    
    seen_bits = 0
    for k in range(n + 1):
        if bits[k] & seen_bits:
            return False
        seen_bits |= bits[k]
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 1
    best_set = [1]
    
    # We try to increase the size of the set
    # We iterate potential sizes from 2 upwards
    # If we fail to find a valid set of size n, we assume n-1 is the max (or close to it)
    # However, it's possible we just got unlucky. 
    # So we will try to find a set of size n for a fixed number of attempts.
    
    current_n = 1
    while current_n < 60:
        target_n = current_n + 1
        found = False
        
        # Try to extend the current best_set
        # 1. Try adding a random element to best_set
        available = [x for x in range(1, 257) if x not in best_set]
        random.shuffle(available)
        
        # Try the first 100 available elements
        for x in available[:100]:
            new_set = best_set + [x]
            if is_valid(new_set):
                best_set = new_set
                best_n = target_n
                found = True
                break
        
        if found:
            current_n = target_n
            continue
            
        # If extension failed, try random restarts for target_n
        # We try a few hundred random sets of size target_n
        restarts = 500
        for _ in range(restarts):
            if time.time() > deadline:
                return best_set
            
            # Generate a random set of size target_n
            # To improve chances, we might bias towards larger numbers?
            # Or just uniform.
            candidates = random.sample(range(1, 257), target_n)
            if is_valid(candidates):
                best_set = candidates
                best_n = target_n
                found = True
                break
        
        if found:
            current_n = target_n
            continue
        else:
            # Failed to find a set of size target_n
            # It's possible that no such set exists, or it's very rare.
            # Given the hereditary property, if we can't find one, we might stop.
            # But let's try a few more with a "greedy random" approach?
            # Or just break and return best_set.
            
            # Let's try to find a valid set of size current_n (which we have) 
            # and see if we can extend *that* one? 
            # We already tried extending best_set.
            
            # Maybe try a different best_set of size current_n?
            # We can run a quick search for a valid set of size current_n 
            # and try to extend it.
            
            found_alt = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                alt_set = random.sample(range(1, 257), current_n)
                if is_valid(alt_set):
                    # Try to extend alt_set
                    avail = [x for x in range(1, 257) if x not in alt_set]
                    random.shuffle(avail)
                    for x in avail[:50]:
                        if is_valid(alt_set + [x]):
                            best_set = alt_set + [x]
                            best_n = current_n + 1
                            found_alt = True
                            break
                    if found_alt:
                        break
            
            if found_alt:
                current_n = best_n
                continue
            else:
                # If we can't even find an alternative extendable set, we stop
                break
                
    return best_set

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 60}

# The construct function is defined above.
# We need to make sure it's accessible as construct().
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program had:
# def construct():
#    return ...
# So we just expose construct.

# EVOLVE-BLOCK-END
