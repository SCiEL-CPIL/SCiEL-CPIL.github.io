import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit, leave 5s margin
    
    best_n = 0
    best_set = []
    
    def check_validity(bits, n):
        """Check if the bitsets represent disjoint sums for different cardinalities."""
        seen = 0
        for k in range(n + 1):
            if bits[k] & seen:
                return False
            seen |= bits[k]
        return True

    def greedy_build(order, current_bits, current_set, current_n, mask_all):
        """Greedily add elements from order to current_set."""
        bits = current_bits[:]
        used = set(current_set)
        n = current_n
        m = mask_all
        
        for x in order:
            if time.time() > deadline:
                return bits, list(current_set), n, m
            if x in used:
                continue
            
            valid = True
            # Optimized check:
            # new_sums for cardinality k+1 is bits[k] << x
            # It must not intersect with any bits[j] where j != k+1
            # Since all bits[j] are disjoint, the union of all bits[j] for j != k+1 is (mask_all ^ bits[k+1])
            # So we check (bits[k] << x) & (mask_all ^ bits[k+1])
            for k in range(n, -1, -1):
                if bits[k]:
                    new_sums = bits[k] << x
                    forbidden = m ^ bits[k+1]
                    if new_sums & forbidden:
                        valid = False
                        break
            
            if valid:
                # Update bits
                for k in range(n, -1, -1):
                    if bits[k]:
                        bits[k + 1] |= bits[k] << x
                current_set.append(x)
                used.add(x)
                n += 1
                # Update mask_all
                # m is OR of all bits.
                # New bits are added to k+1.
                # We can just recompute m or update it.
                # m |= OR of new parts?
                # Actually, m = OR(bits).
                # We can update m by ORing the new contributions?
                # No, simpler to recompute if n is small, but n can be 60.
                # Let's just recompute m.
                m = 0
                for k in range(n + 1):
                    m |= bits[k]
                
                if n > best_n:
                    # We can't update best_n here easily without nonlocal, so return it
                    pass
        
        return bits, list(current_set), n, m

    # We need to track best_n and best_set in the outer scope
    # Let's use a mutable list for best_n to modify inside helper if needed, or just return
    # Since greedy_build returns the state, we can update best_n outside.
    
    # Strategy: Try multiple orders
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        list(range(2, 257, 2)) + list(range(1, 257, 2)),
        list(range(1, 257, 2)) + list(range(2, 257, 2)),
    ]
    
    # Add some random shuffles
    for _ in range(20):
        o = list(range(1, 257))
        random.shuffle(o)
        orders.append(o)
    
    for fam in orders:
        if time.time() > deadline:
            break
        
        current_set = []
        bits = [0] * 61
        bits[0] = 1
        mask_all = 1
        n = 0
        
        bits, current_set, n, mask_all = greedy_build(fam, bits, current_set, n, mask_all)
        
        if n > best_n:
            best_n = n
            best_set = current_set[:]
            
        if best_n >= 60:
            break

    # Random restarts with greedy completion
    if best_n < 60 and time.time() < deadline - 2:
        # Try to extend a random partial set greedily
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Pick a random size to start with, or just random elements
            # Let's pick a random subset of size best_n + 1 and check validity?
            # Or pick random elements and greedily add?
            
            # Better: Start with empty, but random order
            # We already did random shuffles.
            
            # Try "local search" or "restart" with bias
            # Pick a random starting set of size k
            start_size = min(best_n, random.randint(0, 10))
            if start_size > 0:
                initial_set = random.sample(range(1, 257), start_size)
                bits = [0] * 61
                bits[0] = 1
                mask_all = 1
                for x in initial_set:
                    for k in range(start_size - 1, -1, -1):
                        if bits[k]:
                            bits[k + 1] |= bits[k] << x
                # Check if initial set is valid
                if not check_validity(bits, start_size):
                    continue
                mask_all = 0
                for k in range(start_size + 1):
                    mask_all |= bits[k]
                
                # Greedily add more
                remaining = [x for x in range(1, 257) if x not in initial_set]
                random.shuffle(remaining)
                
                bits, current_set, n, mask_all = greedy_build(remaining, bits, initial_set, start_size, mask_all)
                
                if n > best_n:
                    best_n = n
                    best_set = current_set[:]
                    
                if best_n >= 60:
                    break

    # Ensure we return a valid list of ints
    result = [int(x) for x in best_set]
    return result

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
