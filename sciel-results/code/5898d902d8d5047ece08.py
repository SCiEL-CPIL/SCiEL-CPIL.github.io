"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    
    def has_distinct_subset_sums(elements):
        """Check if all subset sums are distinct. Returns True if valid."""
        sums = {0}
        for e in elements:
            new_sums = set()
            for s in sums:
                ns = s + e
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums |= new_sums
        return True
    
    # Start with powers of 2
    best = [1 << i for i in range(n)]
    best_max = max(best)
    
    # Local search: try to reduce random elements
    current = best[:]
    
    random.seed(42)
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Pick a random element to try to reduce
        idx = random.randint(0, n - 1)
        old_val = current[idx]
        
        if old_val <= 1:
            continue
        
        # Try reducing this element to a smaller value
        # Try a few random smaller values
        tried = 0
        while tried < 20:
            new_val = random.randint(1, old_val - 1)
            if new_val in current:
                tried += 1
                continue
            
            current[idx] = new_val
            if has_distinct_subset_sums(current):
                cur_max = max(current)
                if cur_max < best_max:
                    best_max = cur_max
                    best = current[:]
                # Keep this improvement and continue
                break
            else:
                current[idx] = old_val
                tried += 1
    
    return best
# EVOLVE-BLOCK-END
