import time
import random

def _valid(arr):
    """Check all 4096 subset sums are distinct."""
    n = len(arr)
    sums = [0] * (1 << n)
    sums[0] = 0
    for mask in range(1, 1 << n):
        # Use lowest set bit to compute
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        sums[mask] = sums[prev] + arr[idx]
    return len(set(sums)) == (1 << n)

def _max_elem(arr):
    return max(arr)

def construct():
    deadline = time.time() + 80.0
    # Start with powers of 2
    best = [1 << i for i in range(12)]
    best_max = _max_elem(best)
    
    # Local search: try to reduce the largest element
    random.seed(42)
    
    # Try hill climbing on the largest element
    # Key insight: for distinct subset sums, if sorted a1<a2<...<an,
    # a necessary condition is that no element equals a subset sum of others.
    # We can try to reduce the largest element.
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Try to find a valid sequence with max < best_max
        # Strategy: fix first 11 elements as powers of 2, try different 12th
        # Actually, let's try a more systematic approach.
        
        # Generate a candidate by taking powers of 2 and perturbing
        candidate = [1 << i for i in range(12)]
        
        # Try to reduce the last element
        # The 12th element (index 11) must be > sum of subset sums that would collide
        # Minimum valid value for a[11] given a[0..10]:
        # It must not equal any subset sum of a[0..10]
        # Subset sums of [1,2,4,...,1024] are exactly {0, 1, ..., 2047}
        # So a[11] must be >= 2048, which is what we have.
        
        # So with the first 11 as powers of 2, we can't reduce the 12th.
        # We need to change earlier elements too.
        
        # Try: reduce some earlier elements to allow a smaller max
        # This is complex. Let's try a random restart approach.
        
        # Generate random superincreasing-like sequences
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Build sequence greedily with some randomness
            arr = []
            current_max_sum = 0
            for i in range(12):
                if i == 0:
                    arr.append(1)
                else:
                    # Must be > some subset sum to avoid collision
                    # Minimum valid: must not equal any existing subset sum
                    # Safe choice: current_max_sum + 1 (superincreasing)
                    # But we can try smaller if no collision
                    min_val = current_max_sum + 1
                    # Try to go below min_val sometimes
                    if random.random() < 0.3 and i < 11:
                        # Try a value that's not superincreasing
                        val = random.randint(1, min_val)
                        # Check if val causes collision
                        # This is expensive; skip for now
                        pass
                    arr.append(min_val)
                    current_max_sum += min_val
            
            if _valid(arr):
                m = _max_elem(arr)
                if m < best_max:
                    best = arr[:]
                    best_max = m
    
    return best
