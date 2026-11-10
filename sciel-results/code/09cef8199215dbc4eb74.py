import random
import time

def construct():
    base = 51975
    # Generate divisors
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    def run_greedy(order):
        covered = bytearray(base)
        classes = []
        for m in order:
            best_r = 0
            max_uncovered = -1
            for r in range(m):
                count = covered[r::m].count(0)
                if count > max_uncovered:
                    max_uncovered = count
                    best_r = r
                # Early break if we found a perfect cover for this modulus?
                # If max_uncovered == len(range(r, base, m)), we can't do better.
                # Max possible is len(range(0, base, m)) which is base//m + (1 if base%m!=0 else 0)
                # Actually, for r=0, count is max possible.
                # If we hit the theoretical max, we can break.
                # Theoretical max for any r is roughly base/m.
                # If max_uncovered == base//m + (1 if base % m != 0 else 0) and best_r == 0? No.
                # Just check if max_uncovered is the max possible for this m.
                # Max possible uncovered for a specific r is the number of elements in the progression.
                # This number depends on r.
                # For r=0, it's (base-1)//m + 1.
                # For r>0, it might be smaller.
                # So we can't easily break.
                
            classes.append([m, best_r])
            # Update covered array
            # Calculate length of slice
            if best_r < base:
                length = (base - 1 - best_r) // m + 1
                covered[best_r::m] = b'\x01' * length
            # If best_r >= base, no update needed (but r < m <= base, so best_r < base always)
            
        uncovered = covered.count(0)
        return uncovered, classes

    # Initial run with sorted divisors (increasing)
    sorted_divs = sorted(divisors)
    best_uncovered, best_classes = run_greedy(sorted_divs)
    
    # Randomized restarts
    start_time = time.time()
    time_limit = 110
    
    # We can also try decreasing order
    rev_divs = sorted(divisors, reverse=True)
    unc, cls = run_greedy(rev_divs)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = cls

    while time.time() - start_time < time_limit:
        random_order = divisors[:]
        random.shuffle(random_order)
        unc, cls = run_greedy(random_order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = cls
            
    return best_classes
