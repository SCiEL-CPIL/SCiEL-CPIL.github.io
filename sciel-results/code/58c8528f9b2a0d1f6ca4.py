"""Find the longest interval in [1,1008] with no modular inverse pairs mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses: inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse elements: x^2 ≡ 1 (mod p) => x ≡ 1 or x ≡ -1 = 1008
    self_inv = {1, 1008}
    
    # For each start s, find the maximum valid end e using a forward sweep.
    # An interval [s, e] is valid iff for every x in [s, e]:
    #   - x is not self-inverse
    #   - inv[x] is not in [s, e]
    #
    # Strategy: for each s, sweep e forward. Maintain a set of "forbidden" elements
    # that would break validity. When we add element e, it forbids inv[e] (and vice versa).
    # We track the minimum left-bound that any element in [s, e] imposes.
    #
    # More efficient: for each s, the constraint is that for all x in [s, e],
    # inv[x] < s or inv[x] > e. So e must be < min(inv[x] for x in [s,e] if inv[x] > e)
    # This is circular. Instead, use the sliding window from the left.
    
    # Two-pointer sliding window (correct and O(N)):
    best_start = 1
    best_length = 0
    
    left = 1
    for right in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Self-inverse elements cannot be in any valid interval
        if right in self_inv:
            left = right + 1
            continue
        
        ix = inv[right]
        
        # If inv[right] is in the current window [left, right], advance left past it
        if left <= ix <= right:
            left = ix + 1
        
        # Ensure window is non-empty
        if left > right:
            continue
        
        length = right - left + 1
        if length > best_length:
            best_length = length
            best_start = left
    
    # Now do an exhaustive O(N^2) verification to ensure optimality.
    # For each start s, find the maximum valid end e.
    # This guards against any subtle bug in the sliding window.
    if time.time() < deadline - 5:
        for s in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            # Find max e such that [s, e] is valid
            # An interval [s, e] is valid iff:
            #   For all x in [s, e]: x not in self_inv AND inv[x] not in [s, e]
            # Equivalently: for all x in [s, e], inv[x] < s or inv[x] > e
            # So e must satisfy: for all x in [s, e], if inv[x] > s then inv[x] > e
            # i.e., e < min(inv[x] for x in [s, e] where inv[x] > s)
            # This is self-referential, so we sweep e forward.
            
            if s in self_inv:
                continue
            
            # Sweep e from s upward
            # Track: the minimum inv[x] for x in [s, current_e] that is > s
            # If that minimum is <= current_e, we've hit a conflict.
            # Actually simpler: just check incrementally.
            min_conflict = n + 1  # the smallest inv[x] > s for x in [s, e]
            for e in range(s, n + 1):
                if time.time() > deadline - 5:
                    break
                if e in self_inv:
                    break
                ix = inv[e]
                if ix > s:
                    # inv[e] is to the right of s, so it must be > e for validity
                    if ix < min_conflict:
                        min_conflict = ix
                    if min_conflict <= e:
                        # Conflict: some element in [s, e] has its inverse also in [s, e]
                        break
                # else inv[e] < s, no conflict within [s, e]
                length = e - s + 1
                if length > best_length:
                    best_length = length
                    best_start = s
    
    if best_length <= 0:
        for x in range(1, n + 1):
            if x not in self_inv:
                best_start = x
                best_length = 1
                break
    
    # Final verification
    s = best_start
    L = best_length
    elements = list(range(s, s + L))
    valid = True
    for i in range(len(elements)):
        x = elements[i]
        if (x * x) % p == 1:
            valid = False
            break
        for j in range(i + 1, len(elements)):
            y = elements[j]
            if (x * y) % p == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        best_start = 2
        best_length = 1
    
    return [best_start, best_length]
