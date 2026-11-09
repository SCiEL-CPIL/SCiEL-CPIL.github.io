import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    S = []
    
    # Process from largest to smallest
    for x in range(n, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Check if x can be added to S
        # Since we iterate descending, all elements in S are > x.
        # Thus x will be 'a' in any new triple (a, b, c).
        # We need to ensure x does not divide (b+c) for any b, c in S with b <= c.
        
        valid = True
        # S is a list, but we can keep it as a set or list.
        # For O(N^2) check, list is fine.
        # We need to check all pairs (b, c) in S such that b <= c.
        # Note: b and c are distinct elements of the set?
        # The problem says "subset". So elements are unique.
        # But b <= c allows b == c?
        # "a < b <= c".
        # If b == c, then b+c = 2b.
        # Does the set contain b twice? No, it's a set.
        # But the triplet is defined by values a, b, c in the set.
        # "For every a < b <= c in the set".
        # This implies we pick three values from the set.
        # If b == c, it means we pick the value b for both positions.
        # So yes, we must check pairs (b, c) where b and c are elements of S, and b <= c.
        # This includes the case b == c.
        
        m = len(S)
        # Optimization: If m is large, this is O(m^2).
        # Max m is 256. 256^2 = 65536 ops.
        # 256 * 65536 = ~16 million ops.
        # Python might be slow for 16M ops within 60s?
        # 16M simple modulo ops should be well under 1 second in C, but in Python maybe 1-2 seconds.
        # Total time should be safe.
        
        for i in range(m):
            b = S[i]
            for j in range(i, m):
                c = S[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            
    return sorted(S)
