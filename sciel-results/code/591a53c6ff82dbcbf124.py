import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s

    N = 128

    # Start with all elements
    S = set(range(1, N + 1))
    
    def make_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def compute_summask(mask):
        """Compute bitmask where bit s is set iff there exist two distinct elements in S with sum s."""
        sm = 0
        elems = [x for x in range(1, N+1) if (mask >> x) & 1]
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i+1, n):
                b = elems[j]
                s = a + b
                if s <= N:
                    sm |= (1 << s)
                else:
                    break
        return sm

    def find_violations(mask, summask):
        """Find all triples (a,b,c) in S such that a+b, a+c, b+c are in S."""
        violations = []
        elems = [x for x in range(1, N+1) if (mask >> x) & 1]
        n = len(elems)
        # For each pair a,b in S
        for i in range(n):
            a = elems[i]
            for j in range(i+1, n):
                b = elems[j]
                s_ab = a + b
                if s_ab > N:
                    continue
                if not ((mask >> s_ab) & 1):
                    continue
                # a+b is in S. Now find c in S, c!=a, c!=b, such that a+c in S and b+c in S.
                # Iterate c in S
                for k in range(n):
                    c = elems[k]
                    if c == a or c == b:
                        continue
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac > N or s_bc > N:
                        continue
                    if ((mask >> s_ac) & 1) and ((mask >> s_bc) & 1):
                        # Violation found: a,b,c
                        violations.append((a, b, c))
        return violations

    # Greedy removal
    mask = make_mask(S)
    summask = compute_summask(mask)
    
    while time.time() < deadline:
        violations = find_violations(mask, summask)
        if not violations:
            break
        
        # Count how many times each element appears in violations
        # Elements involved: a,b,c,a+b,a+c,b+c
        freq = {}
        for a, b, c in violations:
            elems_involved = [a, b, c, a+b, a+c, b+c]
            for e in elems_involved:
                if e in S:
                    freq[e] = freq.get(e, 0) + 1
        
        # Pick the element with highest frequency to remove
        # If tie, pick randomly
        max_freq = max(freq.values())
        candidates = [e for e, f in freq.items() if f == max_freq]
        rem = random.choice(candidates)
        
        S.remove(rem)
        mask &= ~(1 << rem)
        summask = compute_summask(mask)

    # Local Search: Try to add back removed elements
    # We can try to add any element not in S
    best_S = set(S)
    best_mask = mask
    best_summask = summask
    
    # Try to add elements greedily
    missing = [x for x in range(1, N+1) if x not in best_S]
    random.shuffle(missing)
    for x in missing:
        if time.time() > deadline:
            break
        # Check if adding x is valid
        # Re-use logic from previous can_add or just check validity of new set?
        # Checking validity of new set is O(N^3) or O(N^2) with summask?
        # Let's use a fast check:
        # New mask = best_mask | (1 << x)
        # New summask = best_summask | (sums involving x)
        # But we need to check if x creates a violation.
        # We can just call a helper `is_valid_addition`
        
        # For simplicity and speed, let's just try to add and check full validity if size increases?
        # No, that's too slow.
        # Let's implement `can_add` again, it's fast.
        
        def can_add(mask, summask, candidate):
            # Case 1: candidate is a pair-sum in S
            if (summask >> candidate) & 1:
                for a in range(1, candidate):
                    b = candidate - a
                    if b <= a:
                        break
                    if not ((mask >> b) & 1):
                        continue
                    hi_c = N - max(a, b)
                    if hi_c < 1:
                        continue
                    for c in range(1, hi_c + 1):
                        if c == a or c == b:
                            continue
                        if not ((mask >> c) & 1):
                            continue
                        ac = a + c
                        bc = b + c
                        if (mask >> ac) & 1 and (mask >> bc) & 1:
                            return False

            # Case 2: candidate is a new member. Check triples (candidate, x, y) with x,y in S
            elems = [x for x in range(1, N+1) if (mask >> x) & 1]
            n = len(elems)
            for i in range(n):
                x = elems[i]
                cx = candidate + x
                if cx > N:
                    continue
                cx_in = ((mask >> cx) & 1) or (cx == candidate)
                if not cx_in:
                    continue
                for j in range(i + 1, n):
                    y = elems[j]
                    cy = candidate + y
                    if cy > N:
                        continue
                    cy_in = ((mask >> cy) & 1) or (cy == candidate)
                    if not cy_in:
                        continue
                    xy = x + y
                    if xy > N:
                        continue
                    if (mask >> xy) & 1 or xy == candidate:
                        return False

            return True

        if can_add(best_mask, best_summask, x):
            best_S.add(x)
            best_mask |= (1 << x)
            best_summask = compute_summask(best_mask)

    result = sorted(best_S)
    return [int(x) for x in result]
