"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses for all residues 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse residues: x where x*x % p == 1, i.e., x = 1 or x = p-1 = 1008
    # These cannot be in any valid interval
    
    # For an interval [a, b] to be valid, for every x in [a, b]:
    #   1. x != 1 and x != 1008 (self-inverse check)
    #   2. inv(x) not in [a, b] (no pair x, inv(x) both in interval)
    
    # Key insight: if we fix the interval [a, b], the constraint is that
    # for all x in [a, b], inv[x] < a or inv[x] > b.
    
    # Strategy: For each possible start a, find the maximum valid end b.
    # An interval [a, b] is invalid if there exists x in [a, b] such that inv[x] in [a, b].
    # Equivalently, for each x in [a, b], inv[x] must be outside [a, b].
    
    # Precompute: for each position x, what's the "conflict" - i.e., inv[x].
    # If we extend the interval to include b, we need to check that for all x in [a, b],
    # inv[x] is not in [a, b].
    
    # More efficient approach: 
    # For a given start a, the interval can extend to b only if for all x in [a, b],
    # inv[x] < a or inv[x] > b.
    # As we increase b, the constraint "inv[x] > b" becomes harder to satisfy.
    # Specifically, if for some x in [a, b], inv[x] is in [a, b], the interval is invalid.
    
    # Let's think differently: 
    # Define for each x, its "partner" inv[x]. Note inv[inv[x]] = x.
    # If x == inv[x], then x is self-inverse (x=1 or x=1008).
    # Otherwise, x and inv[x] form a pair.
    # An interval [a, b] is valid iff:
    #   - It doesn't contain 1 or 1008
    #   - For no pair {x, inv[x]} are both x and inv[x] in [a, b]
    
    # So we need to find the longest interval [a, b] within [2, 1007] (excluding 1 and 1008)
    # that doesn't contain both elements of any inverse pair.
    
    # Build list of inverse pairs (excluding self-inverse)
    pairs = []
    seen = set()
    for x in range(2, 1008):  # 2..1007, excluding 1 and 1008
        if x in seen:
            continue
        y = inv[x]
        if y == x:
            # Self-inverse, skip (shouldn't happen for 2..1007 in p=1009)
            seen.add(x)
        else:
            pairs.append((x, y))
            seen.add(x)
            seen.add(y)
    
    # Now find the longest interval [a, b] with a >= 2, b <= 1007
    # such that no pair (x, y) has both x and y in [a, b].
    
    # For each pair (x, y) with x < y, the interval [a, b] is invalid if
    # a <= x and y <= b (assuming x < y).
    
    # We can use a two-pointer / sliding window approach:
    # For a fixed left endpoint a, find the maximum b such that no pair is fully contained.
    
    # Actually, let's use a different approach:
    # Sort all pairs by their smaller element.
    # For a given interval [a, b], it's invalid if there's a pair (x, y) with min(x,y) >= a and max(x,y) <= b.
    
    # Let's precompute: for each possible "forbidden" configuration, what intervals are blocked.
    # A pair (x, y) with x < y blocks any interval [a, b] where a <= x and b >= y.
    
    # So for a fixed a, the maximum valid b is the minimum over all pairs (x, y) with x >= a of (y - 1),
    # but only if y > a. Wait, let me reconsider.
    
    # For a fixed a, an interval [a, b] is valid iff for every pair (x, y) with x < y:
    #   NOT (a <= x AND y <= b)
    # This means: for every pair (x, y) with x >= a, we need b < y.
    # So b < min{y : (x, y) is a pair, x >= a, y > a} ... wait, also need to consider pairs where y >= a and x < a? No, if x < a, then x is not in [a, b], so the pair isn't fully contained.
    
    # Let me re-examine: pair (x, y) with x < y is fully contained in [a, b] iff a <= x AND y <= b.
    # So for fixed a, the constraint is: for all pairs (x, y) with x >= a (and x < y), we need b < y.
    # Also, we need b >= a.
    # So max valid b for start a is: min(y - 1 for all pairs (x, y) with x >= a) ... but only if such pairs exist.
    # Wait, we also need to handle the case where there are no such pairs, then b can go up to 1007.
    
    # But wait, we also need x >= a AND x < y. Since x < y always in our pairs, we just need x >= a.
    # But also, if y < a, then y is not in [a, b] either, so the pair isn't contained. But since x < y, if x >= a then y > x >= a, so y >= a+1 > a. So y is automatically >= a when x >= a.
    
    # So: for fixed a, max_b(a) = min(y - 1 for (x, y) in pairs if x >= a), or 1007 if no such pair.
    # But we also need b >= a, and the interval must be non-empty.
    # Also, we need a >= 2 and b <= 1007.
    
    # Let's precompute this efficiently.
    # Sort pairs by x. For each a from 2 to 1007, we want the minimum y among pairs with x >= a.
    
    # Create an array min_y_from_a where min_y_from_a[a] = min(y for (x,y) in pairs if x >= a)
    # We can compute this by processing a from high to low.
    
    max_y_val = 1007  # default if no constraint
    min_y_from_a = [max_y_val] * (n + 2)
    current_min = max_y_val
    # Process a from 1007 down to 2
    # For each a, add pairs with x == a to the consideration
    # Actually, let's build it differently.
    
    # For each x, find the minimum y among pairs where the smaller element is x.
    # Wait, each pair has a unique smaller element. Let me reorganize.
    
    # Actually, multiple pairs can have the same x? No, each x appears in exactly one pair (or is self-inverse).
    # So for each x in 2..1007, there's at most one pair where x is the smaller element.
    
    # Let me create: min_y_for_x[x] = y if x is the smaller element of a pair (x, y), else infinity.
    # Then min_y_from_a[a] = min(min_y_for_x[x] for x in range(a, 1008))
    
    INF = float('inf')
    min_y_for_x = [INF] * (n + 2)
    for x, y in pairs:
        smaller = min(x, y)
        larger = max(x, y)
        min_y_for_x[smaller] = min(min_y_for_x[smaller], larger)
    
    # Now compute min_y_from_a from right to left
    min_y_from_a = [INF] * (n + 2)
    running_min = INF
    for a in range(1007, 1, -1):
        running_min = min(running_min, min_y_for_x[a])
        min_y_from_a[a] = running_min
    
    # Now for each a in 2..1007:
    #   max_b = min(min_y_from_a[a] - 1, 1007) if min_y_from_a[a] < INF else 1007
    #   length = max_b - a + 1 if max_b >= a else 0
    
    best_start = 2
    best_len = 0
    
    for a in range(2, 1008):
        if min_y_from_a[a] < INF:
            max_b = min(min_y_from_a[a] - 1, 1007)
        else:
            max_b = 1007
        
        if max_b >= a:
            length = max_b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
    
    # Verify the result
    def is_valid(start, length):
        end = start + length - 1
        if start < 1 or end > 1008:
            return False
        for x in range(start, end + 1):
            if x * x % p == 1:
                return False
            if inv[x] >= start and inv[x] <= end:
                return False
        return True
    
    if not is_valid(best_start, best_len):
        # Fallback: try a simple valid interval
        best_start = 2
        best_len = 1
        for a in range(2, 1008):
            for b in range(1007, a - 1, -1):
                if is_valid(a, b - a + 1):
                    if b - a + 1 > best_len:
                        best_len = b - a + 1
                        best_start = a
                    break
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 2:
        # Shouldn't happen, but safety
        pass
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
