"""Rewritten: greedy superincreasing construction with DP verification."""
import time

def _check_property(elements):
    """Check that no two subsets of different cardinalities have the same sum."""
    n = len(elements)
    # For each cardinality c, compute set of achievable sums
    # sums[c] = set of all subset sums using exactly c elements
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    
    for i, x in enumerate(elements):
        # Update in reverse to avoid using x multiple times
        for c in range(i, -1, -1):
            for s in list(sums[c]):
                sums[c + 1].add(s + x)
    
    # Check pairwise disjointness across different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums[i] & sums[j]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Greedy superincreasing: pick smallest possible next element
    # that is > sum of all previously picked elements
    # Start with 1, then greedily pick smallest valid element
    elements = [1]
    current_sum = 1
    
    # Greedily build superincreasing sequence
    while True:
        next_val = current_sum + 1  # smallest that is > current_sum
        if next_val > 256:
            break
        elements.append(next_val)
        current_sum += next_val
        if time.time() > deadline:
            break
    
    # Verify with DP
    if _check_property(elements):
        return [int(x) for x in elements]
    
    # Fallback: powers of 2 up to 256
    fallback = []
    v = 1
    while v <= 256:
        fallback.append(v)
        v *= 2
    return [int(x) for x in fallback]
