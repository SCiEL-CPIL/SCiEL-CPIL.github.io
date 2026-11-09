import time

def construct():
    """
    Returns a list of exactly n=12 distinct positive integers such that all subset sums are distinct.
    The goal is to minimize the largest integer.
    
    Uses a greedy minimal-extension algorithm: iteratively select the smallest unused
    positive integer that does not create a collision with any existing subset sum.
    """
    n = 12
    deadline = time.time() + 80  # safety margin before 90s limit
    
    result = []
    subset_sums = {0}
    
    for i in range(n):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            # Check if adding 'candidate' creates any collision
            # New subset sums would be old_sums and old_sums + candidate
            # Collision if any (s + candidate) is in subset_sums for s in subset_sums
            # i.e., if (s1 + candidate) == s2 for some s1, s2 in subset_sums
            # Equivalent to: candidate in (s2 - s1) for s1 != s2 in subset_sums
            # Simpler: check if any (s + candidate) in subset_sums
            collides = False
            for s in subset_sums:
                if s + candidate in subset_sums:
                    collides = True
                    break
            if not collides:
                break
            candidate += 1
        
        result.append(candidate)
        # Update subset sums
        new_sums = set()
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums.update(new_sums)
    
    # If we didn't complete in time, fall back to powers of 2
    if len(result) < n:
        return [1 << i for i in range(n)]
    
    return result
