"""
Constructs a set of 12 distinct positive integers with distinct subset sums,
minimizing the largest element using a greedy minimal extension algorithm.
"""
import time

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Greedy minimal extension:
    # Start with empty set, iteratively add the smallest unused positive integer
    # that doesn't create a duplicate subset sum.
    
    # subset_sums contains all current subset sums
    subset_sums = {0}
    selected = []
    used = set()
    
    for i in range(n):
        if time.time() > deadline:
            # Fallback to binary sequence if we run out of time
            return [1 << j for j in range(n)]
        
        # Try candidates starting from 1
        candidate = 1
        while True:
            if candidate in used:
                candidate += 1
                continue
            
            # Check if adding 'candidate' creates any duplicate subset sum
            # New sums would be {s + candidate for s in subset_sums}
            # Collision if any new sum is already in subset_sums
            has_collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    has_collision = True
                    break
            
            if not has_collision:
                # Accept this candidate
                selected.append(candidate)
                used.add(candidate)
                # Update subset sums
                new_sums = set(subset_sums)
                for s in subset_sums:
                    new_sums.add(s + candidate)
                subset_sums = new_sums
                break
            
            candidate += 1
    
    # Verify the result
    if len(selected) != n:
        return [1 << j for j in range(n)]
    
    # Final verification: check all 2^n subset sums are distinct
    all_sums = set()
    for mask in range(1 << n):
        s = 0
        for j in range(n):
            if mask & (1 << j):
                s += selected[j]
        if s in all_sums:
            # Should not happen, but fallback
            return [1 << j for j in range(n)]
        all_sums.add(s)
    
    if len(all_sums) != (1 << n):
        return [1 << j for j in range(n)]
    
    if max(selected) > 1048576:
        return [1 << j for j in range(n)]
    
    return selected
