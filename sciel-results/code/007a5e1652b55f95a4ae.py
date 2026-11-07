import random
import time

def is_valid(A):
    n = len(A)
    if n == 0:
        return True
    if n > 22: # Limit to keep it fast
        return False
        
    sums = set()
    # Generate subset sums
    # Use iterative addition
    current_sums = {0}
    for a in A:
        new_sums = set()
        for s in current_sums:
            ns = s + a
            if ns in sums:
                return False
            new_sums.add(ns)
        # Check if any new sum is divisible by existing or vice versa
        # Actually, we need to check all pairs.
        # Let's collect all sums first.
        sums.update(new_sums)
        current_sums = new_sums | current_sums # Wait, this is wrong.
        
    # Correct subset sum generation
    sums = set()
    sums.add(0)
    for a in A:
        new_sums = set()
        for s in sums:
            ns = s + a
            new_sums.add(ns)
        # Check duplicates
        if len(sums | new_sums) != len(sums) + len(new_sums):
             # Intersection is not just {0}? 
             # If s1+a == s2, and s1 != s2, then duplicate?
             # Actually, if sums are unique before, and a is not 0, 
             # then s1+a == s2 implies a = s2-s1.
             # This doesn't necessarily mean duplicate sum in the set of all sums.
             # It means two subsets have the same sum.
             # So if new_sums intersects sums, and the intersection is not empty,
             # we have a collision.
             # Wait, s=0 is in sums. 0+a = a. If a is in sums, collision.
             # But a is not in sums (distinct elements, sums are non-empty? No, 0 is empty).
             # The task says NONEMPTY subset sums.
             # So we should not include 0 in the final check, but it helps generation.
             
             # Let's just generate all non-empty sums into a set.
             pass

    # Regenerate properly
    all_sums = set()
    n = len(A)
    for i in range(1, 1 << n):
        s = 0
        for j in range(n):
            if i & (1 << j):
                s += A[j]
        if s in all_sums:
            return False
        all_sums.add(s)
        
    # Check divisibility
    max_sum = max(all_sums)
    present = [False] * (max_sum + 1)
    for s in all_sums:
        present[s] = True
        
    for s in all_sums:
        m = 2 * s
        while m <= max_sum:
            if present[m]:
                return False
            m += s
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = [31, 32, 34]
    best_size = 3
    
    # Random restarts
    for _ in range(200):
        if time.time() > deadline:
            break
            
        # Start with a random small set
        # Try to grow it
        A = []
        # Pick a random starting element
        A.append(random.randint(1, 256))
        
        # Try to add more elements
        while True:
            added = False
            # Shuffle candidates to avoid bias
            candidates = [x for x in range(1, 257) if x not in A]
            random.shuffle(candidates)
            
            for x in candidates:
                new_A = A + [x]
                if is_valid(new_A):
                    A = new_A
                    added = True
                    break
            
            if not added:
                break
                
        if len(A) > best_size:
            best_size = len(A)
            best_set = A[:]
            
    return best_set
