"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    n = 12
    max_val = 1048576
    
    # Greedy: pick smallest positive integer not yet used such that
    # all subset sums remain distinct.
    # Track: used elements, and the set of all subset sums.
    
    elements = []
    subset_sums = {0}  # all subset sums of elements chosen so far
    
    for k in range(n):
        # Try candidates from 1 upward
        # But we can start from a smarter lower bound
        # The new element must not collide: for all s in subset_sums, c+s not in subset_sums
        # Also c must not be in elements (distinct)
        
        # Lower bound: at least 1, but practically we want small values
        # Upper bound: max_val
        # If c > max(subset_sums), then c+s > max(subset_sums) >= s for all s, 
        # but we still need c+s not in subset_sums. Since c+s > max(subset_sums) >= all elements of subset_sums,
        # they're automatically disjoint. So any c > max(subset_sums) works.
        
        max_sum = max(subset_sums)
        
        # Try candidates from 1 to max_sum first (these need checking),
        # then max_sum+1 is guaranteed to work.
        
        found = False
        # We want the smallest valid candidate
        # Start from 1, but skip used elements
        used_set = set(elements)
        
        # Optimization: the candidate must be such that c + s != s' for all s, s' in subset_sums
        # i.e., c != s' - s for all s, s' in subset_sums
        # The forbidden differences are {s' - s : s, s' in subset_sums, s' > s}
        # So c must not be in the set of positive differences
        
        # Compute forbidden values
        # This could be expensive if subset_sums is large, but it's at most 2^k elements
        # For k up to 11, that's 2048 elements, differences up to ~4M pairs — too many
        
        # Alternative: just check directly for each candidate
        # For a candidate c, check if (c + subset_sums) is disjoint from subset_sums
        # i.e., no s in subset_sums such that c + s is also in subset_sums
        # Equivalently: no s1, s2 in subset_sums with s2 - s1 == c
        
        # Let's just iterate candidates and check
        # To speed up, we can precompute the set of subset_sums for O(1) lookups
        
        # Try candidates from 1 upward
        # But we can start from a reasonable lower bound
        # The minimum possible is 1, but let's just start from 1
        
        # Actually, for efficiency, let's start from 1 and go up
        # But we can skip values that are clearly too small
        
        c = 1
        while c <= max_val:
            if c in used_set:
                c += 1
                continue
            
            # Check if c + s is in subset_sums for any s in subset_sums
            # If yes, collision
            # Equivalent: check if there exists s in subset_sums such that (c + s) in subset_sums
            # i.e., c is in {s2 - s1 : s1, s2 in subset_sums, s2 > s1}
            
            valid = True
            # Check: for each s in subset_sums, is (c + s) in subset_sums?
            # But this is O(|subset_sums|) per candidate
            # |subset_sums| = 2^k, so for k=11, that's 2048 checks per candidate
            
            # Alternative: iterate subset_sums and check
            # We can break early
            for s in subset_sums:
                if (c + s) in subset_sums:
                    valid = False
                    break
            
            if valid:
                # Accept c
                elements.append(c)
                used_set.add(c)
                # Update subset_sums: add {c + s for s in subset_sums}
                new_sums = set()
                for s in subset_sums:
                    new_sums.add(c + s)
                subset_sums |= new_sums
                found = True
                break
            
            c += 1
            
            # Time check
            if time.time() > deadline:
                break
        
        if not found:
            # Fallback: use a value > max_sum which is guaranteed valid
            c = max_sum + 1
            if c > max_val:
                # Shouldn't happen for n=12
                c = max_val
            elements.append(c)
            used_set.add(c)
            new_sums = set()
            for s in subset_sums:
                new_sums.add(c + s)
            subset_sums |= new_sums
        
        if time.time() > deadline:
            break
    
    # If we didn't get all 12 elements, fill with large values
    while len(elements) < n:
        # Add a value larger than max subset sum to guarantee no collision
        c = max(subset_sums) + 1
        if c > max_val:
            c = max_val
        # Make sure it's distinct
        while c in set(elements):
            c -= 1
            if c < 1:
                break
        elements.append(c)
        new_sums = set()
        for s in subset_sums:
            new_sums.add(c + s)
        subset_sums |= new_sums
    
    return elements
# EVOLVE-BLOCK-END
