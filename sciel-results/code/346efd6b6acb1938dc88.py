"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 256
    
    # Build conflict graph:
    # Edge between b and c (b <= c) if there exists a < b such that a divides (b+c)
    # We want a subset S such that for all a < b <= c in S, a does NOT divide b+c.
    # Equivalently, for any pair (b,c) with b <= c in S, we must ensure no a in S with a < b divides b+c.
    
    # Let's think about this differently.
    # For a set S, the constraint is: for all a < b <= c in S, a ∤ (b+c).
    # 
    # A conflict exists between elements b and c (b <= c) if there exists some a < b 
    # such that a divides b+c. But a must also be in S.
    # 
    # Actually, the constraint is global: for every triple a < b <= c all in S, a must not divide b+c.
    # 
    # Let's model this as: we select a subset S. For each pair (b, c) with b <= c, 
    # if b+c is divisible by some a < b, then we cannot have all three a, b, c in S.
    # This is a 3-uniform hypergraph independent set problem, which is hard.
    
    # Alternative approach: Greedy construction.
    # Start with all elements, then remove elements that cause conflicts.
    # Or build up greedily.
    
    # Let's try a different angle: which sets are "safe"?
    # If we take only large numbers, say [129, 256], then for a < b <= c in this range:
    # a >= 129, b+c <= 256+256 = 512.
    # a divides b+c means b+c = k*a for some integer k >= 1.
    # Since b+c <= 512 and a >= 129, k can be 1, 2, or 3.
    # k=1: b+c = a, but a < b <= c, so a < b <= c implies a < b+c, contradiction since b+c > a.
    #   Wait, a < b, so a < b <= c, thus a < b+c always. So k=1 means b+c=a which is impossible since a < b <= c implies b+c >= 2b > b > a.
    # k=2: b+c = 2a. Since a < b <= c, we have b >= a+1, c >= b >= a+1, so b+c >= 2a+2 > 2a. Contradiction.
    # k=3: b+c = 3a. Possible? a < b <= c, b+c = 3a.
    #   b >= a+1, c >= b. So b+c >= 2(a+1) = 2a+2. And 3a >= 2a+2 means a >= 2.
    #   Also c = 3a - b <= a (since c <= a would violate c >= b > a... wait c >= b > a, so c > a).
    #   c = 3a - b. We need c >= b, so 3a - b >= b, i.e., 3a >= 2b, i.e., b <= 3a/2.
    #   And b > a, so a < b <= 3a/2.
    #   So for a given a, b ranges from a+1 to floor(3a/2), and c = 3a - b.
    #   We need c <= 256 and c in our set.
    
    # So for the upper half [129, 256], conflicts only arise from k=3 cases.
    # Let's check: for a in [129, 256], b in [a+1, floor(3a/2)], c = 3a - b.
    # c = 3a - b. Since b <= 3a/2, c >= 3a - 3a/2 = 3a/2 > a. Good, c > a.
    # c = 3a - b <= 3a - (a+1) = 2a - 1. For a up to 256, c up to 511, but we need c <= 256.
    # So 3a - b <= 256, i.e., b >= 3a - 256.
    # So b ranges from max(a+1, 3a-256) to min(floor(3a/2), 256).
    
    # This is getting complex. Let's just do a greedy approach with verification.
    
    # Strategy: Start with a candidate set, verify it, and if invalid, remove problematic elements.
    # Or: Build the set incrementally, adding elements that don't create conflicts.
    
    # Let's try: Start with all numbers, then iteratively remove elements that are part of conflicts.
    # An element x is "problematic" if there exist a < b <= c in S with x in {a,b,c} and a | (b+c).
    
    # Simpler: For each pair (b, c) with b <= c, compute the set of a < b that divide b+c.
    # If any such a is in S, then we have a conflict.
    
    # Let's build a conflict matrix for pairs (b, c) -> set of a's that cause conflict.
    
    # Precompute: for each b, c with b <= c, find all a < b such that a | (b+c).
    # Store as: conflict[b][c] = set of a values.
    
    # Then, a set S is valid iff for all b <= c in S, conflict[b][c] ∩ S = empty set.
    # (Where we only care about a < b, which is already enforced in the precomputation.)
    
    # This is still a hard constraint satisfaction problem.
    
    # Let's try a greedy removal approach:
    # 1. Start with S = {1, 2, ..., 256}
    # 2. Find all conflicts (triples a < b <= c in S with a | (b+c))
    # 3. Remove one element from each conflict (e.g., the largest, or the one in most conflicts)
    # 4. Repeat until no conflicts remain.
    
    # But this might remove too many. Let's try a smarter approach.
    
    # Alternative: The seed solution uses [192, 256], size 65.
    # Let's see if we can do better.
    
    # Let's try [160, 256], size 97, and check validity.
    
    def is_valid(S):
        S_set = set(S)
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for j in range(i+1, len(S_sorted)):
                b = S_sorted[j]
                for k in range(j, len(S_sorted)):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # This is O(n^3) which for n=97 is about 97^3/6 ≈ 150k checks. Fast enough.
    # But for n=256, it's 256^3/6 ≈ 2.8M, still fast.
    
    # Let's try various upper tails and see which are valid.
    
    # Actually, let's be smarter. Let's try to find the maximum valid subset.
    
    # Approach: Greedy with random restarts.
    # 1. Try the upper tail [k, 256] for various k, find the smallest k that gives a valid set.
    # 2. Then try to add elements from below k to see if we can expand.
    
    # First, let's find the best upper tail.
    
    best_set = None
    best_size = 0
    
    # Try upper tails
    for k in range(1, 257):
        S = list(range(k, 257))
        if is_valid(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    # Now try to expand from the best upper tail by adding elements from below.
    if best_set is not None:
        S = set(best_set)
        # Try adding elements from 1 to k-1 in some order
        # Use a greedy approach: try to add each element if it doesn't create a conflict.
        
        # We need to check: adding x to S creates a conflict if there exist a < b <= c in S ∪ {x} 
        # with a | (b+c) and x involved.
        
        # x can be a, b, or c.
        # Case 1: x = a. Then we need b <= c in S with x < b and x | (b+c).
        # Case 2: x = b. Then we need a < x in S and c >= x in S with a | (x+c).
        # Case 3: x = c. Then we need a < b in S with b <= x and a | (b+x).
        
        # Let's implement a function to check if adding x to S is safe.
        
        def can_add(S, x):
            S_list = sorted(S)
            # Case 1: x is the smallest (a)
            # Need b <= c in S with x < b and x | (b+c)
            for i in range(len(S_list)):
                b = S_list[i]
                if b <= x:
                    continue
                for j in range(i, len(S_list)):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
            # Case 2: x is the middle (b)
            # Need a < x in S and c >= x in S with a | (x+c)
            for a in S_list:
                if a >= x:
                    break
                for c in S_list:
                    if c < x:
                        continue
                    if (x + c) % a == 0:
                        return False
            # Case 3: x is the largest (c)
            # Need a < b in S with b <= x and a | (b+x)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    if b > x:
                        break
                    if (b + x) % a == 0:
                        return False
            return True
        
        # Greedily add elements from below
        # Try multiple orders
        candidates = list(range(1, k))
        random.shuffle(candidates)
        
        improved = True
        while improved:
            improved = False
            for x in candidates:
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
            if time.time() > deadline:
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Try random restarts with different initial sets
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start with a random subset of the upper range
        # Or start with the best set and randomly remove some, then re-add
        if best_set is None:
            S = set(range(192, 257))
        else:
            S = set(best_set)
        
        # Randomly remove some elements
        remove_count = random.randint(0, 10)
        to_remove = random.sample(list(S), min(remove_count, len(S)))
        for x in to_remove:
            S.discard(x)
        
        # Try to add elements
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        improved = True
        while improved:
            improved = False
            for x in candidates:
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
            if time.time() > deadline:
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    if best_set is None:
        best_set = list(range(192, 257))
    
    return best_set
# EVOLVE-BLOCK-END
