"""Large sets without a triple and all its pair sums."""
# EVOLVE-BLOCK-START
import time

def construct():
    t_start = time.time()
    deadline = t_start + 80  # Safety margin
    limit = 128
    
    best_set = set()
    best_size = 0
    
    def try_add_fast(S, x):
        """Check if x can be added to S without creating forbidden triple."""
        if x > limit or x in S:
            return False
        
        # For each pair (a, b) in S with a < b:
        # Check if adding x creates a forbidden triple
        # New triples involving x:
        #   (x, a, b) where a+b, a+x, b+x all in S
        # So for each pair (a,b) in S: if a+b in S and a+x in S and b+x in S -> forbidden
        # But a+x and b+x are NOT in S yet (only x is new)
        # Wait - the triple is (a, b, x) where a+b, a+x, b+x are all in S_after_add
        # S_after_add = S ∪ {x}
        # a+b must be in S_after_add
        # a+x must be in S_after_add  
        # b+x must be in S_after_add
        
        # Since x is new, a+x and b+x can only be in S_after_add if they equal x (impossible since a,b >= 1)
        # OR they are already in S
        
        # So: a+x in S_after_add means (a+x) in S or (a+x) == x (impossible)
        #     so (a+x) in S
        # Similarly b+x in S
        # And a+b in S_after_add means (a+b) in S or (a+b) == x
        
        # Case 1: a+b == x. Then need a+x in S and b+x in S
        # Case 2: a+b != x. Then need a+b in S, a+x in S, b+x in S
        
        for a in S:
            if a + x > limit:
                continue
            ax = a + x
            if ax not in S:
                continue
            for b in S:
                if b <= a:
                    continue
                if b + x > limit:
                    break
                bx = b + x
                if bx not in S:
                    continue
                ab = a + b
                if ab == x or ab in S:
                    # Found forbidden triple (a, b, x)
                    return False
        return True
    
    # Strategy: Greedy + local search
    # Start with odds (known good base)
    S1 = set(range(1, limit+1, 2))
    for x in range(limit, 0, -2):  # try evens high to low
        if time.time() > deadline:
            break
        if try_add_fast(S1, x):
            S1.add(x)
    # Local search
    for _ in range(3000):
        if time.time() > deadline:
            break
        added = False
        for x in range(limit, 0, -1):
            if x in S1:
                continue
            if try_add_fast(S1, x):
                S1.add(x)
                added = True
                break
        if not added:
            break
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = set(S1)
    
    # Strategy 2: Greedy from high to low
    S2 = set()
    for x in range(limit, 0, -1):
        if time.time() > deadline:
            break
        if try_add_fast(S2, x):
            S2.add(x)
    for _ in range(3000):
        if time.time() > deadline:
            break
        added = False
        for x in range(1, limit+1):
            if x in S2:
                continue
            if try_add_fast(S2, x):
                S2.add(x)
                added = True
                break
        if not added:
            break
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = set(S2)
    
    # Strategy 3: Greedy from low to high
    S3 = set()
    for x in range(1, limit+1):
        if time.time() > deadline:
            break
        if try_add_fast(S3, x):
            S3.add(x)
    for _ in range(3000):
        if time.time() > deadline:
            break
        added = False
        for x in range(limit, 0, -1):
            if x in S3:
                continue
            if try_add_fast(S3, x):
                S3.add(x)
                added = True
                break
        if not added:
            break
    if len(S3) > best_size:
        best_size = len(S3)
        best_set = set(S3)
    
    # Final polish
    for _ in range(2000):
        if time.time() > deadline:
            break
        added = False
        for x in range(limit, 0, -1):
            if x in best_set:
                continue
            if try_add_fast(best_set, x):
                best_set.add(x)
                added = True
                break
        if not added:
            break
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
