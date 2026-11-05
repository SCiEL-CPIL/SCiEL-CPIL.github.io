"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys

def _triple_sums_involving_x(x, S):
    """Return the set of triple sums that involve x, given x is NOT in S."""
    sums = set()
    n = len(S)
    # (x, x, x)
    sums.add(3 * x)
    # (x, x, y) for y in S
    for y in S:
        sums.add(2 * x + y)
    # (x, y, y) for y in S
    for y in S:
        sums.add(x + 2 * y)
    # (x, y, z) for y < z in S
    for i in range(n):
        for j in range(i + 1, n):
            sums.add(x + S[i] + S[j])
    return sums

def _greedy_build(deadline, seed=0):
    """Greedy build with randomization. Returns the largest valid set found."""
    rng = random.Random(seed)
    S = []
    existing_sums = set()
    
    # Build incrementally: try to add elements in a smart order
    # Strategy: start with small numbers, then try larger ones
    # Use a priority approach: try candidates in increasing order but with some randomness
    
    candidates = list(range(1, 4097))
    
    # First pass: greedy in increasing order
    for x in candidates:
        if time.time() > deadline:
            break
        new_sums = _triple_sums_involving_x(x, S)
        # Check for collisions with existing sums
        if new_sums & existing_sums:
            continue
        # Check for collisions among new sums (they should all be distinct)
        # Since we're using a set, if len matches expected count, they're all distinct
        # Actually we need to verify they're all distinct - but the set already handles that
        # The issue is: could two different triples involving x give the same sum?
        # (x,x,x)=3x, (x,x,y)=2x+y, (x,y,y)=x+2y, (x,y,z)=x+y+z
        # These are structurally different unless there's a collision
        # Since we use a set, if any two are equal, the set will have fewer elements
        # But we don't know the expected count easily, so let's just trust the set
        # Actually, the real check is: are all new_sums distinct from each other AND from existing_sums
        # new_sums is a set, so internal collisions are already merged
        # We just need to check: len of the union would increase by len(new_sums - existing_sums)
        # And new_sums must not intersect existing_sums
        if new_sums & existing_sums:
            continue
        # Also need to check that all elements in new_sums are distinct
        # But since it's a set, we can't tell if two different triples gave the same sum
        # We need to verify: count the number of distinct triples vs distinct sums
        # For efficiency, let's just add and track; if there's a collision within new_sums,
        # it means two different triples involving x have the same sum, which is invalid
        # Let's compute the expected number of distinct sums
        n = len(S)
        expected_count = 1 + n + n + n*(n-1)//2  # (xxx) + (xxy) + (xyy) + (xyz)
        if len(new_sums) < expected_count:
            # Collision within new sums - invalid
            continue
        # Valid: add x
        S.append(x)
        existing_sums |= new_sums
    
    return S

def _greedy_build_optimized(deadline, seed=0):
    """Optimized greedy: precompute pair sums for faster triple sum computation."""
    rng = random.Random(seed)
    S = []
    existing_sums = set()
    pair_sums = set()  # sums of pairs (y,z) with y<=z from S
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if x in set(S):
            continue
        
        n = len(S)
        if n == 0:
            new_sums = {3 * x}
            expected = 1
        elif n == 1:
            y = S[0]
            new_sums = {3*x, 2*x+y, x+2*y, x+y+y if x != y else 3*x}
            # More carefully:
            new_sums = {3*x}
            new_sums.add(2*x + y)  # (x,x,y)
            new_sums.add(x + 2*y)  # (x,y,y)
            if x != y:
                # (x,y,y) already counted, (x,y,z) doesn't apply with only one element
                # Actually with n=1, the only pairs are (y,y)
                # (x,y,y) = x+2y already added
                pass
            expected = 1 + n + n + 0  # no y<z pairs with n=1... wait
            # With n=1: (xxx), (xxy), (xyy) where y is the single element
            # (xyz) requires y<z, which needs n>=2
            expected = 3  # (xxx), (xxy), (xyy) - but if x==y, (xxy)=(xyy)=(xxx)
            # This is getting complicated. Let me use a different approach.
            pass
        else:
            # General case
            new_sums = set()
            new_sums.add(3 * x)
            for y in S:
                new_sums.add(2 * x + y)
                new_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i + 1, n):
                    new_sums.add(x + S[i] + S[j])
            expected = 1 + n + n + n*(n-1)//2
        
        if len(new_sums) < expected:
            continue
        if new_sums & existing_sums:
            continue
        
        S.append(x)
        existing_sums |= new_sums
        # Update pair sums
        for y in S[:-1]:
            pair_sums.add(x + y)
        pair_sums.add(2 * x)
    
    return S

def _greedy_with_backtracking(deadline, seed=0):
    """Greedy with occasional backtracking to explore alternatives."""
    rng = random.Random(seed)
    S = []
    existing_sums = set()
    
    # Try multiple random orderings
    best = []
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Create a shuffled but biased ordering
        # Start with small numbers for better packing
        order = list(range(1, 4097))
        # Keep small numbers early, shuffle the rest
        small = order[:200]
        large = order[200:]
        rng.shuffle(large)
        order = small + large
        
        S = []
        existing_sums = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            n = len(S)
            new_sums = set()
            new_sums.add(3 * x)
            for y in S:
                new_sums.add(2 * x + y)
                new_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i + 1, n):
                    new_sums.add(x + S[i] + S[j])
            
            expected = 1 + n + n + n*(n-1)//2
            if len(new_sums) < expected:
                continue
            if new_sums & existing_sums:
                continue
            
            S.append(x)
            existing_sums |= new_sums
        
        if len(S) > len(best):
            best = S[:]
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Try multiple strategies
    best = []
    
    # Strategy 1: Simple greedy in increasing order
    S1 = _greedy_build(deadline, seed=42)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Greedy with randomization
    if time.time() < deadline:
        S2 = _greedy_with_backtracking(deadline, seed=123)
        if len(S2) > len(best):
            best = S2[:]
    
    return best
