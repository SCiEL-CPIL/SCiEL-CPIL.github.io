"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    from math import lcm
    from fractions import Fraction

    def greedy_fill(seed_val, forbidden=set()):
        """
        Greedily fills a set starting with seed_val, avoiding forbidden numbers.
        """
        if seed_val in forbidden:
            return set()
        
        selected = [seed_val]
        # Iterate through candidates in ascending order to prioritize small numbers (high reciprocal)
        for x in range(2, 257):
            if x == seed_val or x in forbidden:
                continue
            # Check compatibility with all selected
            compatible = True
            for y in selected:
                if lcm(x, y) <= 256:
                    compatible = False
                    break
            if compatible:
                selected.append(x)
                
        return set(selected)

    best_set = set()
    best_sum = Fraction(0)

    # Try starting with each small number as the "anchor"
    # The anchor determines the "cluster" we belong to.
    # We try anchors 2..10.
    for anchor in range(2, 11):
        if time.time() > deadline:
            break
            
        # When we pick an anchor, we must exclude any number that conflicts with it
        # But the greedy loop handles conflicts dynamically.
        # However, if we start with 3, the greedy loop will naturally skip 2 if 2 conflicts with 3.
        # Wait, the greedy loop iterates x=2..256.
        # If anchor=3, and we just run the standard greedy, it will pick 2 first!
        # So we must FORCE the anchor to be the first element, and prevent smaller conflicting elements from being picked?
        # No, if we pick 3, we can't pick 2.
        # So for anchor=3, we should run greedy but SKIP 2?
        # Or better: Run greedy, but initialize with anchor, and iterate x > anchor?
        # No, we might want to pick x < anchor if compatible.
        # But if anchor is small, x < anchor is even smaller.
        # If x < anchor and compatible with anchor, we should pick it.
        # But if we pick x < anchor, then anchor might not be the "true" anchor.
        
        # Strategy:
        # For each anchor a, we assume a is in the set.
        # We find the maximal set containing a using greedy.
        # To do this, we iterate x from 2 to 256.
        # If x == a, add it.
        # If x conflicts with any currently selected, skip.
        # This will naturally find the greedy set for that anchor, 
        # BUT it might pick a smaller number b < a first, which conflicts with a, thus blocking a.
        # To force a to be included, we must ensure no b < a that conflicts with a is picked.
        # This is complex.
        
        # Simpler: Just run the standard greedy (starts with 2).
        # And also run a variant that skips 2 (starts with 3, 4, etc).
        # Variant: Iterate x from 2 to 256.
        # If x is in a "blacklist", skip.
        # For anchor=3, blacklist={2}.
        # For anchor=4, blacklist={2,3}? No, 3 and 4 are compatible? lcm(3,4)=12. No.
        # So for anchor=4, we can't have 2 or 3.
        
        # Let's just try blacklisting {2}, {2,3}, {2,3,4}, etc.
        pass

    # Refined Strategy:
    # The optimal set is likely one of the "clusters".
    # Cluster 2: Contains 2. Maximal greedy set.
    # Cluster 3: Contains 3, not 2. Maximal greedy set.
    # Cluster 4: Contains 4, not 2,3.
    # ...
    
    # We will generate the greedy set for each "exclusion prefix".
    # Exclusion prefix P means we forbid all numbers in P.
    # We try P = {}, {2}, {2,3}, {2,3,4}, {2,3,4,5}, {2,3,4,5,6}, {2,3,4,5,6,7}, {2,3,4,5,6,7,8}, {2,3,4,5,6,7,8,9}, {2,3,4,5,6,7,8,9,10}
    
    for k in range(11):
        if time.time() > deadline:
            break
        
        forbidden = set(range(2, 2 + k))
        
        selected = []
        for x in range(2, 257):
            if x in forbidden:
                continue
            compatible = True
            for y in selected:
                if lcm(x, y) <= 256:
                    compatible = False
                    break
            if compatible:
                selected.append(x)
        
        if not selected:
            continue
            
        s = sum(Fraction(1, x) for x in selected)
        if s > best_sum:
            best_sum = s
            best_set = set(selected)

    return list(best_set)
# EVOLVE-BLOCK-END
