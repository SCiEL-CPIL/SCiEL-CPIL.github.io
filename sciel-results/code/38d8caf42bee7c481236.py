"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Start with a known large valid set: {64, 68, 69, ..., 100}
    # Min 3-sum: 64+68+69 = 201 > 200, max 2-sum: 99+100 = 199 < 200
    base = set(range(64, 101))
    base.discard(65)
    base.discard(66)
    base.discard(67)
    # So base = {64, 68, 69, ..., 100}, size = 1 + 33 = 34
    # Wait: 68 to 100 inclusive is 33 elements, plus 64 = 34
    # Let me recount: 64, then 68,69,...,100. 68 to 100 is 100-68+1 = 33. Total 34.
    
    # Actually let me reconsider. Let me try to build optimally.
    # Strategy: greedy from largest, but with backtracking/local search.
    
    # Better approach: use the fact that if all elements > 100, no 2 sum to 200.
    # If elements > 66, no 3 sum to 200 (min 3: 67+68+69=204).
    # So {67..100} is valid, size 34.
    # Can we add 66? 66+67+68=201>200, but 66+74=140... need to check if 66+x=200 for x in set (no, max is 100, 66+100=166). 
    # 66+a+b=200 where a,b in {67..100}: min a+b=67+68=135, so 66+135=201>200. Valid!
    # {66, 67..100} = 35 elements. Valid.
    # Add 65? 65+67+68=200. INVALID.
    # Add 64? 64+67+69=200. INVALID.
    # So {66, 67..100} is valid with 35 elements.
    
    # Can we do better by excluding some and adding more?
    # What if we exclude 67 and add 65, 64?
    # {64, 65, 66, 68, 69, ..., 100}: check 64+65+66=195, 64+65+71=200? 71 in set? Yes. INVALID.
    
    # Let me just do a proper search.
    
    # Approach: Start with {66..100} (35 elements). Try adding each of 1-65 if valid.
    selected = set(range(66, 101))
    
    # Track which sums up to 200 are reachable
    def compute_reachable(s):
        """Return set of all subset sums of s that are <= 200."""
        reachable = {0}
        for x in s:
            new = {val + x for val in reachable if val + x <= 200}
            reachable |= new
        return reachable
    
    # For efficiency, maintain reachable incrementally
    reachable = compute_reachable(selected)
    
    candidates = list(range(65, 0, -1))  # try adding smaller elements
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        # Check if adding x creates a 200-sum
        # 200 is reachable with x iff (200 - x) is in reachable
        if (200 - x) not in reachable:
            # Adding x is safe
            selected.add(x)
            # Update reachable
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
    
    # Now try to remove some elements and add others to improve
    # Local search: try swapping
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        # Try removing one element and adding two others
        for rem in list(selected):
            if time.time() > deadline:
                break
            if len(selected) <= 1:
                break
            test_selected = selected - {rem}
            test_reachable = compute_reachable(test_selected)
            
            # Try adding pairs of non-selected elements
            non_sel = [x for x in range(1, 101) if x not in test_selected]
            for i, a in enumerate(non_sel):
                if time.time() > deadline:
                    break
                if (200 - a) in test_reachable:
                    continue
                # Check adding a
                temp_reach = set(test_reachable)
                new_a = {s + a for s in test_reachable if s + a <= 200}
                temp_reach |= new_a
                if 200 in temp_reach:
                    continue
                
                for b in non_sel[i+1:]:
                    if time.time() > deadline:
                        break
                    if b == a:
                        continue
                    if (200 - b) in temp_reach:
                        continue
                    # This would give len+1
                    test_selected.add(a)
                    test_selected.add(b)
                    # Verify
                    final_reach = compute_reachable(test_selected)
                    if 200 not in final_reach:
                        selected = test_selected
                        reachable = final_reach
                        improved = True
                        break
                    else:
                        test_selected.discard(a)
                        test_selected.discard(b)
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return list(selected)

# EVOLVE-BLOCK-END
