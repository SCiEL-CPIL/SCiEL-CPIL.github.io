"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # 85s safety margin within 90s limit
    
    # Greedy construction: build a 4-AP-free set by iterating through [1,512]
    # and adding each number if it doesn't complete a 4-term AP with already selected numbers.
    # This is the seed algorithm, which produces a valid 4-AP-free set.
    
    def greedy_construction():
        selected = set()
        for x in range(1, 513):
            # Check if adding x creates a 4-term AP
            # A 4-term AP: a, a+d, a+2d, a+3d
            # x could be:
            # 1. The last term: x-3d, x-2d, x-d, x -> need x-d, x-2d, x-3d in selected
            # 2. The third term: x-2d, x-d, x, x+d -> need x-2d, x-d, x+d in selected (but x+d > x, not yet selected)
            #    Actually since we process in order, x+d hasn't been considered yet
            # 3. The second term: x-d, x, x+d, x+2d -> need x-d, x+d, x+2d (x+d, x+2d not yet selected)
            # 4. The first term: x, x+d, x+2d, x+3d -> need x+d, x+2d, x+3d (none yet selected)
            # So only case 1 matters: check if there exists d>0 such that x-d, x-2d, x-3d all in selected
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected
    
    # Start with greedy
    selected = greedy_construction()
    best_set = set(selected)
    best_size = len(best_set)
    
    # Try to improve by local search: try to swap out elements to allow more to be added
    # Or try different orderings
    
    # Try reverse greedy: process from 512 down to 1
    def greedy_reverse():
        selected = set()
        for x in range(512, 0, -1):
            # Check if adding x creates a 4-term AP with selected elements
            # x could be first term: x, x+d, x+2d, x+3d where x+d, x+2d, x+3d in selected
            # x could be second term: x-d, x, x+d, x+2d where x-d, x+d, x+2d in selected
            # x could be third term: x-2d, x-d, x, x+d where x-2d, x-d, x+d in selected
            # x could be last term: x-3d, x-2d, x-d, x where x-3d, x-2d, x-d in selected
            valid = True
            # Case: x is first term
            for d in range(1, (512 - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # Case: x is second term
            for d in range(1, x // 1 + 1):
                if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected and (x + 2*d) <= 512:
                    valid = False
                    break
            if not valid:
                continue
            # Case: x is third term
            for d in range(1, x // 2 + 1):
                if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected and (x + d) <= 512:
                    valid = False
                    break
            if not valid:
                continue
            # Case: x is last term
            for d in range(1, x // 3 + 1):
                if (x - 3*d) in selected and (x - 2*d) in selected and (x - d) in selected:
                    valid = False
                    break
            if valid:
                selected.add(x)
        return selected
    
    # Try random orderings
    import random
    
    def greedy_random_order():
        order = list(range(1, 513))
        random.shuffle(order)
        selected = set()
        for x in order:
            # Check all positions x could be in a 4-term AP
            valid = True
            # x is last term: need x-d, x-2d, x-3d in selected
            for d in range(1, (x - 1) // 3 + 1):
                if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # x is first term: need x+d, x+2d, x+3d in selected
            for d in range(1, (512 - x) // 3 + 1):
                if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                    valid = False
                    break
            if not valid:
                continue
            # x is second term: need x-d, x+d, x+2d in selected
            for d in range(1, x + 1):
                if (x - d) >= 1 and (x + 2*d) <= 512:
                    if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                        valid = False
                        break
            if not valid:
                continue
            # x is third term: need x-2d, x-d, x+d in selected
            for d in range(1, x // 2 + 1):
                if (x - 2*d) >= 1 and (x + d) <= 512:
                    if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                        valid = False
                        break
            if valid:
                selected.add(x)
        return selected
    
    # Try reverse greedy
    rev_set = greedy_reverse()
    if len(rev_set) > best_size:
        best_set = set(rev_set)
        best_size = len(best_set)
    
    # Try several random orderings if time permits
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5:
        rand_set = greedy_random_order()
        if len(rand_set) > best_size:
            best_set = set(rand_set)
            best_size = len(best_set)
        attempts += 1
        # Early stop if we've tried many and not improving
        if attempts > 200:
            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
