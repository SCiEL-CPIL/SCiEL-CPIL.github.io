"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start = time.time()
    DEADLINE = 85.0

    def is_valid(selected):
        """Check if no subset of selected sums to 200."""
        if not selected:
            return True
        reachable = {0}
        for x in selected:
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 in new_sums:
                return False
            reachable |= new_sums
        return True

    def greedy_descending():
        selected = []
        reachable = {0}
        for x in range(100, 0, -1):
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    def greedy_ascending():
        selected = []
        reachable = {0}
        for x in range(1, 101):
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    def structured_block():
        # {66, 67, ..., 100}: size 35, verified valid
        return list(range(66, 101))

    # Try multiple strategies
    candidates = []

    # Strategy 1: Descending greedy
    c1 = greedy_descending()
    candidates.append(c1)

    # Strategy 2: Ascending greedy
    c2 = greedy_ascending()
    candidates.append(c2)

    # Strategy 3: Structured block {66..100}
    c3 = structured_block()
    candidates.append(c3)

    # Strategy 4: Try to extend structured block by checking if we can add any element
    # For {66..100}, we showed no element < 66 can be added.
    # But let's try a different base: maybe skip some large elements to add more small ones.

    # Strategy 5: Greedy that tries to maximize count - try each element, 
    # prefer adding smaller elements first (they're "cheaper" in terms of sum)
    def greedy_smart():
        selected = []
        reachable = {0}
        # Try elements in order of increasing value, but with a twist:
        # For each element, check if adding it keeps validity
        # Prefer elements that don't block many future additions
        for x in range(1, 101):
            if time.time() - start > DEADLINE:
                break
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected

    c5 = greedy_smart()
    candidates.append(c5)

    # Strategy 6: Local search - start from best candidate, try swapping elements
    best = max(candidates, key=lambda c: len(c) if is_valid(c) else 0)
    
    # Try to improve via local search
    if time.time() - start < DEADLINE - 5:
        improved = list(best)
        improved_set = set(improved)
        improved_valid = is_valid(improved)
        
        # Try adding elements not in the set
        for x in range(1, 101):
            if time.time() - start > DEADLINE - 2:
                break
            if x in improved_set:
                continue
            trial = improved + [x]
            if is_valid(trial):
                improved = trial
                improved_set.add(x)
        
        # Try swapping: remove one element, add two others
        for i in range(len(improved)):
            if time.time() - start > DEADLINE - 2:
                break
            for j in range(i+1, len(improved)):
                if time.time() - start > DEADLINE - 2:
                    break
                removed = {improved[i], improved[j]}
                trial = [x for x in improved if x not in removed]
                for a in range(1, 101):
                    if a in removed or a in set(trial):
                        continue
                    trial_a = trial + [a]
                    if is_valid(trial_a):
                        for b in range(a+1, 101):
                            if b in removed or b in set(trial_a):
                                continue
                            trial_ab = trial_a + [b]
                            if is_valid(trial_ab) and len(trial_ab) > len(improved):
                                improved = trial_ab
                                break
                        break
    
    candidates.append(improved)

    # Return the best valid candidate
    best_candidate = [1]
    best_len = 0
    for c in candidates:
        if is_valid(c) and len(c) > best_len:
            best_candidate = c
            best_len = len(c)

    return best_candidate
# EVOLVE-BLOCK-END
