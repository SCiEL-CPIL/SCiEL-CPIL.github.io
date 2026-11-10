"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
def construct():
    def is_maximal(selected):
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                sums.add(selected[i] + selected[j])
        for x in range(1, N + 1):
            if x in selected:
                continue
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add and 2 * x in sums:
                can_add = False
            if can_add:
                return False
        return True

    def greedy_construct(order):
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    # Strategy 1: Largest first
    order_large = range(N, 0, -1)
    set_large = greedy_construct(order_large)
    
    # Strategy 2: Smallest first
    order_small = range(1, N + 1)
    set_small = greedy_construct(order_small)
    
    # Strategy 3: Step 2 (odd numbers first, then even? or just skip)
    # Let's stick to the two main deterministic ones first.
    
    best = set_large if len(set_large) <= len(set_small) else set_small
    
    # Ensure maximality (should be true by construction)
    if not is_maximal(best):
        # Fallback to small if large failed maximality check (unlikely)
        if is_maximal(set_small):
            best = set_small
        else:
            # If both fail (logic error), return small
            best = set_small
            
    return best
# EVOLVE-BLOCK-END
