import time
from itertools import combinations

def is_valid(s):
    """Check cross-cardinality subset sum constraint."""
    k = len(s)
    sums_by_card = [set() for _ in range(k + 1)]
    sums_by_card[0].add(0)
    for c in range(1, k + 1):
        for comb in combinations(s, c):
            sums_by_card[c].add(sum(comb))
    seen = {}
    for c in range(k + 1):
        for sm in sums_by_card[c]:
            if sm in seen:
                return False
            seen[sm] = c
    return True

def construct():
    start = time.time()
    deadline = start + 85

    # Known valid set: superincreasing {1,2,4,8,16,32,64,128,256}
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = base[:]

    if not is_valid(base):
        return base

    # Greedy: try adding each remaining element
    used = set(base)
    candidates = [x for x in range(1, 257) if x not in used]

    for x in candidates:
        if time.time() > deadline - 5:
            break
        new_set = sorted(base + [x])
        if is_valid(new_set):
            best = new_set
            base = new_set
            used.add(x)

    # Try to improve with local search: swap elements
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        improved = False
        current_set = set(best)
        for i, elem in enumerate(best):
            for x in range(1, 257):
                if time.time() > deadline - 5:
                    break
                if x in current_set:
                    continue
                new_set = sorted(list(current_set - {elem}) + [x])
                if is_valid(new_set):
                    # Check if this helps (same size, but maybe allows further extension)
                    # For now, just track if we can extend further
                    pass
        # Try extending again after potential swaps
        used = set(best)
        candidates = [x for x in range(1, 257) if x not in used]
        for x in candidates:
            if time.time() > deadline - 5:
                break
            new_set = sorted(best + [x])
            if is_valid(new_set):
                best = new_set
                used.add(x)

    return best
# EVOLVE-BLOCK-END
