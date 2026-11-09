import time

# EVOLVE-BLOCK-START
def _greedy(order):
    selected = []
    reachable = {0}
    for x in order:
        # new sums created by adding x
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _valid(subset):
    # verify no subset sums to 200
    reachable = {0}
    for x in subset:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return False
        reachable |= new
    return True

def _local_search(subset):
    # try to add any excluded element that keeps validity
    S = set(subset)
    changed = True
    while changed:
        changed = False
        for x in range(1, 101):
            if x in S:
                continue
            trial = S | {x}
            if _valid(trial):
                S = trial
                changed = True
    # try swaps: remove one, add one (net zero) to possibly enable more later
    return list(S)

def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin under 90s kill

    orders = [
        list(range(100, 0, -1)),   # high to low (seed)
        list(range(1, 101)),       # low to high
    ]

    best = []
    for order in orders:
        cand = _greedy(order)
        cand = _local_search(cand)
        if len(cand) > len(best):
            best = cand
        if time.time() - t0 > DEADLINE:
            break

    # If we still have time, try a few more randomized orderings
    import random
    rng = random.Random(12345)
    extra = 0
    while time.time() - t0 < DEADLINE - 5.0 and extra < 40:
        order = list(range(1, 101))
        rng.shuffle(order)
        cand = _greedy(order)
        cand = _local_search(cand)
        if len(cand) > len(best):
            best = cand
        extra += 1

    # Final safety: ensure best is nonempty and valid
    if not best:
        best = list(range(100, 0, -1))
    if not _valid(best):
        # fall back to guaranteed-valid high range
        best = list(range(67, 101))
    return best
# EVOLVE-BLOCK-END
