import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0
    T = 200
    N = 100
    mask = (1 << (T + 1)) - 1
    
    best_set = None
    best_size = -1
    
    def greedy(order):
        selected = []
        reachable = 1  # bit 0 is set
        for x in order:
            if x > T:
                continue
            # Check if adding x creates sum T: is bit (T-x) set in reachable?
            if x <= T and ((reachable >> (T - x)) & 1):
                continue
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= mask
        return selected, reachable
    
    def local_improve(selected, reachable):
        """Try to insert excluded elements one at a time if safe."""
        sset = set(selected)
        excluded = [x for x in range(N, 0, -1) if x not in sset]
        changed = True
        while changed:
            changed = False
            for x in excluded:
                if x in sset:
                    continue
                # Can we add x without creating sum T?
                # Adding x creates sum T iff bit (T-x) is set in reachable
                if x <= T and ((reachable >> (T - x)) & 1):
                    continue
                # Safe to add
                selected.append(x)
                sset.add(x)
                reachable |= (reachable << x)
                reachable &= mask
                excluded.remove(x)
                changed = True
                if time.time() > deadline:
                    return selected, reachable
            # After one full pass with no changes, we're at a local optimum
            # for insertion. Adding elements only creates more sums, so
            # no previously-blocked element becomes unblocked.
        return selected, reachable
    
    # 1. Deterministic greedy (descending)
    baseline, reach_base = greedy(range(100, 0, -1))
    baseline, reach_base = local_improve(baseline, reach_base)
    best_set = baseline
    best_size = len(baseline)
    
    # 2. Deterministic greedy (ascending)
    asc, reach_asc = greedy(range(1, 101))
    asc, reach_asc = local_improve(asc, reach_asc)
    if len(asc) > best_size:
        best_set = asc
        best_size = len(asc)
    
    # 3. Randomized greedy with local improvement
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        candidate, reach_cand = greedy(order)
        candidate, reach_cand = local_improve(candidate, reach_cand)
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)
    
    # Fallback: ensure nonempty
    if not best_set:
        best_set = [100]
    
    return best_set
# EVOLVE-BLOCK-END
