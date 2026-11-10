"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin
    
    def make_bitset():
        # bit i set means sum i is reachable, for i in 0..200
        return np.zeros(201, dtype=bool)
    
    def add_to_bitset(bs, x):
        # returns new bitset after adding x, or None if 200 would be reached
        # new sums: old sums + x
        new_sums = np.zeros(201, dtype=bool)
        for i in range(201 - x):
            if bs[i]:
                new_sums[i + x] = True
        if new_sums[200]:
            return None
        return bs | new_sums
    
    # Greedy: try largest first
    def greedy(order):
        selected = []
        bs = make_bitset()
        bs[0] = True
        for x in order:
            new_bs = add_to_bitset(bs, x)
            if new_bs is not None:
                selected.append(x)
                bs = new_bs
        return selected
    
    # Try largest-first
    best = greedy(range(100, 0, -1))
    best_size = len(best)
    
    # Try smallest-first
    if time.time() < deadline:
        candidate = greedy(range(1, 101))
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Try random orderings
    import random
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        order = list(range(1, 101))
        rng.shuffle(order)
        candidate = greedy(order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(best)
    
    # Local search: try to add elements not in best
    if time.time() < deadline - 5.0:
        best_set = set(best)
        bs = make_bitset()
        bs[0] = True
        for x in best:
            bs = add_to_bitset(bs, x)  # should not be None
        # Try adding each missing element
        for x in range(1, 101):
            if x not in best_set:
                new_bs = add_to_bitset(bs, x)
                if new_bs is not None:
                    best.append(x)
                    best_set.add(x)
                    bs = new_bs
                    best_size = len(best)
    
    # Another pass of local search
    if time.time() < deadline - 5.0:
        best_set = set(best)
        bs = make_bitset()
        bs[0] = True
        for x in best:
            bs = add_to_bitset(bs, x)
        for x in range(1, 101):
            if x not in best_set:
                new_bs = add_to_bitset(bs, x)
                if new_bs is not None:
                    best.append(x)
                    best_set.add(x)
                    bs = new_bs
                    best_size = len(best)
    
    if time.time() > deadline:
        return best
    
    # Try swap-based local search: remove one element, try to add two
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            best_set = set(best)
            improved = False
            for remove_x in list(best):
                # Try removing remove_x and see if we can add more
                remaining = [x for x in best if x != remove_x]
                bs2 = make_bitset()
                bs2[0] = True
                valid = True
                for x in remaining:
                    new_bs2 = add_to_bitset(bs2, x)
                    if new_bs2 is None:
                        valid = False
                        break
                    bs2 = new_bs2
                if not valid:
                    continue
                # Try adding all missing elements
                new_best = list(remaining)
                new_set = set(remaining)
                for x in range(1, 101):
                    if x not in new_set:
                        new_bs2 = add_to_bitset(bs2, x)
                        if new_bs2 is not None:
                            new_best.append(x)
                            new_set.add(x)
                            bs2 = new_bs2
                if len(new_best) > best_size:
                    best = new_best
                    best_size = len(best)
                    improved = True
                    break
            if not improved:
                break
    
    return best
# EVOLVE-BLOCK-END
