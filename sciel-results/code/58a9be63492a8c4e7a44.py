"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Greedy covering with deadline guard.
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        block = list(min(uncovered))
        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    # If we didn't finish, return what we have (may be invalid but better than timeout)
    return blocks
# EVOLVE-BLOCK-END
