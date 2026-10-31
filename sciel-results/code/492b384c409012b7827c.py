"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _tsubsets(block, t):
    return set(itertools.combinations(sorted(block), t))


def _is_valid_covering(blocks, v, t):
    covered = set()
    for b in blocks:
        covered.update(itertools.combinations(sorted(b), t))
    for c in itertools.combinations(range(v), t):
        if c not in covered:
            return False
    return True


def _redundant(block, blocks, v, t):
    """Check if removing `block` still leaves a valid covering."""
    others = [b for b in blocks if b is not block]
    covered = set()
    for b in others:
        covered.update(itertools.combinations(sorted(b), t))
    for c in itertools.combinations(sorted(block), t):
        if c not in covered:
            return False
    return True


def _greedy_construct(v, k, t, deadline, rng):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        if time.time() > deadline:
            break
        block = list(min(uncovered))
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks, uncovered


def _local_search(blocks, v, k, t, deadline, rng):
    """Iteratively remove redundant blocks."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Shuffle order to avoid bias
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            if _redundant(blocks[i], blocks, v, t):
                del blocks[i]
                changed = True
                break  # restart scan since indices shifted
    return blocks


def construct(v, k, t):
    start_time = time.time()
    # Safety margin: stop at 140s to leave room for final validation
    deadline = start_time + 140.0
    
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    blocks, uncovered = _greedy_construct(v, k, t, deadline, rng)
    
    # If greedy didn't finish, we still have a partial covering — need to complete it
    # but with less time. Just continue greedily for remaining uncovered.
    while uncovered and time.time() < deadline:
        block = list(min(uncovered))
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    
    # Phase 2: Local search — remove redundant blocks
    blocks = _local_search(blocks, v, k, t, deadline, rng)
    
    return blocks
# EVOLVE-BLOCK-END
