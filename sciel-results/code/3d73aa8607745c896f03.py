"""Improved snake-in-the-box: randomized greedy with bit-order diversity and lookahead."""
import time
import random

N = 9
TOTAL = 1 << N
DEADLINE = 110.0  # safety margin before 120s kill (evaluator says 150s wall-clock)

# EVOLVE-BLOCK-START
def _valid_extension(v, path, used):
    """Check v is not used and not adjacent (dist 1) to any non-consecutive path vertex."""
    if v in used:
        return False
    # Must differ by >=2 bits from all path vertices except the last (consecutive)
    # path[:-1] are non-consecutive with v (v would be appended after path[-1])
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, used, bit_order, lookahead=False):
    """Extend path greedily using given bit_order. Returns new path or None if stuck."""
    cur = path[-1]
    best_v = None
    best_score = -1
    for b in bit_order:
        v = cur ^ (1 << b)
        if v in used:
            continue
        if not _valid_extension(v, path, used):
            continue
        if lookahead:
            # Count how many of v's neighbors are still free (future option count)
            score = 0
            for bb in range(N):
                w = v ^ (1 << bb)
                if w not in used:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        else:
            best_v = v
            break
    if best_v is None:
        return None
    path.append(best_v)
    used.add(best_v)
    return path

def _construct_greedy(bit_order, lookahead=False):
    """Build a snake using the given bit order and optional lookahead."""
    path = [0]
    used = {0}
    while True:
        new_path = _greedy_extend(path, used, bit_order, lookahead)
        if new_path is None:
            break
    return path

def _try_backtrack_extend(path, used, depth_limit=3, max_steps=200):
    """Try to extend a stuck snake by undoing recent choices and re-extending.
    Returns an improved path or the original."""
    if len(path) <= depth_limit:
        return path
    # Try removing the last few vertices and re-extending with shuffled bit orders
    for back in range(1, depth_limit + 1):
        if len(path) <= back:
            break
        # Save state
        removed = path[-back:]
        for v in removed:
            used.discard(v)
        path = path[:-back]
        # Try a few random bit orders to re-extend
        for _ in range(max_steps // depth_limit):
            bo = list(range(N))
            random.shuffle(bo)
            test_path = path[:]
            test_used = used.copy()
            # Re-add removed? No, we want to extend from the truncated end differently
            while True:
                np = _greedy_extend(test_path, test_used, bo, lookahead=True)
                if np is None:
                    break
            if len(test_path) > len(path) + back:
                # Improved
                return test_path
            # Restore and try again
            for v in test_path[len(path):]:
                test_used.discard(v)
        # Restore removed for next back depth (only if not improved)
        # Actually we already removed; re-add to continue outer loop
        for v in removed:
            used.add(v)
        path = path + removed
    return path

def construct():
    start = time.time()
    best_path = None
    best_len = 0

    # First: run the original greedy (bit order 0..N-1) for a baseline
    base = _construct_greedy(list(range(N)), lookahead=False)
    best_path = base
    best_len = len(base)

    # Randomized restarts with diverse bit orders
    rng = random.Random(12345)
    iterations = 0
    while True:
        elapsed = time.time() - start
        if elapsed > DEADLINE - 5.0:
            break
        iterations += 1

        # Generate a random bit order
        bo = list(range(N))
        rng.shuffle(bo)

        # Occasionally use lookahead
        use_la = (iterations % 3 == 0)

        path = _construct_greedy(bo, lookahead=use_la)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

            # Try backtracking extension on the improved path
            elapsed = time.time() - start
            if elapsed < DEADLINE - 10.0:
                used = set(best_path)
                improved = _try_backtrack_extend(best_path, used, depth_limit=4, max_steps=300)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = improved

    # Final: if time permits, do a few more lookahead-only runs
    elapsed = time.time() - start
    if elapsed < DEADLINE - 2.0:
        for _ in range(20):
            if time.time() - start > DEADLINE - 1.0:
                break
            bo = list(range(N))
            rng.shuffle(bo)
            path = _construct_greedy(bo, lookahead=True)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

    return best_path
# EVOLVE-BLOCK-END
