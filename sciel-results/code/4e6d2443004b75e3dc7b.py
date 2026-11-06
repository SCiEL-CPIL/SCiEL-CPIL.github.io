"""Combined program: structural framework (deadline + multi-strategy + local search)
from the inspiration, with a fast vectorized validity check and aggressive search.
Keeps the seed's greedy snake as the guaranteed baseline."""
import time
import random
import numpy as np

N = 9
TOTAL_VERTICES = 1 << N
BIT_MASKS = [1 << b for b in range(N)]
DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 150s kill
TIME_LIMIT = 150.0

# Precompute popcount table for the whole cube (fast scalar bit_count lookups)
POP = np.array([i.bit_count() for i in range(TOTAL_VERTICES)], dtype=np.uint8)


def _conflict_mask(cand, path_arr):
    """Boolean array: True for path vertices that conflict with cand (dist==1)."""
    return POP[cand ^ path_arr] == 1


def _can_append(v, path, used, path_arr):
    """Check if v can be appended to the end of path (induced-path rule)."""
    if v in used:
        return False
    # Must not be distance 1 from any non-last vertex
    if path_arr.size >= 2 and (_conflict_mask(v, path_arr[:-1])).any():
        return False
    return True


def _greedy_extend(path, used, bit_order, deadline):
    """Extend path greedily using the given bit order until stuck or deadline."""
    path_arr = np.array(path, dtype=np.int64)
    while True:
        if time.time() > deadline:
            return path, used, path_arr
        cur = path[-1]
        nxt = None
        for b in bit_order:
            v = cur ^ BIT_MASKS[b]
            if v in used:
                continue
            if path_arr.size >= 2 and (_conflict_mask(v, path_arr[:-1])).any():
                continue
            nxt = v
            break
        if nxt is None:
            return path, used, path_arr
        path.append(nxt)
        used.add(nxt)
        path_arr = np.append(path_arr, nxt)


def _seed_greedy():
    """The original seed greedy (lowest bit flip) -> guaranteed 104-edge baseline."""
    path, used = [0], {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ BIT_MASKS[b]
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _local_search_extend(path, deadline, max_backtrack=3):
    """Greedy with limited backtracking to escape dead-ends and extend the path."""
    path = list(path)
    used = set(path)

    def try_extend(depth):
        if depth <= 0 or time.time() > deadline:
            return False
        cur = path[-1]
        for b in range(N):
            v = cur ^ BIT_MASKS[b]
            if v in used:
                continue
            if any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            path.append(v)
            used.add(v)
            if try_extend(depth - 1):
                return True
            path.pop()
            used.remove(v)
        return False

    try_extend(max_backtrack)
    return path


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - DEADLINE_MARGIN

    # Strategy 0: seed greedy baseline (fast, guaranteed 104 edges)
    best_path = _seed_greedy()
    best_len = len(best_path) - 1

    # Strategy 1: randomized greedy restarts with different bit orders / starts
    rng = random.Random(12345)
    trial = 0
    while time.time() < deadline:
        start_v = rng.randint(0, TOTAL_VERTICES - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path, used, _ = _greedy_extend([start_v], {start_v}, bit_order, deadline)
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path[:]
        trial += 1
        # periodic safety check (loop condition already guards deadline)

    # Strategy 2: local-search / backtracking to extend the best path
    if time.time() < deadline:
        improved = _local_search_extend(best_path, deadline, max_backtrack=3)
        if len(improved) - 1 > best_len:
            best_len = len(improved) - 1
            best_path = improved

    # Strategy 3: a few more randomized restarts if we have time, then try to extend
    rng2 = random.Random(98765)
    for _ in range(200):
        if time.time() > deadline:
            break
        start_v = rng2.randint(0, TOTAL_VERTICES - 1)
        bit_order = list(range(N))
        rng2.shuffle(bit_order)
        path, used, _ = _greedy_extend([start_v], {start_v}, bit_order, deadline)
        if len(path) - 1 > best_len:
            best_len = len(path) - 1
            best_path = path[:]
        if time.time() < deadline:
            improved = _local_search_extend(path, deadline, max_backtrack=2)
            if len(improved) - 1 > best_len:
                best_len = len(improved) - 1
                best_path = improved

    return best_path
