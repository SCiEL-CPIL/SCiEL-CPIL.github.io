"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TOTAL = 1 << N  # 512
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s kill


# EVOLVE-BLOCK-START

def _is_valid_snake(path):
    """Check if a list of integers forms a valid induced path in Q_N."""
    n = len(path)
    if n < 2:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_seed():
    """Generate a greedy snake of length ~104 in Q_9."""
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check against all non-consecutive vertices
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path


def _dfs_backtrack(path, used, deadline, best_path):
    """
    DFS backtracking to extend the snake.
    path: current path (list)
    used: set of used vertices
    deadline: time.time() limit
    best_path: list to store the best path found so far
    """
    if len(path) > len(best_path):
        best_path.clear()
        best_path.extend(path)

    if time.time() > deadline:
        return

    cur = path[-1]
    # Generate candidate neighbors
    candidates = []
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used:
            continue
        # Check against all non-consecutive vertices
        valid = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                valid = False
                break
        if valid:
            candidates.append(v)

    if not candidates:
        return

    # Order candidates: prefer those that lead to more future options
    # Use a simple heuristic: prefer vertices with fewer used neighbors
    def heuristic(v):
        count = 0
        for b in range(N):
            nb = v ^ (1 << b)
            if nb in used:
                count += 1
        return count

    candidates.sort(key=heuristic)

    for v in candidates:
        if time.time() > deadline:
            return
        path.append(v)
        used.add(v)
        _dfs_backtrack(path, used, deadline, best_path)
        used.remove(v)
        path.pop()
        if time.time() > deadline:
            return


def _warnsdorff_greedy():
    """
    Warnsdorff-like greedy: at each step, pick the neighbor that minimizes
    the number of future options (degree in the remaining graph).
    """
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                # Count how many neighbors of v are still available
                degree = 0
                for b2 in range(N):
                    nb = v ^ (1 << b2)
                    if nb not in used:
                        # Check if nb would be valid (not adjacent to non-consecutive)
                        ok = True
                        for p in path[:-1]:
                            if (nb ^ p).bit_count() == 1:
                                ok = False
                                break
                        if ok:
                            degree += 1
                candidates.append((degree, v))
        if not candidates:
            break
        candidates.sort()
        nxt = candidates[0][1]
        path.append(nxt)
        used.add(nxt)
    return path


def _multi_start_dfs(deadline, best_path, seed_path):
    """
    Try multiple starting points and initial greedy prefixes, then use DFS.
    """
    start_time = time.time()
    rng = random.Random(42)

    # Try the default seed
    _dfs_backtrack(list(seed_path), set(seed_path), deadline, best_path)

    # Try different starting vertices
    for start in range(TOTAL):
        if time.time() > deadline:
            break
        # Try a short greedy prefix from this start
        path = [start]
        used = {start}
        # Greedy extend for a few steps
        for _ in range(10):
            cur = path[-1]
            candidates = []
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            if not candidates:
                break
            # Pick randomly among candidates for diversity
            nxt = rng.choice(candidates)
            path.append(nxt)
            used.add(nxt)

        if len(path) > len(best_path):
            best_path.clear()
            best_path.extend(path)

        _dfs_backtrack(path, used, deadline, best_path)

    return best_path


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers in [0, 2**N).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN  # 140s effective

    # Get a good initial seed
    seed = _greedy_seed()
    best_path = list(seed)

    # Try Warnsdorff greedy for a potentially better seed
    ws = _warnsdorff_greedy()
    if len(ws) > len(best_path):
        best_path = list(ws)

    # Multi-start DFS with backtracking
    _multi_start_dfs(deadline, best_path, seed)

    # Final validation
    if not _is_valid_snake(best_path):
        # Fallback to the greedy seed
        best_path = _greedy_seed()

    return best_path

# EVOLVE-BLOCK-END
