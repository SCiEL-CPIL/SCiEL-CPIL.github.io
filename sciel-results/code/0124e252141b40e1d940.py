"""Snake-in-the-box in Q_9: randomized backtracking with heuristic ordering and multiple restarts."""
import time
import random
import sys

N = 9
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0  # return at least this many seconds before the hard kill

# Precompute bit masks for convenience
MASKS = [1 << b for b in range(N)]
ALL_BITS_MASK = (1 << N) - 1


def _is_valid_extension(v, path, used):
    """Check if vertex v can be appended to path (induced path constraint)."""
    if v in used:
        return False
    # v must differ from all vertices except the last by at least 2 bits
    for i in range(len(path) - 1):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True


def _greedy_extend(path, used, rng, time_budget, deadline):
    """
    Extend the path greedily with randomized tie-breaking.
    Returns the extended path and updated used set.
    """
    while True:
        if time.time() > deadline:
            break
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ MASKS[b]
            if v in used:
                continue
            if _is_valid_extension(v, path, used):
                candidates.append(v)
        if not candidates:
            break
        # Randomized selection: pick a random candidate (or weighted by some heuristic)
        # Use a simple random pick for diversity
        nxt = rng.choice(candidates)
        path.append(nxt)
        used.add(nxt)
    return path, used


def _backtrack_extend(path, used, rng, deadline, max_backtrack_depth=3):
    """
    Try to extend the path. If stuck, backtrack up to max_backtrack_depth steps
    and try alternative branches. Returns (path, used, success).
    """
    while time.time() < deadline:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ MASKS[b]
            if v in used:
                continue
            if _is_valid_extension(v, path, used):
                candidates.append(v)
        if candidates:
            # Try candidates in random order
            rng.shuffle(candidates)
            for v in candidates:
                path.append(v)
                used.add(v)
                # Try to extend further
                new_path, new_used, ok = _backtrack_extend(path, used, rng, deadline, max_backtrack_depth)
                if len(new_path) > len(path) - 1:
                    return new_path, new_used, True
                # Undo
                path.pop()
                used.discard(v)
            # No candidate worked; backtrack
            if len(path) > 1:
                path.pop()
                used.discard(path[-1])
            else:
                break
        else:
            # Stuck; backtrack
            if len(path) > 1:
                path.pop()
                used.discard(path[-1])
            else:
                break
    return path, used, False


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Uses randomized backtracking with multiple restarts to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    best_path = None
    best_len = 0

    # Use a fixed seed for reproducibility within a single run, but vary across restarts
    rng = random.Random()

    # First, get a baseline greedy snake
    path, used = [0], {0}
    path, used = _greedy_extend(path, used, rng, 10, deadline)
    if len(path) > best_len:
        best_len = len(path)
        best_path = path[:]

    # Now do randomized restarts with backtracking
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Start from a random vertex
        start_v = rng.randint(0, (1 << N) - 1)
        path, used = [start_v], {start_v}

        # Greedy extension with randomization
        path, used = _greedy_extend(path, used, rng, 5, deadline)

        # If stuck, try backtracking to find a longer path
        if time.time() < deadline:
            path, used, _ = _backtrack_extend(path, used, rng, deadline, max_backtrack_depth=2)

        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

        # Safety check
        if time.time() > deadline:
            break

    # Final polish: try to extend the best path found
    if best_path is not None and time.time() < deadline:
        used = set(best_path)
        path = best_path[:]
        path, used = _greedy_extend(path, used, rng, 5, deadline)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]

    if best_path is None:
        # Fallback: return the simple greedy snake
        path, used = [0], {0}
        path, used = _greedy_extend(path, used, rng, 5, deadline)
        best_path = path

    return best_path
