import time
import random

N = 9
MAX_V = 1 << N
MAX_TIME = 115.0  # Safety margin before 150s kill

# Precompute bit counts for all possible XOR results
_BIT_COUNT = [i.bit_count() for i in range(MAX_V)]

# Precompute neighbors for all vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * MAX_V
for v in range(MAX_V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask


def _warnsdorff_dfs(seed_offset=0, target_len=None, deadline=None):
    """
    Pure recursive DFS with Warnsdorff's heuristic (fewest remaining moves first).
    Uses bitmask for fast forbidden-vertex checking.
    Returns the longest path found.
    """
    rng = random.Random(seed_offset)
    best_path = [0]
    best_len = 1
    aborted = False
    node_count = 0

    def dfs(current, forbidden, path, depth):
        nonlocal best_path, best_len, aborted, node_count

        if aborted:
            return False

        if deadline is not None and time.time() > deadline:
            aborted = True
            return False

        node_count += 1

        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

        # If we reached target length, we can stop
        if target_len is not None and len(path) >= target_len:
            return True

        # Get valid neighbors
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                valid_neighbors.append(v)

        if not valid_neighbors:
            return False

        # Warnsdorff: score each neighbor by how many valid moves it has
        scored = []
        for v in valid_neighbors:
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~new_forbidden
            count = valid_mask.bit_count()
            scored.append((count, v))

        # Sort by count ascending (Warnsdorff: fewest options first)
        # Add small random perturbation for tie-breaking diversity
        for i in range(len(scored)):
            scored[i] = (scored[i][0] + rng.random() * 0.001, scored[i][1])
        scored.sort(key=lambda x: x[0])

        for count, v in scored:
            if aborted:
                return False

            if deadline is not None and time.time() > deadline:
                aborted = True
                return False

            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            path.append(v)

            if dfs(v, new_forbidden, path, depth + 1):
                path.pop()
                return True

            path.pop()

        return False

    path = [0]
    forbidden = 1  # vertex 0 is forbidden (used)
    dfs(0, forbidden, path, 0)

    return best_path


def _warnsdorff_iterative_deepening(seed_offset=0, deadline=None):
    """
    Iterative deepening: try to find paths of increasing length.
    Start from a reasonable length and increase.
    """
    rng = random.Random(seed_offset)
    best_path = [0]
    best_len = 1

    # Start with a greedy path to get a baseline
    path = [0]
    forbidden = 1
    current = 0
    while True:
        nxt = None
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << nxt)
        current = nxt

    best_path = path
    best_len = len(best_path)

    # Try to extend beyond the greedy length
    # Use iterative deepening: try target lengths
    target = best_len + 1
    max_target = min(200, best_len + 50)

    while target <= max_target and (deadline is None or time.time() < deadline):
        p = _warnsdorff_dfs(seed_offset, target_len=target, deadline=deadline)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
            target = best_len + 1  # Try to extend further
        else:
            target += 1

    return best_path


def _multi_start_warnsdorff(seed_offset=0, deadline=None):
    """
    Try multiple starting vertices and Warnsdorff DFS from each.
    """
    rng = random.Random(seed_offset)
    best_path = [0]
    best_len = 1

    # Try a few different starting vertices
    start_vertices = [0, (1 << N) - 1]  # 0 and all-ones
    # Add some random starts
    for _ in range(5):
        start_vertices.append(rng.randint(0, MAX_V - 1))

    for sv in start_vertices:
        if deadline is not None and time.time() > deadline:
            break

        # Quick greedy from this start
        path = [sv]
        forbidden = 1 << sv
        current = sv
        while True:
            nxt = None
            for v in _NEIGHBORS[current]:
                if not (forbidden >> v) & 1:
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            forbidden |= _NEIGHBOR_MASK[current]
            forbidden |= (1 << nxt)
            current = nxt

        if len(path) > best_len:
            best_path = path
            best_len = len(path)

    # Now do full Warnsdorff DFS from the best start
    if deadline is None or time.time() < deadline:
        p = _warnsdorff_dfs(seed_offset, deadline=deadline)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)

    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME

    best_path = [0]
    best_len = 1

    # Phase 1: Quick greedy to establish baseline
    path = [0]
    forbidden = 1
    current = 0
    while True:
        nxt = None
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << nxt)
        current = nxt

    best_path = path
    best_len = len(best_path)

    # Phase 2: Warnsdorff DFS with iterative deepening
    if time.time() < deadline - 5:
        p = _warnsdorff_iterative_deepening(0, deadline=deadline)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)

    # Phase 3: More Warnsdorff DFS with different seeds
    seed = 1
    while time.time() < deadline - 2:
        p = _warnsdorff_dfs(seed, deadline=deadline)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        seed += 1

    # Phase 4: Try multi-start if time remains
    if time.time() < deadline - 1:
        p = _multi_start_warnsdorff(seed, deadline=deadline)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)

    return best_path
