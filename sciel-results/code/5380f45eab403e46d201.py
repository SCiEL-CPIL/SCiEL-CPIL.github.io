"""
Improved Snake-in-the-Box solver for Q_9.
Combines Warnsdorff's rule with backtracking, randomized restarts,
bounded dead-end memoization, and a final polish/extension pass.
"""
import time
import random
import sys

N = 9
NUM_VERTS = 1 << N

# Precompute neighbors for each vertex
_NEIGHBORS = [[] for _ in range(NUM_VERTS)]
for v in range(NUM_VERTS):
    for b in range(N):
        _NEIGHBORS[v].append(v ^ (1 << b))

# Precompute bit_count lookup for 9-bit values (0 to 511)
_BIT_COUNT = [0] * NUM_VERTS
for i in range(1, NUM_VERTS):
    _BIT_COUNT[i] = _BIT_COUNT[i >> 1] + (i & 1)

# Precompute adjacency bitmask: for each vertex, a 512-bit integer whose set bits
# are its 9 neighbors. This lets us test "is v adjacent to any path vertex" quickly
# via bitwise operations on a Python int representing the used set.
_ADJ_MASK = [0] * NUM_VERTS
for v in range(NUM_VERTS):
    m = 0
    for nb in _NEIGHBORS[v]:
        m |= (1 << nb)
    _ADJ_MASK[v] = m

# Parity (bipartition) of each vertex: parity = popcount(v) % 2
_PARITY = [0] * NUM_VERTS
for v in range(NUM_VERTS):
    _PARITY[v] = _BIT_COUNT[v] & 1


def _validate_snake(path):
    """Validate that path is a valid induced path in Q_N. Returns True if valid."""
    if not path:
        return True
    if len(set(path)) != len(path):
        return False
    for i in range(len(path) - 1):
        if _BIT_COUNT[path[i] ^ path[i + 1]] != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if _BIT_COUNT[path[i] ^ path[j]] == 1:
                return False
    return True


def _is_safe_mask(v, used_mask, last):
    """
    Fast safety check using bitmasks.
    v is safe iff v not in used_mask, and v is not adjacent to any vertex in
    the path except the last one (which is its intended predecessor).
    We check: (adjacency of v to all used) has only the 'last' bit set (or none).
    """
    if (used_mask >> v) & 1:
        return False
    # v's neighbors that are already in the path
    adj_to_used = _ADJ_MASK[v] & used_mask
    # The only allowed adjacent used vertex is 'last'
    allowed = (1 << last)
    if adj_to_used & ~allowed:
        return False
    return True


def _greedy_extend(path, used_mask, rng, time_limit):
    """
    Greedily extend the path using randomized neighbor order.
    Mutates path and used_mask in place. Returns final length.
    """
    while time.time() < time_limit:
        cur = path[-1]
        neighbors = list(_NEIGHBORS[cur])
        rng.shuffle(neighbors)
        extended = False
        for nb in neighbors:
            if _is_safe_mask(nb, used_mask, cur):
                path.append(nb)
                used_mask |= (1 << nb)
                extended = True
                break
        if not extended:
            break
    return len(path)


def _search_warnsdorff(start, time_limit, rng, memo):
    """
    Search using Warnsdorff's rule with backtracking and bounded dead-end memoization.
    Returns the best path found within time_limit.
    """
    deadline = time.time() + time_limit
    best_path = [start]
    best_len = 1

    path = [start]
    used_mask = 1 << start

    # Memoization of dead-end states: (current_vertex, used_mask) -> True (dead end)
    # Bounded to avoid unbounded memory growth.
    MEMO_MAX = 200000

    def _state_key(cur, um):
        return (cur, um)

    def backtrack():
        nonlocal best_path, best_len

        if time.time() > deadline:
            return False

        cur = path[-1]

        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

        # Generate candidate moves
        candidates = []
        for nb in _NEIGHBORS[cur]:
            if not _is_safe_mask(nb, used_mask, cur):
                continue
            # Warnsdorff score: number of safe moves from nb
            score = 0
            for nb2 in _NEIGHBORS[nb]:
                # nb2 must be safe to add after nb
                # After adding nb, the "last" for nb2 would be nb
                if (used_mask | (1 << nb)) >> nb2 & 1:
                    continue
                adj_to = _ADJ_MASK[nb2] & (used_mask | (1 << nb))
                allowed = (1 << nb)
                if not (adj_to & ~allowed):
                    score += 1
            # Parity bias: prefer moving to the less-used parity side (heuristic)
            # We keep it simple: just use score.
            candidates.append((score, rng.random(), nb))

        if not candidates:
            # Dead end: memoize this state
            if len(memo) < MEMO_MAX:
                memo[_state_key(cur, used_mask)] = True
            return False

        # Sort by (score ascending, random tie-breaker)
        candidates.sort(key=lambda x: (x[0], x[1]))

        for (score, rand, nb) in candidates:
            if time.time() > deadline:
                return False

            # Make the move
            path.append(nb)
            used_mask |= (1 << nb)

            # Recurse
            if backtrack():
                return True  # timeout or success, unwind

            # Undo move
            path.pop()
            used_mask &= ~(1 << nb)

        # All candidates exhausted: this state is a dead end
        if len(memo) < MEMO_MAX:
            memo[_state_key(cur, used_mask)] = True
        return False

    # Check memo before starting
    if _state_key(start, used_mask) in memo:
        return best_path

    backtrack()
    return best_path


def construct():
    """
    Construct a long induced path (snake) in the hypercube Q_9.
    Returns a list of distinct integers in [0, 2**N) representing the vertices
    of an induced path in order.
    """
    start_time = time.time()
    deadline = start_time + 112  # Safety margin before 120s limit

    best_path = [0]
    best_len = 1
    rng = random.Random(42)

    # Shared memoization across searches (bounded)
    memo = {}

    # Starting points: mix of structured and random
    starts = [0, 1, 255, 128, 511, 384, 64, 32, 16, 8, 4, 2, 3, 254, 129]
    rng.shuffle(starts)

    # Phase 1: Multiple Warnsdorff searches with backtracking
    num_searches = 25
    for i in range(num_searches):
        if time.time() > deadline - 8:  # Reserve 8s for polish
            break

        if i < len(starts):
            start = starts[i]
        else:
            start = rng.randint(0, NUM_VERTS - 1)

        # Check memo: if this start state is known dead-end, skip
        if (start, 1 << start) in memo:
            continue

        search_rng = random.Random(rng.randint(0, 2**32))
        time_per_search = min(4.0, (deadline - 8 - time.time()) / max(1, num_searches - i))

        path = _search_warnsdorff(start, time_per_search, search_rng, memo)

        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 2: Polish — try to extend the best path with randomized greedy
    if time.time() < deadline - 2:
        for attempt in range(30):
            if time.time() > deadline - 2:
                break
            cur_path = list(best_path)
            cur_used = 0
            for v in cur_path:
                cur_used |= (1 << v)
            polish_rng = random.Random(rng.randint(0, 2**32))
            _greedy_extend(cur_path, cur_used, polish_rng, deadline - 2)
            if len(cur_path) > best_len:
                best_len = len(cur_path)
                best_path = cur_path

    # Phase 3: Final validation and fallback
    if not _validate_snake(best_path):
        # Fallback to a simple greedy seed
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if any(_BIT_COUNT[v ^ p] == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        best_path = path

    return best_path
