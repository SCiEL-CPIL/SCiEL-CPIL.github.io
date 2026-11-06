"""Longest simple path in the pinned G(100, 1.8/100) graph."""
import random
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency as frozensets for speed
_ADJ = [set() for _ in range(N)]
for u, v in EDGES:
    _ADJ[u].add(v)
    _ADJ[v].add(u)
_ADJ = [frozenset(s) for s in _ADJ]

# Precompute a global order for deterministic tie-breaking
_DEG = [len(s) for s in _ADJ]
_ORDER = list(range(N))  # vertex id as tiebreak


def _greedy_extend(path, seen, rng=None):
    """Extend path from both ends greedily, preferring the more constrained end.
    Returns the extended path."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        # Evaluate both ends
        for end in (0, 1):
            if improved:
                break
            if end == 0:
                tail = path[0]
                candidates = [v for v in _ADJ[tail] if v not in seen]
            else:
                tail = path[-1]
                candidates = [v for v in _ADJ[tail] if v not in seen]
            if not candidates:
                continue
            # Pick the candidate with fewest remaining unvisited neighbors (min-degree heuristic)
            # Ties broken by vertex id (or random if rng provided)
            def score(v):
                rem = len(_ADJ[v] - seen)
                return (rem, v)
            if rng is not None and rng.random() < 0.15:
                # Occasionally pick a random candidate to diversify
                v = rng.choice(candidates)
            else:
                v = min(candidates, key=score)
            if end == 0:
                path.insert(0, v)
            else:
                path.append(v)
            seen.add(v)
            improved = True
    return path


def _two_opt(path, seen):
    """Try 2-opt reversals to rearrange the path. Returns a possibly longer path
    after 2-opt + re-extension."""
    n = len(path)
    if n < 4:
        return path
    path = list(path)
    seen = set(path)
    improved = True
    max_iter = 200
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        # Try all 2-opt moves: reverse path[i..j]
        # After reversal, the path must still be a valid simple path.
        # The reversal changes edges: (path[i-1], path[i]) -> (path[i-1], path[j])
        # and (path[j], path[j+1]) -> (path[i], path[j+1])
        for i in range(1, n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                # Check if reversing path[i..j] creates valid edges
                # Left connection: path[i-1] must connect to path[j]
                # Right connection: path[i] must connect to path[j+1] (if j+1 < n)
                if j < n - 1:
                    if path[j] not in _ADJ[path[i - 1]]:
                        continue
                    if path[i] not in _ADJ[path[j + 1]]:
                        continue
                else:
                    # j == n-1: no right connection needed
                    if path[j] not in _ADJ[path[i - 1]]:
                        continue
                # Perform the reversal
                path[i:j + 1] = path[i:j + 1][::-1]
                improved = True
                break
    # Re-extend after 2-opt
    seen = set(path)
    path = _greedy_extend(path, seen)
    return path


def _randomized_greedy(start, rng):
    """Greedy path from a given start with randomization."""
    path = [start]
    seen = {start}
    path = _greedy_extend(path, seen, rng)
    return path


def construct():
    t0 = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    best = []

    # First: deterministic greedy from every vertex (fast, gives a solid baseline)
    for start in range(N):
        if time.time() - t0 > DEADLINE:
            break
        path = _randomized_greedy(start, None)
        if len(path) > len(best):
            best = list(path)

    # Second: 2-opt on the best path found so far
    if best:
        if time.time() - t0 > DEADLINE:
            return best
        improved = _two_opt(best, set(best))
        if len(improved) > len(best):
            best = improved

    # Third: randomized restarts with 2-opt
    rng = random.Random(42)
    # Try different start vertices in random order
    starts = list(range(N))
    rng.shuffle(starts)

    restarts_done = 0
    for start in starts:
        if time.time() - t0 > DEADLINE - 5.0:  # leave margin for final 2-opt
            break
        path = _randomized_greedy(start, rng)
        if len(path) > len(best):
            best = list(path)
        # Apply 2-opt if path is promising
        if len(path) >= len(best) - 2 and len(path) > 3:
            if time.time() - t0 > DEADLINE - 2.0:
                break
            improved = _two_opt(path, set(path))
            if len(improved) > len(best):
                best = list(improved)
        restarts_done += 1
        # If we've found a very long path, reduce effort
        if len(best) >= N - 1:
            break

    # Final: 2-opt on the overall best
    if best and time.time() - t0 < DEADLINE - 1.0:
        improved = _two_opt(best, set(best))
        if len(improved) > len(best):
            best = improved

    # Ensure we return a valid nonempty path
    if not best:
        # Fallback: just return a single vertex (always valid)
        best = [0]

    # Verify validity
    for i in range(len(best) - 1):
        if best[i + 1] not in _ADJ[best[i]]:
            # Should not happen, but truncate to valid prefix
            valid = [best[0]]
            for v in best[1:]:
                if v in _ADJ[valid[-1]]:
                    valid.append(v)
                else:
                    break
            best = valid
            break

    return best
