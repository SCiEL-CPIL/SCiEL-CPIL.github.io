import time
import random

N = 40
TARGET_DIFFS = set(range(1, 39))  # 1 to 38
NUM_TARGETS = len(TARGET_DIFFS)


def _cover_count(S):
    """Count how many target differences are covered by S."""
    S = sorted(S)
    covered = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = S[j] - S[i]
            if 1 <= d <= 38:
                covered.add(d)
    return len(covered)


def _is_valid_basis(S):
    return _cover_count(S) == NUM_TARGETS


def _find_basis(target_size, time_limit=5.0, seed=None):
    """Find a subset S of {1..39} of size target_size covering all diffs 1..38."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    best_S = None
    best_score = 0

    # Multiple restarts
    while time.time() - start < time_limit:
        # Random init
        S = set(random.sample(range(1, 40), target_size))
        score = _cover_count(S)

        # Local Search with first-improvement
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            S_list = sorted(S)
            for i in range(len(S_list)):
                if time.time() - start >= time_limit:
                    break
                old = S_list[i]
                # Try all possible new values
                for nv in range(1, 40):
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    new_score = _cover_count(S)
                    if new_score > score:
                        score = new_score
                        improved = True
                        if score == NUM_TARGETS:
                            return sorted(S)
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if score == NUM_TARGETS:
                    return sorted(S)

        if score > best_score:
            best_score = score
            best_S = sorted(S)

    if best_S and _is_valid_basis(best_S):
        return best_S
    return None


def _greedy_basis(target_size, time_limit=3.0):
    """Greedy construction: add elements one by one to maximize coverage."""
    start = time.time()
    S = []
    for _ in range(target_size):
        if time.time() - start > time_limit:
            break
        best_next = None
        best_cov = 0
        for x in range(1, 40):
            if x in S:
                continue
            temp_S = S + [x]
            cov = _cover_count(temp_S)
            if cov > best_cov:
                best_cov = cov
                best_next = x
        if best_next is None:
            break
        S.append(best_next)
    S = sorted(set(S))
    if _is_valid_basis(S):
        return S
    return None


def _construct_graph(S):
    """Build the fan graph from basis S and return edges and cycles."""
    S_set = set(S)
    edges = []
    edges_set = set()

    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))

    # Path 1-39
    for i in range(1, N - 1):
        add_edge(i, i + 1)
    # Spokes 0-s
    for s in S:
        add_edge(0, s)

    cycles = []
    for L in range(3, N + 1):
        diff = L - 2
        found = False
        for u in sorted(S_set):
            v = u + diff
            if v in S_set:
                cycle = [0] + list(range(u, v + 1))
                # Verify all edges exist
                ok = True
                for i in range(len(cycle) - 1):
                    a, b = cycle[i], cycle[i + 1]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                        break
                if ok:
                    a, b = cycle[-1], cycle[0]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                if ok:
                    cycles.append(cycle)
                    found = True
                    break
        if not found:
            return None  # Invalid basis

    return {'edges': edges, 'cycles': cycles}


def construct():
    deadline = time.time() + 85.0  # Safety margin: kill at 90s, leave 5s buffer

    # Try to find smallest basis, starting from size 9
    S = None
    for size in [9, 10, 11]:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break

        # Allocate time: more for smaller sizes (harder)
        t_limit = min(15.0, remaining * 0.4) if size == 9 else min(8.0, remaining * 0.3)

        # Try greedy first (fast)
        S = _greedy_basis(size, time_limit=min(2.0, t_limit * 0.2))
        if S:
            break

        # Try local search
        S = _find_basis(size, time_limit=t_limit)
        if S:
            break

        # If time is running out, try one more with remaining time
        remaining = deadline - time.time()
        if remaining > 3.0 and size == 11:
            S = _find_basis(11, time_limit=min(remaining - 2.0, 10.0))
            if S:
                break

    # If no basis found, use a known valid size-12 basis as fallback
    if S is None:
        # Fallback: use a precomputed valid basis
        # A valid basis of size 12: {1, 2, 3, 5, 8, 13, 21, 34, 6, 10, 17, 28}
        # Let's verify or just use greedy with more time
        S = _greedy_basis(12, time_limit=5.0)
        if S is None:
            S = _find_basis(12, time_limit=5.0)
        if S is None:
            # Absolute last resort: star + path (valid but many edges)
            edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
            cycles = [list(range(k)) for k in range(3, N + 1)]
            return {'edges': edges, 'cycles': cycles}

    # Construct graph
    result = _construct_graph(S)
    if result is None:
        # Basis was invalid, fallback
        edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
        cycles = [list(range(k)) for k in range(3, N + 1)]
        return {'edges': edges, 'cycles': cycles}

    return result
