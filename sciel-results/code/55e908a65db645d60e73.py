import time
import random
from collections import Counter

_DEADLINE = 50.0
_N = 256

def _build_conflict_graph():
    """Build adjacency sets: conflict[a] = set of b != a such that (a*b)%(a+b)==0."""
    conflict = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    return conflict

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_build(conflict, order):
    """Greedy: add x if it doesn't conflict with any already-chosen element."""
    S = set()
    for x in order:
        if not (S & conflict[x]):
            S.add(x)
    return S

def _conflict_counts_in_S(conflict, S):
    """Return dict {v: count of neighbors of v in S} for v not in S."""
    counts = {}
    for u in S:
        for v in conflict[u]:
            if v not in S:
                counts[v] = counts.get(v, 0) + 1
    return counts

def _local_search(conflict, S, time_limit, seed):
    """
    Local search on an independent set.
    Moves: add, remove, swap, double-swap.
    Objective: maximize |S|.
    """
    rng = random.Random(seed)
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start = time.time()

    all_vertices = list(range(1, _N + 1))

    # Cache conflict counts
    conf_counts = _conflict_counts_in_S(conflict, S)

    def refresh_conf_counts():
        nonlocal conf_counts
        conf_counts = _conflict_counts_in_S(conflict, S)

    def can_add(v):
        return v not in S and not (S & conflict[v])

    while time.time() - start < time_limit:
        improved = False

        # 1. Try to add any vertex that has no conflict with S
        #    Use cached counts for efficiency
        candidates_to_add = [v for v in all_vertices if v not in S and conf_counts.get(v, 0) == 0]
        if candidates_to_add:
            # Add the one with fewest total conflicts (to leave room)
            v = min(candidates_to_add, key=lambda x: len(conflict[x]))
            S.add(v)
            improved = True
            refresh_conf_counts()
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

        # 2. Try swaps: remove u from S, add v not in S
        if not improved:
            swaps_tried = 0
            for _ in range(300):
                if time.time() - start > time_limit:
                    break
                if not S:
                    break
                u = rng.choice(list(S))
                v = rng.choice(all_vertices)
                if v in S:
                    continue
                # Check if v conflicts with S \ {u}
                new_S = S - {u}
                if not (new_S & conflict[v]):
                    S = new_S | {v}
                    improved = True
                    refresh_conf_counts()
                    break
                swaps_tried += 1

        # 3. Try double-swap: remove two from S, add two not in S
        if not improved and len(S) > 3:
            for _ in range(100):
                if time.time() - start > time_limit:
                    break
                u1, u2 = rng.sample(list(S), 2)
                v1, v2 = rng.sample(all_vertices, 2)
                if v1 in S or v2 in S:
                    continue
                new_S = S - {u1, u2}
                if not (new_S & conflict[v1]) and not (new_S & conflict[v2]) and (v1 * v2) % (v1 + v2) != 0:
                    S = new_S | {v1, v2}
                    improved = True
                    refresh_conf_counts()
                    break

        # 4. If no improvement, do a small random perturbation
        if not improved:
            # Remove 1-2 random elements
            if len(S) >= 2:
                S.discard(rng.choice(list(S)))
                if S:
                    S.discard(rng.choice(list(S)))
            # Add random elements if possible
            candidates = [v for v in all_vertices if v not in S and not (S & conflict[v])]
            if candidates:
                rng.shuffle(candidates)
                for v in candidates[:2]:
                    if not (S & conflict[v]):
                        S.add(v)
            refresh_conf_counts()

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    return best_S

def _multi_start_search(conflict, total_time):
    """Run multiple rounds with diverse initializations."""
    start = time.time()
    best_S = set()
    best_size = 0
    seed = 42

    all_vertices = list(range(1, _N + 1))
    degrees = {v: len(conflict[v]) for v in all_vertices}

    while time.time() - start < total_time:
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining < 0.5:
            break

        # Strategy 1: Randomized greedy
        order = list(range(1, _N + 1))
        rng = random.Random(seed)
        rng.shuffle(order)
        S_init = _greedy_build(conflict, order)

        # Strategy 2: Degree-ordered greedy (prefer low-degree)
        deg_order = sorted(all_vertices, key=lambda v: (degrees[v], rng.random()))
        S_deg = _greedy_build(conflict, deg_order)

        # Strategy 3: Random weighted order
        weights = {v: rng.random() for v in all_vertices}
        w_order = sorted(all_vertices, key=lambda v: weights[v])
        S_w = _greedy_build(conflict, w_order)

        # Pick the best initialization
        candidates = [S_init, S_deg, S_w]
        best_init = max(candidates, key=len)

        # Local search on the best initialization
        ls_time = min(remaining * 0.35, 4.0)
        S_ls = _local_search(conflict, best_init, ls_time, seed)
        seed += 1

        if len(S_ls) > best_size:
            best_size = len(S_ls)
            best_S = set(S_ls)

        # If we still have time, try local search on the second-best init
        elapsed = time.time() - start
        remaining = total_time - elapsed
        if remaining > 2.0:
            second_best = max([c for c in candidates if c != best_init], key=len) if len(candidates) > 1 else best_init
            ls_time2 = min(remaining * 0.3, 3.0)
            S_ls2 = _local_search(conflict, second_best, ls_time2, seed + 1000)
            seed += 1
            if len(S_ls2) > best_size:
                best_size = len(S_ls2)
                best_S = set(S_ls2)

    return best_S

def _try_add_to_set(conflict, S):
    """Try to add more elements to S by finding vertices with no conflict."""
    added = True
    while added:
        added = False
        for v in range(1, _N + 1):
            if v not in S and not (S & conflict[v]):
                S.add(v)
                added = True
    return S

def _polish(conflict, S, time_limit):
    """Final polish: try to find a larger independent set via targeted swaps."""
    rng = random.Random(12345)
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    start = time.time()

    all_vertices = list(range(1, _N + 1))

    while time.time() - start < time_limit:
        improved = False

        # Try to add
        for v in all_vertices:
            if v not in S and not (S & conflict[v]):
                S.add(v)
                improved = True

        # Try single swaps: for each u in S, try replacing with v
        if not improved:
            for u in list(S):
                for v in all_vertices:
                    if v in S:
                        continue
                    new_S = S - {u}
                    if not (new_S & conflict[v]):
                        S = new_S | {v}
                        improved = True
                        break
                if improved:
                    break

        # Try double swaps
        if not improved and len(S) >= 4:
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    u1, u2 = S_list[i], S_list[j]
                    new_S = S - {u1, u2}
                    for v1 in all_vertices:
                        if v1 in S:
                            continue
                        if new_S & conflict[v1]:
                            continue
                        for v2 in all_vertices:
                            if v2 in S or v2 == v1:
                                continue
                            if (new_S & conflict[v2]) or (v1 * v2) % (v1 + v2) == 0:
                                continue
                            S = new_S | {v1, v2}
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

        if not improved:
            # Perturbation: remove 2 random, add 2 random if possible
            if len(S) >= 4:
                S.discard(rng.choice(list(S)))
                S.discard(rng.choice(list(S)))
            candidates = [v for v in all_vertices if v not in S and not (S & conflict[v])]
            if candidates:
                rng.shuffle(candidates)
                for v in candidates[:3]:
                    if not (S & conflict[v]):
                        S.add(v)

    return best_S

def construct():
    start = time.time()
    conflict = _build_conflict_graph()

    # Quick baseline: all odds
    S_odd = set(range(1, _N + 1, 2))
    for e in range(2, _N + 1, 2):
        if not (S_odd & conflict[e]):
            S_odd.add(e)
    baseline_size = len(S_odd)

    # Allocate time
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 3.0  # 3s safety margin
    if remaining < 5:
        remaining = 5

    # Run the main search
    best_S = _multi_start_search(conflict, remaining)

    # Final greedy fill
    best_S = _try_add_to_set(conflict, best_S)

    # Polish pass
    elapsed = time.time() - start
    remaining = _DEADLINE - elapsed - 1.0
    if remaining > 1.0:
        best_S = _polish(conflict, best_S, min(remaining, 8.0))
        best_S = _try_add_to_set(conflict, best_S)

    # Ensure baseline is at least as good
    if len(best_S) < baseline_size:
        best_S = S_odd
        best_S = _try_add_to_set(conflict, best_S)

    result = sorted(best_S)
    if not result:
        return [1]
    return result
