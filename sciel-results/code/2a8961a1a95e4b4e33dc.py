"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a Sidon set (all unordered sums a+b, a<=b, distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal: adding any missing element breaks Sidon."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            sums.add(a + selected[j])
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # New sums introduced by x: 2x and x+a for a in selected
        if (2 * x) in sums:
            continue
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok:
            return False
    return True


def _addable(x, selected, sums):
    """Can x be added to selected (with current sum set) without conflict?"""
    if (2 * x) in sums:
        return False
    for a in selected:
        if (x + a) in sums:
            return False
    return True


def _build_conflict_graph(N):
    """
    Build conflict graph for Sidon set in [1,N].
    Two vertices u,v (u!=v) are connected if they cannot coexist in a Sidon set.
    Also, a vertex u conflicts with itself if 2u would create a duplicate sum
    with some other pair - but for the graph we only consider distinct pairs.
    
    For distinct u,v (u<v), they conflict if there exist a,b in [1,N] with a<=b
    such that u+v = a+b and {a,b} != {u,v}.
    """
    # Precompute all pair sums: sum -> list of pairs (a,b) with a<=b, a+b=sum
    sum_to_pairs = {}
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            s = a + b
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((a, b))
    
    # Build adjacency: conflict[u] = set of v != u that conflict with u
    conflict = [set() for _ in range(N + 1)]
    
    for s, pairs in sum_to_pairs.items():
        if len(pairs) > 1:
            # All elements in any pair in this sum group conflict with elements in other pairs
            for i, (a, b) in enumerate(pairs):
                for j, (c, d) in enumerate(pairs):
                    if i == j:
                        continue
                    # a and c conflict, a and d conflict, b and c conflict, b and d conflict
                    # But skip if same element
                    for x in (a, b):
                        for y in (c, d):
                            if x != y:
                                conflict[x].add(y)
    
    return conflict


def _greedy_conflict_elimination(conflict, N, deadline, seed=None):
    """
    Greedy vertex elimination on conflict graph to find a small maximal independent set.
    Iteratively pick the vertex with the highest degree (most conflicts) and add it
    to our Sidon set, then remove it and all its neighbors from consideration.
    """
    rng = random.Random(seed) if seed is not None else None
    remaining = set(range(1, N + 1))
    selected = []
    # degree[x] = number of remaining vertices that conflict with x
    degree = {x: len(conflict[x] & remaining) for x in remaining}
    
    while remaining:
        if time.time() > deadline:
            break
        # Pick vertex with highest degree; break ties randomly
        max_deg = max(degree.values())
        candidates = [x for x in remaining if degree[x] == max_deg]
        x = rng.choice(candidates)
        selected.append(x)
        # Remove x and all its neighbors from remaining
        neighbors = conflict[x] & remaining
        remaining.discard(x)
        remaining -= neighbors
        # Update degrees for remaining vertices
        for y in remaining:
            # y loses degree for each removed neighbor
            removed_neighbors = (neighbors | {x}) & conflict[y]
            degree[y] -= len(removed_neighbors)
        # Remove degrees for non-remaining
        for y in list(degree.keys()):
            if y not in remaining:
                del degree[y]
    
    return selected


def _greedy_conflict_elimination_weighted(conflict, N, deadline, seed=None):
    """
    Weighted greedy: pick vertex that maximizes (degree + coverage of low-degree vertices).
    This tends to block more of the remaining candidates.
    """
    rng = random.Random(seed) if seed is not None else None
    remaining = set(range(1, N + 1))
    selected = []
    degree = {x: len(conflict[x] & remaining) for x in remaining}
    
    while remaining:
        if time.time() > deadline:
            break
        best_x = None
        best_score = -1
        for x in remaining:
            neighbors = conflict[x] & remaining
            # Score: primary is degree, secondary is how many low-degree vertices it blocks
            blocked_low = sum(1 for y in neighbors if degree[y] < max(degree.values()) / 2)
            score = degree[x] * 1000 + blocked_low
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        neighbors = conflict[best_x] & remaining
        remaining.discard(best_x)
        remaining -= neighbors
        for y in remaining:
            removed_neighbors = (neighbors | {best_x}) & conflict[y]
            degree[y] -= len(removed_neighbors)
        for y in list(degree.keys()):
            if y not in remaining:
                del degree[y]
    
    return selected


def _coverage_greedy_fill(selected, N, deadline, seed=None):
    """
    Complete a Sidon set to inclusion-maximal by greedily adding elements that
    cover (block) the most remaining candidates. Minimizes added count.
    """
    rng = random.Random(seed) if seed is not None else None
    selected = list(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    sel_set = set(selected)
    remaining = [x for x in range(1, N + 1) if x not in sel_set]
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sums)]
        if not valid:
            break
        # Pick valid x covering the most remaining candidates
        best_x = None
        best_cov = -1
        for x in valid:
            new_sums = sums | {(2 * x)} | {(x + a) for a in selected}
            cov = 0
            for y in remaining:
                if y == x:
                    continue
                if (2 * y) in new_sums:
                    cov += 1
                    continue
                blocked = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (x + y) in new_sums:
                    blocked = True
                if blocked:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        sel_set.add(best_x)
        remaining.remove(best_x)
        sums.add(2 * best_x)
        for a in selected[:-1]:
            sums.add(best_x + a)
    return selected


def _local_search_remove(selected, N, deadline):
    """
    Local search: repeatedly remove an element while the set stays Sidon AND
    maximal. Shrinks cardinality while preserving inclusion-maximality.
    """
    best = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removals; prefer removing elements that leave set maximal
        for x in list(best):
            if time.time() > deadline:
                break
            cand = [v for v in best if v != x]
            if not _is_valid_sidon(cand):
                continue
            if _is_maximal(cand, N):
                best = cand
                improved = True
                break
    return best


def _random_greedy_seed(N, deadline, seed):
    """Build a Sidon seed from a randomized order (max-blocking flavor)."""
    rng = random.Random(seed)
    remaining = list(range(1, N + 1))
    selected = []
    sums = set()
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sums)]
        if not valid:
            break
        # Pick weighted by coverage, with randomness
        best_x = None
        best_cov = -1
        top = []
        for x in valid:
            new_sums = sums | {(2 * x)} | {(x + a) for a in selected}
            cov = 0
            for y in valid:
                if y == x:
                    continue
                if (2 * y) in new_sums:
                    cov += 1
                    continue
                blocked = False
                for a in selected:
                    if (y + a) in new_sums:
                        blocked = True
                        break
                if not blocked and (x + y) in new_sums:
                    blocked = True
                if blocked:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                top = [x]
            elif cov == best_cov:
                top.append(x)
        x = rng.choice(top) if top else rng.choice(valid)
        selected.append(x)
        remaining.remove(x)
        sums.add(2 * x)
        for a in selected[:-1]:
            sums.add(x + a)
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill

    best_result = None
    best_size = float('inf')

    def consider(selected):
        nonlocal best_result, best_size
        if not selected:
            return
        if not _is_valid_sidon(selected):
            return
        if not _is_maximal(selected, N):
            return
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)

    # Build conflict graph once
    conflict = _build_conflict_graph(N)

    # Strategy A: Greedy conflict elimination -> complete to maximal -> local search shrink
    for seed in range(20):
        if time.time() > deadline:
            break
        partial = _greedy_conflict_elimination(conflict, N, deadline, seed)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Strategy B: Weighted greedy conflict elimination
    for seed in range(15):
        if time.time() > deadline:
            break
        partial = _greedy_conflict_elimination_weighted(conflict, N, deadline, seed)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Strategy C: Randomized seeds -> complete -> shrink
    for seed in range(10):
        if time.time() > deadline:
            break
        partial = _random_greedy_seed(N, deadline, 1000 + seed)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Fallback: ascending greedy, then make maximal
    if best_result is None:
        selected, sums = [], set()
        for x in range(1, N + 1):
            if _addable(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        maximal = _coverage_greedy_fill(selected, N, deadline, 0)
        consider(maximal)
        if best_result is None:
            return maximal

    return best_result

# EVOLVE-BLOCK-END
