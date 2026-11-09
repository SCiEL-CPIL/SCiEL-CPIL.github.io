"""Circulant-graph search for K4-free graphs with independence number <= 5."""
import time
import itertools
from functools import lru_cache

_DEADLINE = 145.0  # safety margin before the 150 s wall-clock kill
_START = None


def _elapsed():
    return time.time() - _START


def _independence_number(adj_masks, n):
    """Exact maximum independent set size via bitmask DP (n <= ~40)."""
    # adj_masks[i] is a bitmask of neighbours of i
    # Max independent set = max over subsets S with no internal edges of |S|
    # Use the standard complement-graph clique DP (Bron–Kerbosch with pivot) on complement.
    # Build complement adjacency masks.
    full = (1 << n) - 1
    comp = [full ^ adj_masks[i] ^ (1 << i) for i in range(n)]
    best = [0]

    def bronk(R, P, X):
        # R: current independent set size, P: candidates, X: excluded
        if P == 0:
            if R > best[0]:
                best[0] = R
            return
        # pivot
        u = (P | X).bit_length() - 1
        cand = P & ~comp[u]
        while cand:
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            bronk(R + 1, P & comp[v], X & comp[v])
            P ^= lb
            X |= lb
            cand ^= lb
            if _elapsed() > _DEADLINE:
                return
    bronk(0, full, 0)
    return best[0]


def _has_k4(adj_masks, n):
    """Return True if the graph contains a K4."""
    # For each pair (u,v) that is an edge, look for a common neighbour w that is
    # adjacent to both, then check for a fourth vertex adjacent to all three.
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj_masks[u] >> v) & 1:
                continue
            common = adj_masks[u] & adj_masks[v]
            # remove u and v themselves
            common &= ~(1 << u) & ~(1 << v)
            cw = common
            while cw:
                lb = cw & (-cw)
                w = lb.bit_length() - 1
                # need a vertex x adjacent to u, v, w
                common3 = adj_masks[u] & adj_masks[v] & adj_masks[w]
                common3 &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                if common3:
                    return True
                cw ^= lb
    return False


def _circulant_adj_masks(n, S):
    """Build adjacency masks for a circulant graph with connection set S."""
    masks = [0] * n
    for s in S:
        s %= n
        if s == 0:
            continue
        for i in range(n):
            a = (i + s) % n
            b = (i - s) % n
            masks[i] |= (1 << a) | (1 << b)
    return masks


def _check(n, S):
    masks = _circulant_adj_masks(n, S)
    if _has_k4(masks, n):
        return False, None
    if _independence_number(masks, n) > 5:
        return False, None
    return True, masks


def _build_edges(n, S):
    edges = []
    for s in S:
        s %= n
        if s == 0:
            continue
        seen = set()
        for i in range(n):
            a = (i + s) % n
            if i < a and (i, a) not in seen:
                edges.append([i, a])
                seen.add((i, a))
    return edges


def construct():
    global _START
    _START = time.time()

    best_n = 0
    best_edges = None

    # Search n from high to low. For each n, enumerate connection sets.
    # We bias towards denser connection sets (to keep alpha small) but
    # structured to avoid K4.
    for n in range(36, 16, -1):
        if _elapsed() > _DEADLINE - 5:
            break
        half = n // 2
        # Candidate connection sets: try subsets of {1..half}.
        # To keep alpha small we want enough edges; to avoid K4 we want
        # structure. We'll iterate by increasing size and try to find valid.
        # Use a heuristic order: prefer sets that include small steps.
        # We'll do a randomized + systematic hybrid.
        found = False
        # Systematic: try all subsets up to a manageable size via combinatorics,
        # but that's 2^half which is too big for half=18. Instead, use a
        # local-search / greedy approach to find a connection set.
        # Strategy: start from a random connection set, then hill-climb.
        import random
        rng = random.Random(12345 + n)
        half = n // 2
        universe = list(range(1, half + 1))
        # Try several random starts
        for attempt in range(200):
            if _elapsed() > _DEADLINE - 5:
                break
            # random connection set with moderate size
            k = rng.randint(half // 2, half)
            S = set(rng.sample(universe, k))
            ok, masks = _check(n, S)
            if ok:
                best_n = n
                best_edges = _build_edges(n, S)
                found = True
                break
            # If not ok, do a few local-search steps
            for _step in range(30):
                if _elapsed() > _DEADLINE - 5:
                    break
                # pick a random element to toggle
                if S:
                    elem = rng.choice(sorted(S))
                    S.discard(elem)
                else:
                    elem = rng.choice(universe)
                    S.add(elem)
                # sometimes add instead
                if rng.random() < 0.5:
                    if len(S) < half:
                        cand = rng.choice([x for x in universe if x not in S])
                        S.add(cand)
                ok, masks = _check(n, S)
                if ok:
                    best_n = n
                    best_edges = _build_edges(n, S)
                    found = True
                    break
        if found:
            break

    if best_n == 0:
        # Fallback: the Paley graph on 17 vertices
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

    return {'n': best_n, 'edges': best_edges}
