"""
Construct a large graph with no regular induced subgraph on 6+ vertices.
Strategy: Start from random graphs, use bitmask-based violation detection,
and apply local search (edge flips within violating sets) to repair.
Structural idea: also try graphs built as unions of random perfect matchings
(1-factorization subsets) as an alternative initialization family.
"""
import time
import random
import itertools

DEADLINE = 150
SAFETY_MARGIN = 15
START_TIME = None


def find_violation_fast(adj_masks, n):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None.
    Optimized: precomputed adjacency bitmasks, check size-6 first, early exit on mismatch.
    """
    # Check size-6 subsets first
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok:
            return combo

    # Check larger subsets
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            ok = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    ok = False
                    break
            if ok:
                return combo
    return None


def get_edges_from_masks(adj_masks, n):
    """Convert adjacency masks to edge list."""
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e


def random_graph(n, p, rng):
    """Generate a random graph with given edge probability p."""
    adj_masks = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
    return adj_masks


def random_union_of_matchings(n, rng):
    """
    Generate a graph as a union of a random subset of a random 1-factorization of K_n.
    n must be even. This creates a graph with bounded codegrees, structurally limiting
    regular induced subgraphs.
    """
    if n % 2 != 0:
        return random_graph(n, 0.4, rng)

    # Generate a random permutation of vertices
    verts = list(range(n))
    rng.shuffle(verts)

    # Build a 1-factorization using the standard circle method
    # Fix one vertex, rotate the rest
    fixed = verts[0]
    rotatable = verts[1:]
    m = n // 2  # number of matchings in a 1-factorization of K_n
    matchings = []
    for i in range(m - 1):
        matching = []
        # The fixed vertex pairs with rotatable[i]
        matching.append((fixed, rotatable[i]))
        # Pair up the remaining rotatable vertices
        remaining = [rotatable[(i + 1 + j) % (m - 1)] for j in range(m - 1)]
        # Actually let's use a cleaner circle method
        # For K_{2k}, fix vertex 0, vertices 1..2k-1 on a circle
        # Matching i: (0, i), (1+i, 2k-1-i), (2+i, 2k-2-i), ...
        # Let me redo this properly
        pass

    # Simpler approach: just generate random perfect matchings repeatedly
    adj_masks = [0] * n
    num_matchings = rng.randint(n // 4, n // 2)
    for _ in range(num_matchings):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(0, n, 2):
            u, v = perm[i], perm[i + 1]
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
    return adj_masks


def local_search(adj_masks, n, rng, deadline):
    """Flip edges to eliminate regular induced subgraphs. Prefer flipping edges inside a violating set."""
    cur = adj_masks[:]

    viol = find_violation_fast(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    it = 0
    max_iters = 500
    while it < max_iters:
        if time.time() > deadline:
            break

        viol = find_violation_fast(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = set(viol)
        # Collect candidate edges: prefer edges within the violating set
        internal_edges = []
        internal_nonedges = []
        for idx, u in enumerate(viol):
            for v in viol[idx + 1:]:
                if cur[u] & (1 << v):
                    internal_edges.append((u, v))
                else:
                    internal_nonedges.append((u, v))

        # Strategy: flip an edge in the violating set to break regularity
        if internal_edges and internal_nonedges:
            if rng.random() < 0.5:
                u, v = rng.choice(internal_edges)
            else:
                u, v = rng.choice(internal_nonedges)
        elif internal_edges:
            u, v = rng.choice(internal_edges)
        elif internal_nonedges:
            u, v = rng.choice(internal_nonedges)
        else:
            # Pick a random edge
            u = rng.randrange(n)
            v = rng.randrange(n)
            while v == u:
                v = rng.randrange(n)
            if u > v:
                u, v = v, u

        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1

    viol = find_violation_fast(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(42)

    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # More attempts for larger n
        if n >= 21:
            max_attempts = 20
        elif n >= 19:
            max_attempts = 15
        else:
            max_attempts = 20

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Alternate between random graph and union-of-matchings initialization
            if attempt % 3 == 0:
                # Union of random perfect matchings (structural family)
                adj_masks = random_union_of_matchings(n, rng)
            else:
                # Standard random graph with varied density
                p_choices = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                p = p_choices[rng.randrange(len(p_choices))]
                adj_masks = random_graph(n, p, rng)

            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = get_edges_from_masks(adj_masks, n)
                    best_valid = True
                break

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            continue

    if not best_valid or best_n < 5:
        # Fallback: return a small valid graph
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
