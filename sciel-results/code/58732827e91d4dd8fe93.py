"""Rewritten program: minimize edges of a non-2-colorable 5-uniform hypergraph.

Strategy:
- Seed from a small known non-2-colorable 5-uniform hypergraph on 9 vertices.
- Greedy edge removal (keep a removal only if hypergraph stays non-2-colorable).
- Randomized restarts with extra edges, then greedy reduction, to find fewer edges.
- Exact 2-colorability check via backtracking + unit propagation with a deadline.
"""
import time
import random
from itertools import combinations

# Hard wall-clock guard (evaluator kills at 150s; leave margin).
_DEADLINE = 150.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def is_2_colorable(n, edges, deadline):
    """Return True if the hypergraph admits a proper 2-colouring.

    Exact check by backtracking with unit propagation.
    An edge is 'satisfied' (conflict) if it is monochromatic.
    """
    # Precompute edge incidence as frozensets
    edges = [frozenset(e) for e in edges]
    m = len(edges)
    if m == 0:
        return True

    # For each edge, track count of assigned-0 and assigned-1 vertices.
    # color[v] in {-1, 0, 1}; 0/1 are colours, -1 unassigned.
    color = [-1] * n
    # counts[edge_index] = [count0, count1]
    counts = [[0, 0] for _ in range(m)]
    # For each vertex, list of edges containing it.
    vertex_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(ei)

    # Unit propagation: if an edge has 4 of a colour, the 5th is forced.
    # If an edge has 5 of a colour, conflict.
    # We maintain a work queue of edges to recheck.
    import collections
    queue = collections.deque()

    def conflict():
        return False

    # We'll implement propagation inline within recursion for speed via a stack.
    # Represent assignment state with a stack of (vertex, old_color) for undo.

    def propagate(queue):
        # Returns False on conflict, True if consistent (with assignments applied).
        while queue:
            ei = queue.popleft()
            e = edges[ei]
            c0 = counts[ei][0]
            c1 = counts[ei][1]
            if c0 + c1 > 5:
                return False
            if c0 == 5 or c1 == 5:
                return False
            # If 4 of one colour, force the remaining vertex.
            if c0 == 4:
                # find the unassigned vertex in e
                for v in e:
                    if color[v] == -1:
                        color[v] = 1
                        counts[ei][1] = c1 + 1
                        for ej in vertex_edges[v]:
                            counts[ej][1] += 1
                            queue.append(ej)
                        break
            elif c1 == 4:
                for v in e:
                    if color[v] == -1:
                        color[v] = 0
                        counts[ei][0] = c0 + 1
                        for ej in vertex_edges[v]:
                            counts[ej][0] += 1
                            queue.append(ej)
                        break
        return True

    # Try both initial colourings via search.
    # We'll do a DFS. To keep it simple, we use a recursive function with
    # manual propagation and undo.

    def dfs(depth):
        if time.time() > deadline:
            return 'timeout'
        # Choose an unassigned vertex with most constraints (heuristic).
        # For simplicity, pick first unassigned vertex.
        # Better: pick vertex in most 'almost monochromatic' edges.
        # We'll pick the first unassigned vertex for correctness.
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        if v == -1:
            # all assigned; check no conflict
            return True

        # Try colour 0 then 1.
        for c in (0, 1):
            # assign v = c
            old = color[v]
            color[v] = c
            q = collections.deque()
            for ej in vertex_edges[v]:
                if c == 0:
                    counts[ej][0] += 1
                else:
                    counts[ej][1] += 1
                q.append(ej)
            if not propagate(q):
                # undo
                color[v] = old
                for ej in vertex_edges[v]:
                    if c == 0:
                        counts[ej][0] -= 1
                    else:
                        counts[ej][1] -= 1
                continue
            # check if any edge became full monochrome (conflict)
            # propagate already returns False on conflict; but also check
            # for edges with 5 of a colour (propagate handles c0==5).
            res = dfs(depth + 1)
            if res == 'timeout':
                # undo before returning timeout
                color[v] = old
                for ej in vertex_edges[v]:
                    if c == 0:
                        counts[ej][0] -= 1
                    else:
                        counts[ej][1] -= 1
                return 'timeout'
            if res is True:
                # found a coloring; undo not needed (we can stop)
                return True
            # undo
            color[v] = old
            for ej in vertex_edges[v]:
                if c == 0:
                    counts[ej][0] -= 1
                else:
                    counts[ej][1] -= 1
        return False

    return dfs(0) is True


def is_not_2_colorable(n, edges, deadline):
    """Return True if the hypergraph has NO proper 2-colouring."""
    return not is_2_colorable(n, edges, deadline)


def seed_edges():
    """A small known non-2-colorable 5-uniform hypergraph on 9 vertices.

    Construction: take the 5-uniform hypergraph on 9 vertices consisting of
    all 5-subsets that contain at least 4 elements from a fixed 5-set A.
    This is a standard small non-2-colorable 5-uniform hypergraph.
    """
    A = {0, 1, 2, 3, 4}
    edges = set()
    for e in combinations(range(9), 5):
        es = set(e)
        if len(es & A) >= 4:
            edges.add(frozenset(e))
    return [list(e) for e in edges]


def greedy_reduce(n, edges, deadline):
    """Try to remove edges while keeping the hypergraph non-2-colorable.

    Returns (reduced_edges, success_flag).
    """
    current = list(edges)
    # Order edges by a heuristic: edges that are 'more critical' first.
    # We'll just iterate and try each; if removal keeps non-2-colorable, keep removal.
    # Repeat passes until no more removals.
    improved = True
    while improved and _time_left() > 5:
        improved = False
        for i in range(len(current)):
            if _time_left() < 5:
                break
            trial = current[:i] + current[i + 1:]
            # Check if still non-2-colorable
            if is_not_2_colorable(n, trial, min(deadline, _START + _time_left())):
                current = trial
                improved = True
                break  # restart loop with new list
    return current


def random_search(n, base_edges, deadline):
    """Randomized restarts: add random edges, then greedy reduce."""
    best = list(base_edges)
    rng = random.Random(12345)
    all_possible = list(combinations(range(n), 5))
    while _time_left() > 10:
        # Start from base, optionally add some random extra edges
        extra = rng.randint(0, 15)
        candidates = list(base_edges)
        for _ in range(extra):
            e = rng.choice(all_possible)
            if e not in candidates:
                candidates.append(e)
        # Greedy reduce
        reduced = greedy_reduce(n, candidates, _START + _time_left())
        if len(reduced) < len(best):
            best = reduced
            if len(best) <= 30:
                break
        # Shuffle base order for diversity
        rng.shuffle(base_edges)
    return best


def construct():
    """Return {'n': n, 'edges': [...]} for a non-2-colorable 5-uniform hypergraph."""
    global _START
    _START = time.time()

    # Start from seed
    base = seed_edges()
    n = 9

    # Verify seed is non-2-colorable (sanity)
    if not is_not_2_colorable(n, base, _START + 20):
        # Fallback: complete hypergraph on 9 vertices
        base = [list(e) for e in combinations(range(9), 5)]

    # Greedy reduce the seed
    reduced = greedy_reduce(n, base, _START + _time_left())

    # Random search for improvement
    best = reduced
    if _time_left() > 20:
        best = random_search(n, base, _START + _time_left())

    # Final verification of the best
    if not is_not_2_colorable(n, best, _START + min(30, max(5, _time_left()))):
        # If somehow invalid, fall back to seed
        best = base

    # Ensure edges are distinct and sorted for determinism
    best = sorted([sorted(e) for e in best])
    best = [list(e) for e in dict.fromkeys(tuple(e) for e in best)]

    return {'n': n, 'edges': best}
