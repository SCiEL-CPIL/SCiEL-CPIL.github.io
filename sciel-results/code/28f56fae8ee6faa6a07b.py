"""Dense odd-order graph avoiding equal-degree path endpoints.
Fresh design: exhaustive backtracking over a concentric edge ordering,
plus a time-budgeted randomized hill-climbing fallback."""
import time
import random

N = 15
CENTER = 7

# Concentric ordering: center, ring1, ring2, then chords within/across rings.
# Order list: (u, v) with u < v.
def _build_order():
    rings = [[CENTER], [0, 1, 2, 3, 4, 5, 6],
             [8, 9, 10, 11, 12, 13, 14]]
    # spokes: center to ring1, center to ring2
    spokes = []
    for v in rings[1]:
        spokes.append((CENTER, v) if CENTER < v else (v, CENTER))
    for v in rings[2]:
        spokes.append((CENTER, v) if CENTER < v else (v, CENTER))
    # ring1 internal
    r1 = rings[1]
    ring1_chords = [(r1[i], r1[j]) for i in range(len(r1)) for j in range(i + 1, len(r1))]
    # ring2 internal
    r2 = rings[2]
    ring2_chords = [(r2[i], r2[j]) for i in range(len(r2)) for j in range(i + 1, len(r2))]
    # cross ring1-ring2
    cross = [(r1[i], r2[j]) for i in range(len(r1)) for j in range(len(r2))]
    order = spokes + ring1_chords + ring2_chords + cross
    return order

ORDER = _build_order()
M = len(ORDER)
assert M == 105, M

# Precompute for each edge index: the neighbors of u and v among earlier edges'
# endpoints. For edge (u,v) at index k, the "other" endpoints of earlier edges
# that are adjacent to u (and not v) form candidate s for path s-u-v-t.
# But simpler: for edge (u,v), the set of vertices w != u,v that are incident to
# some earlier edge with u (i.e., w in N_u restricted to earlier) are candidate s.
# Actually s must be adjacent to u via an earlier edge. Let's precompute:
#   adj_u_earlier[k] = list of w (w != u,v) such that edge (min(u,w),max(u,w)) has index < k
# Similarly for v.
# Then a path s-u-v-t exists (with s != t) iff there is s in adj_u_earlier and t in adj_v_earlier
# with s != t. That's equivalent to: NOT (adj_u_earlier == adj_v_earlier == {x} same single) ...
# More precisely: there exists s in A, t in B, s != t.
#   If A empty or B empty: no such path.
#   Else if |A|>=2 or |B|>=2 or (A and B and A != B): yes.
#   Only fail when A and B are both singletons and equal.
# Wait: we need s != t. If A={a}, B={b}: need a != b. If A={a},B={a}: no valid (s,t) with s!=t
#   from these sets? s must be in A, t in B, s!=t. Only (a,a) -> invalid. So no path.
# If A={a}, B={b,c}: (a,b) works. So yes.
# So forbidden iff A and B are both non-empty singletons with the same element.

def _precompute_neighbors():
    pos = {}
    for k, (u, v) in enumerate(ORDER):
        pos[(u, v)] = k
        pos[(v, u)] = k
    adj_u = [[] for _ in range(M)]
    adj_v = [[] for _ in range(M)]
    for k, (u, v) in enumerate(ORDER):
        # vertices w != u,v with edge (u,w) earlier
        for w in range(N):
            if w == u or w == v:
                continue
            a, b = (u, w) if u < w else (w, u)
            if (a, b) in pos and pos[(a, b)] < k:
                adj_u[k].append(w)
        for w in range(N):
            if w == u or w == v:
                continue
            a, b = (v, w) if v < w else (w, v)
            if (a, b) in pos and pos[(a, b)] < k:
                adj_v[k].append(w)
    return adj_u, adj_v

ADJ_U, ADJ_V = _precompute_neighbors()

def _forbidden(a_u, a_v):
    # returns True if adding this edge creates a bad path
    if not a_u or not a_v:
        return False
    if len(a_u) == 1 and len(a_v) == 1:
        return a_u[0] == a_v[0]
    return True

# For hill-climbing: full check on a given edge set
def _full_check(edges_set, degrees):
    # degrees: list of int
    # for each edge (u,v), for each s in N(u) \ {v}, for each t in N(v) \ {u}, s!=t:
    #   if deg[s]==deg[t]: invalid
    # We need adjacency from edges_set.
    adj = [set() for _ in range(N)]
    for (u, v) in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    for (u, v) in edges_set:
        nu = adj[u]
        nv = adj[v]
        # s in nu \ {v}, t in nv \ {u}
        for s in nu:
            if s == v:
                continue
            ds = degrees[s]
            for t in nv:
                if t == u or t == s:
                    continue
                if degrees[t] == ds:
                    return False
    return True

def _degrees_from(edges_set):
    deg = [0] * N
    for (u, v) in edges_set:
        deg[u] += 1
        deg[v] += 1
    return deg

def _try_hill_climb(start_edges, deadline):
    # start_edges: set of (u,v)
    edges = set(start_edges)
    deg = _degrees_from(edges)
    if not _full_check(edges, deg):
        return set()
    best = len(edges)
    # randomized order of remaining edges
    remaining = [e for e in ORDER if e not in edges]
    random.shuffle(remaining)
    for e in remaining:
        if time.time() > deadline:
            break
        u, v = e
        # tentatively add
        edges.add(e)
        deg[u] += 1
        deg[v] += 1
        if _full_check(edges, deg):
            best = len(edges)
        else:
            edges.discard(e)
            deg[u] -= 1
            deg[v] -= 1
    return edges

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin vs 90s

    # Best found so far (as a set of edges)
    best_edges = set()
    best_count = -1

    # Backtracking state
    chosen = [False] * M
    chosen_edges = []
    deg = [0] * N

    def update_best():
        nonlocal best_edges, best_count
        if len(chosen_edges) > best_count:
            best_count = len(chosen_edges)
            best_edges = set(chosen_edges)

    def bt(k):
        nonlocal best_count
        if time.time() > deadline:
            return
        if k == M:
            update_best()
            return
        u, v = ORDER[k]
        # Option 1: add edge
        a_u = ADJ_U[k]
        a_v = ADJ_V[k]
        if not _forbidden(a_u, a_v):
            chosen[k] = True
            chosen_edges.append((u, v))
            deg[u] += 1
            deg[v] += 1
            bt(k + 1)
            deg[u] -= 1
            deg[v] -= 1
            chosen_edges.pop()
            chosen[k] = False
            if time.time() > deadline:
                return
        # Option 2: skip edge
        chosen[k] = False
        bt(k + 1)

    bt(0)

    # If backtracking found something, try to improve via hill-climb from it
    if best_edges:
        improved = _try_hill_climb(best_edges, deadline)
        if len(improved) > best_count:
            best_edges = improved
            best_count = len(improved)

    # If we still have time and backtracking didn't finish, do a couple hill-climb restarts
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline:
                break
            # random sparse seed
            seed_edges = set()
            for e in ORDER:
                if random.random() < 0.3:
                    seed_edges.add(e)
            # make seed valid by removing offending edges greedily
            # (just use hill_climb which starts by checking)
            cand = _try_hill_climb(seed_edges, deadline)
            if len(cand) > best_count:
                best_edges = cand
                best_count = len(cand)

    # Fallback: ensure at least the star is valid
    if not best_edges:
        for v in range(1, N):
            best_edges.add((CENTER, v))
        best_count = len(best_edges)

    # Build return list with plain Python ints
    result = []
    for (u, v) in best_edges:
        result.append([int(u), int(v)])
    return result

# Sanity self-test (not run during evaluation to save time, but guard)
if __name__ == '__main__':
    e = construct()
    print(len(e))
