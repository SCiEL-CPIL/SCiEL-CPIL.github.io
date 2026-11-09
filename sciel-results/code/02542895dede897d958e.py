import numpy as np
import itertools
import time
import random
import sys

N = 16
sys.setrecursionlimit(100000)

# Grid points and index mapping
POINTS = np.array(list(itertools.product(range(N), repeat=2)), dtype=int)
NUM_POINTS = N * N
IDX = {(x, y): x * N + y for x in range(N) for y in range(N)}

# Squared distances between every pair of points
diff = POINTS[:, np.newaxis, :] - POINTS[np.newaxis, :, :]
D = np.sum(diff * diff, axis=2)

# ---------------------------------------------------------------------------
# Build the 3-uniform hypergraph of isosceles triangles.
#   edges: list of (a, b, c) with a < b < c, one isosceles triangle
#   adj:   adj[v] = set of vertices that share an edge with v
# A maximal no-isosceles set S is a dominating set: every edge meets S.
# ---------------------------------------------------------------------------
edges = []
adj = [set() for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dij = D[i, j]
        for k in range(j + 1, NUM_POINTS):
            if dij == D[i, k] or dij == D[j, k] or D[i, k] == D[j, k]:
                edges.append((i, j, k))
                adj[i].add(j); adj[i].add(k)
                adj[j].add(i); adj[j].add(k)
                adj[k].add(i); adj[k].add(j)

for v in range(NUM_POINTS):
    adj[v] = list(adj[v])

# For each vertex, the list of edge indices it participates in.
vert_edges = [[] for _ in range(NUM_POINTS)]
for e, (a, b, c) in enumerate(edges):
    vert_edges[a].append(e)
    vert_edges[b].append(e)
    vert_edges[c].append(e)

NUM_EDGES = len(edges)
EDGE_A = np.array([e[0] for e in edges], dtype=np.int32)
EDGE_B = np.array([e[1] for e in edges], dtype=np.int32)
EDGE_C = np.array([e[2] for e in edges], dtype=np.int32)


def _is_valid_no_iso(S):
    """True iff S contains no isosceles triangle (i.e. S is an independent set)."""
    Sset = set(S)
    for a, b, c in edges:
        if a in Sset and b in Sset and c in Sset:
            return False
    return True


def _is_maximal(S):
    """True iff no point outside S can be added without creating an isosceles triangle."""
    Sset = set(S)
    for p in range(NUM_POINTS):
        if p in Sset:
            continue
        # p can be added iff for every S-vertex q, {p, q, r} is not an isosceles edge
        # for any r in S. Equivalently: no edge contains p and two members of S.
        for e in vert_edges[p]:
            a, b, c = edges[e]
            # count how many of a,b,c are in S (p is not in S)
            if (a in Sset) + (b in Sset) + (c in Sset) >= 2:
                return False
    return True


# ---------------------------------------------------------------------------
# Branch-and-bound minimum dominating set of the 3-uniform hypergraph.
# State: dom_mask (bool array, edge is dominated), S (chosen vertices).
# Lower bound: number of undominated edges whose three vertices are all
# "unavailable" (not in S and not adjacent to any S vertex) -- such an edge
# can never be dominated, so the search is dead; otherwise the count of
# undominated edges that can only be covered by not-yet-chosen vertices gives
# a packing lower bound on the number of additional picks needed.
# ---------------------------------------------------------------------------
def _bb_min_dominating(seed_set, budget_iters, deadline):
    """Return a dominating set (list of vertex indices) of size <= current best,
    or None. Uses recursive branch-and-bound with time budget."""
    global _bb_time
    best = {'set': None, 'size': len(seed_set)}

    # Precompute adjacency boolean matrix for fast "available" queries.
    adj_mat = np.zeros((NUM_POINTS, NUM_POINTS), dtype=bool)
    for v in range(NUM_POINTS):
        if adj[v]:
            adj_mat[v, adj[v]] = True

    dom_mask = np.zeros(NUM_EDGES, dtype=bool)
    S = list(seed_set)
    Sset = set(S)
    for v in S:
        for e in vert_edges[v]:
            dom_mask[e] = True

    # Vertices already "covered" by S (in S or adjacent to S) -- these are the
    # only vertices that can dominate any remaining edge.
    covered = np.zeros(NUM_POINTS, dtype=bool)
    for v in S:
        covered[v] = True
        if adj[v]:
            covered[adj[v]] = True

    # Candidate vertices: those that are covered (can dominate something) and
    # not already chosen.
    candidates = [v for v in range(NUM_POINTS) if covered[v] and v not in Sset]

    # Precompute, for each candidate vertex, the bitmask (as set of ints) of
    # undominated edges it would newly dominate.
    def newly_dom(v):
        out = []
        for e in vert_edges[v]:
            if not dom_mask[e]:
                out.append(e)
        return out

    state = {'dom': dom_mask, 'covered': covered, 'S': S, 'Sset': Sset,
             'candidates': candidates}

    # Order candidates by how many undominated edges they cover (most first).
    def cand_score(v):
        return sum(1 for e in vert_edges[v] if not dom_mask[e])

    # Lower bound: number of undominated edges that can ONLY be covered by
    # mutually non-adjacent vertices requires that many picks. We use a greedy
    # packing lower bound: repeatedly pick an undominated edge, count it, and
    # remove all undominated edges that share a vertex with it (one pick can
    # cover at most one from an edge-disjoint-in-vertices set).
    def lower_bound():
        # Greedy packing of vertex-disjoint undominated edges.
        used = np.zeros(NUM_POINTS, dtype=bool)
        cnt = 0
        for e in range(NUM_EDGES):
            if dom_mask[e]:
                continue
            a, b, c = edges[e]
            if used[a] or used[b] or used[c]:
                continue
            used[a] = used[b] = used[c] = True
            cnt += 1
        return cnt

    # Recursive search.
    nodes = [0]
    start = time.time()

    def dfs(depth):
        if time.time() > deadline:
            return
        nodes[0] += 1
        cur_size = len(state['S'])
        if cur_size >= best['size']:
            return

        # All edges dominated?
        if not state['dom'].any():
            if cur_size < best['size']:
                best['size'] = cur_size
                best['set'] = list(state['S'])
            return

        # Lower bound: if even the packing LB forces cur_size + LB >= best, prune.
        lb = lower_bound()
        if cur_size + lb >= best['size']:
            return

        # Dead edge: an undominated edge whose three vertices are all not covered.
        for e in range(NUM_EDGES):
            if state['dom'][e]:
                continue
            a, b, c = edges[e]
            if not (state['covered'][a] or state['covered'][b] or state['covered'][c]):
                # Cannot dominate this edge at all -> dead branch.
                return

        # Choose a vertex to branch on: the candidate covering the most
        # undominated edges.
        best_v = -1
        best_gain = -1
        for v in state['candidates']:
            g = 0
            for e in vert_edges[v]:
                if not state['dom'][e]:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_v = v
        if best_v < 0:
            return

        v = best_v
        # Branch 1: take v
        state['S'].append(v)
        state['Sset'].add(v)
        state['covered'][v] = True
        if adj[v]:
            state['covered'][adj[v]] = True
        new_edges = []
        for e in vert_edges[v]:
            if not state['dom'][e]:
                state['dom'][e] = True
                new_edges.append(e)
        # add newly-covered vertices to candidates
        add_cand = []
        if adj[v]:
            for u in adj[v]:
                if not state['Sset'].__contains__(u) and u not in state['candidates']:
                    add_cand.append(u)
        state['candidates'] = [c for c in state['candidates'] if c != v] + add_cand
        dfs(depth + 1)
        # undo
        state['S'].pop()
        state['Sset'].discard(v)
        for e in new_edges:
            state['dom'][e] = False
        # restore covered: recompute covered from S (cheap enough at small depth)
        state['covered'][:] = False
        for w in state['S']:
            state['covered'][w] = True
            if adj[w]:
                state['covered'][adj[w]] = True
        state['candidates'] = [c for c in state['candidates'] if c not in add_cand and c != v]
        # re-add v to candidates
        state['candidates'].append(v)

        # Branch 2: skip v (only valid if v is not the sole cover of some edge)
        # Determine if skipping v is safe: every undominated edge that v covers
        # must be coverable by another covered vertex.
        skip_ok = True
        for e in vert_edges[v]:
            if state['dom'][e]:
                continue
            a, b, c = edges[e]
            others = [x for x in (a, b, c) if x != v]
            if not any(state['covered'][x] for x in others):
                skip_ok = False
                break
        if skip_ok:
            # remove v from candidates
            state['candidates'] = [c for c in state['candidates'] if c != v]
            dfs(depth + 1)
            state['candidates'].append(v)

    # Initial upper bound from a greedy dominating set.
    def greedy_dominating():
        S = list(seed_set)
        Sset = set(S)
        dom = np.zeros(NUM_EDGES, dtype=bool)
        for v in S:
            for e in vert_edges[v]:
                dom[e] = True
        while dom.any():
            # pick vertex covering most undominated edges
            best_v, best_g = -1, -1
            for v in range(NUM_POINTS):
                if v in Sset:
                    continue
                g = sum(1 for e in vert_edges[v] if not dom[e])
                if g > best_g:
                    best_g = g
                    best_v = v
            if best_v < 0 or best_g <= 0:
                break
            S.append(best_v)
            Sset.add(best_v)
            for e in vert_edges[best_v]:
                dom[e] = True
        return S

    g = greedy_dominating()
    if _is_valid_no_iso(g):
        best['set'] = g
        best['size'] = len(g)

    # Seed a better upper bound using local-search removal on greedy result.
    if best['set'] is not None and time.time() < deadline - 5:
        improved = _local_search(best['set'], max_iters=200, time_limit=max(1.0, deadline - time.time() - 5))
        if len(improved) < best['size']:
            best['set'] = improved
            best['size'] = len(improved)

    dfs(0)

    if best['set'] is None:
        return g  # fallback to greedy
    return best['set']


def _local_search(S, max_iters=500, time_limit=None):
    """Remove vertices while keeping S a valid no-isosceles dominating set."""
    S = list(S)
    Sset = set(S)
    start = time.time()
    for _ in range(max_iters):
        if time_limit and time.time() - start > time_limit:
            break
        improved = False
        order = S[:]
        random.shuffle(order)
        for v in order:
            if v not in Sset:
                continue
            new_S = [x for x in S if x != v]
            if _is_valid_no_iso(new_S) and _is_maximal(new_S):
                S = new_S
                Sset.discard(v)
                improved = True
        if not improved:
            break
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 250  # safety margin under 330s kill

    rng = random.Random(7)

    # Phase 1: greedy dominating sets from random seeds, then local-search.
    best = None
    best_len = float('inf')

    # A few random small seeds to feed the branch-and-bound.
    for trial in range(3):
        if time.time() > deadline - 60:
            break
        # random seed: pick a handful of points
        seed = list(rng.sample(range(NUM_POINTS), rng.randint(0, 4)))
        t0 = time.time()
        cand = _bb_min_dominating(seed, budget_iters=100000, deadline=deadline - 60)
        if cand is not None and _is_valid_no_iso(cand) and _is_maximal(cand):
            if len(cand) < best_len:
                best_len = len(cand)
                best = cand
        # also local-search the greedy result
        if cand is not None:
            ls = _local_search(cand, max_iters=300, time_limit=max(1.0, deadline - 30 - time.time()))
            if _is_valid_no_iso(ls) and _is_maximal(ls) and len(ls) < best_len:
                best_len = len(ls)
                best = ls

    # Phase 2: pure greedy dominating set as a guaranteed valid fallback.
    greedy = _local_search(_greedy_dominating_full(), max_iters=500,
                           time_limit=max(1.0, deadline - time.time()))
    if _is_valid_no_iso(greedy) and _is_maximal(greedy) and len(greedy) < best_len:
        best = greedy
        best_len = len(greedy)

    if best is None:
        # Absolute fallback: a maximal no-isosceles set via greedy addition.
        best = _greedy_add_maximal()

    return [tuple(POINTS[i]) for i in best]


def _greedy_dominating_full():
    S = []
    Sset = set()
    dom = np.zeros(NUM_EDGES, dtype=bool)
    while dom.any():
        best_v, best_g = -1, -1
        for v in range(NUM_POINTS):
            if v in Sset:
                continue
            g = sum(1 for e in vert_edges[v] if not dom[e])
            if g > best_g:
                best_g = g
                best_v = v
        if best_v < 0 or best_g <= 0:
            break
        S.append(best_v)
        Sset.add(best_v)
        for e in vert_edges[best_v]:
            dom[e] = True
    return S


def _greedy_add_maximal():
    """Greedy: add points while no isosceles triangle forms, until maximal."""
    S = []
    Sset = set()
    for v in range(NUM_POINTS):
        ok = True
        for e in vert_edges[v]:
            a, b, c = edges[e]
            if (a in Sset) + (b in Sset) + (c in Sset) >= 2:
                ok = False
                break
        if ok:
            S.append(v)
            Sset.add(v)
    return S
