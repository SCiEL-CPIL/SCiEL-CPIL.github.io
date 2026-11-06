import time
import random
from itertools import combinations

N = 16
Q3_EDGES = 12  # number of edges in the 3-cube
DEADLINE = 115.0  # wall-clock safety margin (evaluator kills at 120s)

# Q3 edge set on vertices 0..7 (binary cube)
def _q3_edge_set():
    e = []
    for v in range(8):
        for b in range(3):
            if (v >> b) & 1 == 0:
                w = v | (1 << b)
                if v < w:
                    e.append((v, w))
    return e

Q3_EDGES_SET = set(_q3_edge_set())
# Precompute all embeddings: for each 8-subset, count of injective maps whose
# image edge-set (under the induced simple graph) contains all 12 Q3 edges.
# We precompute the list of "Q3 patterns": for an 8-vertex set S (sorted), the
# 8! injective assignments are too many; instead we detect Q3 by checking,
# for each 8-subset S, whether the induced subgraph on S contains a Q3 as a
# subgraph. We do this via a fast backtracking matcher.

def _has_q3_in_induced(S, adj):
    """adj: set of frozenset or set of (u,v) with u<v; S: tuple of 8 vertices.
    Returns True if the induced subgraph on S contains Q3 as a subgraph."""
    # Map S to 0..7
    idx = {v: i for i, v in enumerate(S)}
    # local adjacency as bitmasks over 8 positions
    loc = [0] * 8
    for i in range(8):
        for j in range(i + 1, 8):
            a, b = S[i], S[j]
            key = (a, b) if a < b else (b, a)
            if key in adj:
                loc[i] |= 1 << j
                loc[j] |= 1 << i
    # Backtrack: assign Q3-vertices 0..7 to positions 0..7 (injective),
    # requiring all 12 Q3 edges present in loc.
    # Prune: a position p can host Q3-vertex q only if q's neighbors (in Q3)
    # that are already placed are present in loc.
    from functools import lru_cache
    # Q3 adjacency on 0..7
    q3adj = [0]*8
    for (a,b) in Q3_EDGES_SET:
        q3adj[a] |= 1<<b
        q3adj[b] |= 1<<a
    # positions assigned to q3-vertex q -> position
    pos_of_q = [-1]*8
    q_of_pos = [-1]*8
    # candidate positions for each q3-vertex: all 8 (perm). We search over
    # permutations of positions for q3-vertices 0..7, pruning.
    # Order q3-vertices by degree (all 3) - use fixed order 0..7.
    import sys
    sys.setrecursionlimit(10000)
    def feasible(q, p):
        # Check all already-placed neighbors of q
        nb = q3adj[q]
        for r in range(8):
            if (nb >> r) & 1:
                pr = pos_of_q[r]
                if pr != -1:
                    if not (loc[p] >> pr) & 1:
                        return False
        return True
    def bt(q):
        if q == 8:
            return True
        # Try positions not used
        for p in range(8):
            if q_of_pos[p] != -1:
                continue
            if not feasible(q, p):
                continue
            pos_of_q[q] = p
            q_of_pos[p] = q
            if bt(q+1):
                return True
            pos_of_q[q] = -1
            q_of_pos[p] = -1
        return False
    return bt(0)

def _count_q3(embed, adj):
    """Count (with multiplicity over 8-subsets) number of Q3 subgraphs.
    We count each 8-subset once if it contains Q3 (sufficient for our purpose)."""
    cnt = 0
    for S in combinations(range(N), 8):
        if _has_q3_in_induced(S, adj):
            cnt += 1
    return cnt

def _build_adj(edges):
    adj = set()
    for (u, v) in edges:
        key = (u, v) if u < v else (v, u)
        adj.add(key)
    return adj

def _q3_free(edges):
    adj = _build_adj(edges)
    for S in combinations(range(N), 8):
        if _has_q3_in_induced(S, adj):
            return False
    return True

def _seed_clique_union():
    # K7 (0..6) ∪ K7 (7..13) ∪ K2 (14,15) -> 43 edges, provably Q3-free
    edges = []
    for i in range(7):
        for j in range(i+1, 7):
            edges.append([i, j])
    for i in range(7, 14):
        for j in range(i+1, 14):
            edges.append([i, j])
    edges.append([14, 15])
    return edges

def _seed_bipartite():
    # K(3,13) + triangle on A + one edge in B -> 39+3+1 = 43? Let's count:
    # 3*13=39, +3 triangle, +1 = 43. Also a valid fallback.
    edges = []
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
    edges.append([0,1]); edges.append([1,2]); edges.append([0,2])
    edges.append([3,4])
    return edges

def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Start from the best known provable seed
    best_edges = _seed_clique_union()
    best_count = len(best_edges)
    best_adj = _build_adj(best_edges)
    # Verify (should be free)
    if not _q3_free(best_edges):
        best_edges = _seed_bipartite()
        best_count = len(best_edges)
        best_adj = _build_adj(best_edges)

    # Local search: try to add random missing edges; if it creates a Q3,
    # try to fix by removing the "most responsible" edge.
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    def violations(adj):
        # Count 8-subsets that contain a Q3 (0/1 per subset)
        v = 0
        for S in combinations(range(N), 8):
            if _has_q3_in_induced(S, adj):
                v += 1
        return v

    cur_edges = list(best_edges)
    cur_adj = set(best_adj)
    cur_viol = 0  # should be 0

    while time.time() - t0 < DEADLINE:
        # Try adding a random missing edge
        missing = [p for p in all_pairs if p not in cur_adj]
        if not missing:
            break
        # Randomly sample a few candidates
        tried = 0
        improved_this_iter = False
        while tried < 200 and time.time() - t0 < DEADLINE:
            tried += 1
            cand = rng.choice(missing)
            new_adj = cur_adj | {cand}
            v = violations(new_adj)
            if v == 0:
                # Accept: we gained an edge
                cur_adj = new_adj
                cur_edges = list(new_adj)
                if len(cur_edges) > best_count:
                    best_count = len(cur_edges)
                    best_edges = list(cur_edges)
                    best_adj = set(cur_adj)
                    missing = [p for p in all_pairs if p not in cur_adj]
                improved_this_iter = True
                break
            else:
                # Try to repair: remove one edge from the candidate's
                # "conflict" - heuristically remove a random edge incident to
                # one of cand's endpoints, or the cand itself, and re-check.
                # Simple repair: drop cand, drop a random neighbor edge.
                for _ in range(3):
                    # pick a random edge to remove
                    rem = rng.choice(list(cur_adj))
                    trial = (cur_adj | {cand}) - {rem}
                    if len(trial) >= len(cur_adj):  # not smaller
                        vt = violations(trial)
                        if vt == 0:
                            cur_adj = set(trial)
                            cur_edges = list(cur_adj)
                            if len(cur_edges) > best_count:
                                best_count = len(cur_edges)
                                best_edges = list(cur_edges)
                                best_adj = set(cur_adj)
                            improved_this_iter = True
                            break
        if not improved_this_iter:
            # Try a few random removals to escape local optima (only if we
            # can re-add later). Keep it light.
            if len(cur_edges) > best_count + 0:
                # Occasionally drop a random edge to explore
                if rng.random() < 0.1:
                    rem = rng.choice(list(cur_adj))
                    cur_adj = cur_adj - {rem}
                    cur_edges = list(cur_adj)
        # Rebuild missing list periodically
        if tried >= 200 or not improved_this_iter:
            missing = [p for p in all_pairs if p not in cur_adj]
            if not missing:
                break

    # Safety: ensure best is valid
    if not _q3_free(best_edges):
        best_edges = _seed_clique_union()
        if not _q3_free(best_edges):
            best_edges = _seed_bipartite()

    return best_edges
