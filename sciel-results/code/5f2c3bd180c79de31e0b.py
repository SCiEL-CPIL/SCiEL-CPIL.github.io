"""Graph with 40 vertices and 401 edges minimizing the largest book size."""
import time
import numpy as np

N = 40
E = 401
_PARTS = (10, 10, 10, 10)


def _book_score(adj):
    """adj: 40x40 bool adjacency. Return (max_book, count_of_max)."""
    # triangles containing each edge = sum over common neighbors
    common = (adj.astype(np.int8) @ adj.astype(np.int8))
    common = (common > 0) & adj  # triangle edge (u,v) with common neighbor w
    # book size of edge (u,v) = number of common neighbors w
    # count common neighbors per edge
    cn = np.where(adj, common, 0).sum(axis=2)  # for each (u,v): #w with adj[u,v]&adj[u,w]&adj[v,w]
    maxb = int(cn.max())
    return maxb, int((cn == maxb).sum())


def _edge_list(adj):
    out = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u, v]:
                out.append([u, v])
    return out


def _score_of_list(edges):
    adj = np.zeros((N, N), dtype=bool)
    for u, v in edges:
        adj[u, v] = adj[v, u] = True
    b, _ = _book_score(adj)
    return b


def _build_adj(part_of, prob, rng, target_edges):
    """Build a 40x40 adjacency from per-pair probabilities, then tune to exact edge count."""
    adj = np.zeros((N, N), dtype=bool)
    # inter-part edges
    for i in range(N):
        pi = part_of[i]
        for j in range(i + 1, N):
            pj = part_of[j]
            if pi == pj:
                continue
            if rng.random() < prob[pi, pj]:
                adj[i, j] = adj[j, i] = True
    # intra-part edges (random)
    for i in range(N):
        pi = part_of[i]
        for j in range(i + 1, N):
            if part_of[j] == pi:
                if rng.random() < 0.0:
                    adj[i, j] = adj[j, i] = True
    # adjust to exact edge count
    cur = int(adj.sum() // 2)
    if cur < target_edges:
        # add random missing edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_pairs)
        for i, j in all_pairs:
            if cur >= target_edges:
                break
            if not adj[i, j]:
                adj[i, j] = adj[j, i] = True
                cur += 1
    elif cur > target_edges:
        # remove random present edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_pairs)
        for i, j in all_pairs:
            if cur <= target_edges:
                break
            if adj[i, j]:
                adj[i, j] = adj[j, i] = False
                cur -= 1
    return adj


def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    rng = np.random.default_rng(20240501)

    part_of = []
    for p, sz in enumerate(_PARTS):
        part_of.extend([p] * sz)
    part_of = np.array(part_of)

    # Initialize inter-part probabilities: Turán-ish (connect all inter-part, drop some)
    # Total inter-part pairs = 40^2/2 - 4*(10*9/2) = 800 - 180 = 620. Need 401 edges.
    # Start with all inter-part connected = 620, too many. Reduce.
    prob = np.full((4, 4), 0.65)
    np.fill_diagonal(prob, 0.0)

    def _expected_book(adj):
        b, _ = _book_score(adj)
        return b

    best_edges = None
    best_score = 999

    def _eval_and_maybe_update(adj):
        nonlocal best_edges, best_score
        b, _ = _book_score(adj)
        if b < best_score:
            best_score = b
            best_edges = _edge_list(adj)
            return True
        return False

    # Phase 1: optimize inter-part probabilities via hill climbing on expected book
    cur_adj = _build_adj(part_of, prob, rng, E)
    _eval_and_update = _eval_and_maybe_update
    _eval_and_update(cur_adj)

    # coordinate descent on prob entries
    pairs_idx = [(i, j) for i in range(4) for j in range(4) if i != j]
    for it in range(400):
        if time.time() > deadline - 30:
            break
        improved = False
        for (i, j) in pairs_idx:
            old = prob[i, j]
            for delta in (0.05, -0.05, 0.10, -0.10, 0.02, -0.02):
                new = min(1.0, max(0.0, old + delta))
                prob[i, j] = new
                adj = _build_adj(part_of, prob, rng, E)
                b, _ = _book_score(adj)
                if b < best_score:
                    best_score = b
                    best_edges = _edge_list(adj)
                    old = new
                    improved = True
                    break
                else:
                    prob[i, j] = old
        if not improved:
            break

    # Phase 2: local search on the edge set (flip edges to reduce max book)
    if best_edges is not None:
        adj = np.zeros((N, N), dtype=bool)
        for u, v in best_edges:
            adj[u, v] = adj[v, u] = True
        cur_b, _ = _book_score(adj)
        cur_cn = np.where(adj, (adj.astype(np.int8) @ adj.astype(np.int8)) > 0, 0).sum(axis=2)
        # hill climb: try flipping edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        for it in range(30000):
            if time.time() > deadline - 15:
                break
            if cur_b <= 6:
                break
            i, j = all_pairs[rng.integers(len(all_pairs))]
            if adj[i, j]:
                # try removing
                adj[i, j] = adj[j, i] = False
                nb, _ = _book_score(adj)
                if nb < cur_b:
                    cur_b = nb
                    cur_cn = np.where(adj, (adj.astype(np.int8) @ adj.astype(np.int8)) > 0, 0).sum(axis=2)
                else:
                    adj[i, j] = adj[j, i] = True
            else:
                # try adding
                adj[i, j] = adj[j, i] = True
                nb, _ = _book_score(adj)
                if nb < cur_b:
                    cur_b = nb
                    cur_cn = np.where(adj, (adj.astype(np.int8) @ adj.astype(np.int8)) > 0, 0).sum(axis=2)
                else:
                    adj[i, j] = adj[j, i] = False
        if cur_b < best_score:
            best_score = cur_b
            best_edges = _edge_list(adj)

    if best_edges is None or len(best_edges) != E:
        # fallback to seed
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]

    return best_edges
