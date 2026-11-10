import time
import random

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 118

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_bitmask(adj, n=16):
    colors = [-1] * n

    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        # Pick vertex with most colored neighbors
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v

        v = best_v
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]

        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        return True

    return dfs((1 << n) - 1)

def greedy_3_colorable(adj, n=16, max_trials=20):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_fast(adj, n=16):
    if greedy_3_colorable(adj, n, max_trials=10):
        return False
    return is_not_3_colorable_bitmask(adj, n)

def build_groetzsch_edges():
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def get_groetzsch_adj(n=11):
    edges = build_groetzsch_edges()
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def generate_blowup_distributions(num_extra, num_original):
    dists = []
    def rec(rem, idx, current):
        if idx == num_original - 1:
            current.append(rem)
            dists.append(tuple(current))
            current.pop()
            return
        for i in range(rem + 1):
            current.append(i)
            rec(rem - i, idx + 1, current)
            current.pop()
    rec(num_extra, 0, [])
    return dists

def add_edge(adj, edges, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u
    edges.append([u, v])

def remove_edge(adj, edges, u, v):
    adj[u] ^= 1 << v
    adj[v] ^= 1 << u
    edges.pop()

def polish_add_edges(adj, edges, n=16, deadline=None):
    """Repeatedly try to add any triangle-safe non-edge that keeps the graph not-3-colorable."""
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        random.shuffle(candidates)
        for u, v in candidates:
            if deadline is not None and time.time() > deadline:
                break
            add_edge(adj, edges, u, v)
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                remove_edge(adj, edges, u, v)
    return edges

def swap_improve(adj, edges, n=16, deadline=None):
    """2-opt edge swap: remove a present edge (u,v) and add a triangle-safe non-edge (a,b).
    Accept if the result stays not-3-colorable (edge count preserved, structure changed)."""
    improved_any = False
    if deadline is not None and time.time() > deadline:
        return edges, improved_any
    present = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                present.append((u, v))
    random.shuffle(present)
    for (u, v) in present:
        if deadline is not None and time.time() > deadline:
            break
        # Try removing (u,v) then adding a safe non-edge
        remove_edge(adj, edges, u, v)
        candidates = []
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)):
                    if (adj[a] & adj[b]) == 0:
                        candidates.append((a, b))
        random.shuffle(candidates)
        accepted = False
        for (a, b) in candidates:
            if deadline is not None and time.time() > deadline:
                break
            add_edge(adj, edges, a, b)
            if is_not_3_colorable_fast(adj, n):
                accepted = True
                improved_any = True
                break
            else:
                remove_edge(adj, edges, a, b)
        if not accepted:
            # restore original edge
            add_edge(adj, edges, u, v)
    return edges, improved_any

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 118

    n = 16
    num_original = 11
    num_extra = n - num_original

    g_adj = get_groetzsch_adj(num_original)

    best_edges = []
    best_count = 0

    dists = generate_blowup_distributions(num_extra, num_original)

    for dist in dists:
        if time.time() - START_TIME > TIME_LIMIT:
            break

        sizes = [1 + d for d in dist]

        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1

        edges = []
        adj = [0] * n

        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u

        if not is_triangle_free_bitmask(adj, n):
            continue

        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)

    if not best_edges:
        # Fallback
        return build_groetzsch_edges()

    # Polish: greedy edge addition
    adj = build_adj(best_edges, n)
    current_edges = list(best_edges)
    deadline = START_TIME + TIME_LIMIT

    polish_add_edges(adj, current_edges, n, deadline)
    best_count = len(current_edges)
    best_edges = list(current_edges)

    # Swap-based improvement to escape local optima
    if time.time() < deadline:
        current_edges, _ = swap_improve(adj, current_edges, n, deadline)
        # Re-polish after swaps
        polish_add_edges(adj, current_edges, n, deadline)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)

    return best_edges
