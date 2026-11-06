import time
import random
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 110.0  # safety margin before 150s kill

def _build_triangles(adj):
    tris = set()
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                continue
            for k in range(j+1, N):
                if adj[i][k] and adj[j][k]:
                    tris.add((i, j, k))
    return tris

def _max_sum(tris, deg):
    if not tris:
        return 0
    return max(deg[a] + deg[b] + deg[c] for a, b, c in tris)

def _apply_swap(adj, deg, tris, e_rm, e_add):
    """Apply swap: remove e_rm=(u,v), add e_add=(x,y).
    Returns (new_deg, new_tris). Mutates adj and deg in place."""
    u, v = e_rm
    x, y = e_add
    deg[u] -= 1
    deg[v] -= 1
    deg[x] += 1
    deg[y] += 1
    adj[u][v] = adj[v][u] = False
    adj[x][y] = adj[y][x] = True

    # Triangles destroyed: those containing both u and v
    destroyed = set()
    for t in tris:
        tset = set(t)
        if u in tset and v in tset:
            destroyed.add(t)
    new_tris = set(tris - destroyed)

    # Triangles created: those containing both x and y, third vertex w
    # must be adjacent to both x and y (and w != x, w != y)
    for w in range(N):
        if w == x or w == y:
            continue
        if adj[x][w] and adj[y][w]:
            new_tris.add(tuple(sorted((x, y, w))))
    return deg, new_tris

def _undo_swap(adj, deg, e_rm, e_add, old_deg, old_tris):
    u, v = e_rm
    x, y = e_add
    deg[u] = old_deg[u]
    deg[v] = old_deg[v]
    deg[x] = old_deg[x]
    deg[y] = old_deg[y]
    adj[u][v] = adj[v][u] = True
    adj[x][y] = adj[y][x] = False
    return deg, old_tris

def _hill_climb(adj, deg, tris, t0, rng, start_temp=3.0, end_temp=0.05):
    cur_score = _max_sum(tris, deg)
    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_tris = set(tris)

    edges = [(i, j) for i in range(N) for j in range(i+1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i+1, N) if not adj[i][j]]

    max_iters = 300000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    iters = 0
    stall_count = 0
    stall_threshold = 15000

    while iters < max_iters:
        if iters % 500 == 0:
            if time.time() - t0 > DEADLINE:
                break

        u, v = edges[rng.randrange(len(edges))]
        x, y = nonedges[rng.randrange(len(nonedges))]

        if (u == x and v == y) or (u == y and v == x):
            iters += 1
            temp *= cooling_rate
            continue

        old_deg = deg[:]
        old_tris = tris
        new_deg, new_tris = _apply_swap(adj, deg, tris, (u, v), (x, y))
        new_score = _max_sum(new_tris, new_deg)

        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.3
        else:
            delta = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()

        if accept:
            deg, tris = new_deg, new_tris
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_tris = set(tris)
                stall_count = 0
            else:
                stall_count += 1
            edges.remove((u, v))
            nonedges.append((u, v))
            nonedges.remove((x, y))
            edges.append((x, y))
        else:
            adj[u][v] = adj[v][u] = True
            adj[x][y] = adj[y][x] = False
            deg = old_deg
            tris = old_tris

        iters += 1
        temp *= cooling_rate

    return best_adj, best_deg, best_tris, best_score

def construct():
    t0 = time.time()
    rng = random.Random(42)

    def _make_initial(edge_list):
        adj = [[False]*N for _ in range(N)]
        for i, j in edge_list:
            adj[i][j] = adj[j][i] = True
        deg = [0]*N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        tris = _build_triangles(adj)
        return adj, deg, tris

    def _k88_edges(extra_edge):
        """K_{8,8} edges plus one extra edge."""
        edges = []
        for i in range(8):
            for j in range(8, 16):
                edges.append((i, j))
        edges.append(extra_edge)
        return edges

    best_overall_adj = None
    best_overall_score = float('inf')

    # Multiple initial configurations
    initial_configs = [
        _k88_edges((0, 1)),
        _k88_edges((0, 2)),
        _k88_edges((0, 7)),
        _k88_edges((1, 2)),
        _k88_edges((0, 8)),  # cross-partition (already in K88, so skip)
        _k88_edges((1, 3)),
        _k88_edges((2, 5)),
        _k88_edges((0, 15)),  # cross-partition
    ]

    # Also try a near-regular circulant-like start
    circ_edges = []
    for i in range(N):
        for d in [1, 2, 3, 4, 5, 6, 7]:
            j = (i + d) % N
            if i < j:
                circ_edges.append((i, j))
    # Count edges
    ec = len(circ_edges)
    # 16*7/2 = 56 edges. Need 65, add 9 more
    # Add remaining pairs to reach 65
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    existing = set(circ_edges)
    for p in all_pairs:
        if ec >= TARGET_EDGES:
            break
        if p not in existing:
            circ_edges.append(p)
            existing.add(p)
            ec += 1
    if len(circ_edges) == TARGET_EDGES:
        initial_configs.append(circ_edges)

    for cfg in initial_configs:
        if time.time() - t0 > DEADLINE - 5.0:
            break
        adj, deg, tris = _make_initial(cfg)
        best_adj, best_deg, best_tris, best_score = _hill_climb(adj, deg, tris, t0, rng)
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_adj = best_adj

    if best_overall_adj is None:
        # Fallback
        adj, deg, tris = _make_initial(_k88_edges((0, 1)))
        best_overall_adj = adj
        best_overall_score = _max_sum(tris, deg)

    final_edges = [[i, j] for i in range(N) for j in range(i+1, N) if best_overall_adj[i][j]]
    return {'n': N, 'edges': final_edges}
