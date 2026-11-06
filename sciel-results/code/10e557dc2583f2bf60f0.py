import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    def max_triangle_free_size(adj):
        """Exact branch-and-bound for largest triangle-free induced subgraph."""
        triangles = get_triangles(adj)
        if not triangles:
            return n

        # For each vertex, store list of (other1, other2) pairs such that
        # {v, other1, other2} forms a triangle
        vertex_tri = [[] for _ in range(n)]
        for t in triangles:
            verts = [v for v in range(n) if (t >> v) & 1]
            a, b, c = verts
            vertex_tri[a].append((b, c))
            vertex_tri[b].append((a, c))
            vertex_tri[c].append((a, b))

        # Order vertices by number of triangles they participate in (descending)
        # This gives better pruning
        order = sorted(range(n), key=lambda v: len(vertex_tri[v]), reverse=True)
        order_map = [0] * n
        for i, v in enumerate(order):
            order_map[v] = i

        best_found = 0

        def dfs(pos, mask, size):
            nonlocal best_found
            # Upper bound: current size + remaining vertices
            remaining = n - pos
            if size + remaining <= best_found:
                return
            if pos == n:
                if size > best_found:
                    best_found = size
                return

            v = order[pos]

            # Option 1: exclude v
            dfs(pos + 1, mask, size)

            # Option 2: include v (if it doesn't create a triangle with already-included vertices)
            valid = True
            for u, w in vertex_tri[v]:
                if (mask >> u) & 1 and (mask >> w) & 1:
                    valid = False
                    break

            if valid:
                dfs(pos + 1, mask | (1 << v), size + 1)

        dfs(0, 0, 0)
        return best_found

    def find_independent_set_paley(adj_paley):
        """Greedy independent set in the Paley graph on 17 vertices."""
        n_p = 17
        remaining = list(range(n_p))
        indep = []
        while remaining:
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and (adj_paley[v] >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            indep.append(min_v)
            remaining.remove(min_v)
            to_remove = [u for u in remaining if (adj_paley[min_v] >> u) & 1]
            for u in to_remove:
                remaining.remove(u)
        return indep

    # 1. Generate Paley Graph P(17)
    residues = set([i * i % 17 for i in range(17)])
    residues.discard(0)

    adj_paley = [0] * 17
    for u in range(17):
        for v in range(u + 1, 17):
            if (v - u) % 17 in residues:
                adj_paley[u] |= (1 << v)
                adj_paley[v] |= (1 << u)

    # 2. Build initial graph: Paley(17) + vertex 17 connected to an independent set
    random.seed(42)
    indep_set = find_independent_set_paley(adj_paley)

    best_adj = None
    best_score = 18

    # Try the greedy independent set
    adj = adj_paley[:] + [0]
    for v in indep_set:
        adj[17] |= (1 << v)
        adj[v] |= (1 << 17)

    if not has_k4(adj):
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    # Also try a few random independent sets of similar size
    for trial in range(20):
        if time.time() > deadline - 80:
            break
        # Randomly select a subset of vertices that is independent
        # Use random ordering for greedy
        random.shuffle(indep_set)
        adj = adj_paley[:] + [0]
        for v in indep_set:
            adj[17] |= (1 << v)
            adj[v] |= (1 << 17)
        if not has_k4(adj):
            score = max_triangle_free_size(adj)
            if score < best_score:
                best_score = score
                best_adj = adj[:]

    if best_adj is None:
        # Fallback: K_{6,6,6}
        adj1 = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj1[a] |= (1 << b)
                        adj1[b] |= (1 << a)
        best_adj = adj1
        best_score = max_triangle_free_size(best_adj)

    # 3. Local Search: hill-climbing with restarts
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Hill climbing: only accept improving moves
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 5:
        if best_score_sa <= 5:
            break

        if no_improve_count >= max_no_improve:
            # Restart from best solution with a small perturbation
            adj = best_adj_sa[:]
            current_score = best_score_sa
            no_improve_count = 0
            # Perturb: flip 2-3 random edges
            for _ in range(random.randint(2, 3)):
                i, j = random.choice(all_edges)
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                if has_k4(adj):
                    # Revert
                    if (adj[i] >> j) & 1:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if has_k4(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                current_score = max_triangle_free_size(adj)
            continue

        i, j = random.choice(all_edges)
        old_adj = adj[:]

        # Flip edge
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

        if has_k4(adj):
            adj = old_adj
            no_improve_count += 1
            continue

        new_score = max_triangle_free_size(adj)

        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_adj_sa = adj[:]
        else:
            adj = old_adj
            no_improve_count += 1

    return edges_from_adj(best_adj_sa)
