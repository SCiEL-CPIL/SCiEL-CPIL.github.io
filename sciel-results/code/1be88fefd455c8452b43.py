import random
import time
import math

def construct():
    deadline = time.time() + 110  # safety margin
    n = 16
    target_edges = 65

    def score_graph(adj_mask, deg):
        """Compute the max triangle degree-sum using bitmask adjacency."""
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w = common.bit_length() - 1
                    common ^= (1 << w)
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
        return max_sum

    def edges_to_mask(edges):
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_mask, deg

    def mask_to_edges(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def init_edges():
        # Complete bipartite K_{8,8} plus one extra edge
        return [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]

    # Initialize
    best_edges = init_edges()
    best_adj, best_deg = edges_to_mask(best_edges)
    best_score = score_graph(best_adj, best_deg)

    def compute_triangle_degs(adj_mask, deg):
        """For each vertex, compute sum of deg(w) over all triangles containing that vertex."""
        tri_deg_sum = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w = common.bit_length() - 1
                    common ^= (1 << w)
                    tri_deg_sum[u] += deg[w]
                    tri_deg_sum[v] += deg[w]
                    tri_deg_sum[w] += deg[u] + deg[v]
        return tri_deg_sum

    def max_cut_potential(adj_mask, side):
        """Compute number of edges crossing the cut defined by side vector."""
        cut = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    if side[u] != side[v]:
                        cut += 1
        return cut

    def gradient_guided_swap(adj_mask, deg, best_score, best_adj, best_deg):
        """Try gradient-guided swaps: prefer adding edges that cut the Max-Cut structure
        and removing edges within same-side clusters."""
        # Compute triangle degree sums
        tri_deg = compute_triangle_degs(adj_mask, deg)
        
        # Determine side assignment based on triangle degree (high tri_deg -> push apart)
        # Use a threshold: vertices with above-average tri_deg go to side 1
        avg_tri = sum(tri_deg) / n
        side = [0 if tri_deg[i] <= avg_tri else 1 for i in range(n)]
        
        # Try swaps guided by Max-Cut potential
        # Prefer: add edge (u,v) where side[u] != side[v] (crossing), remove edge (x,y) where side[x] == side[y] (within)
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    # Non-edge: candidate to add
                    for x in range(n):
                        for y in range(x + 1, n):
                            if (adj_mask[x] >> y) & 1:
                                if (u == x and v == y) or (u == y and v == x):
                                    continue
                                # Score this swap by Max-Cut alignment
                                add_cross = 1 if side[u] != side[v] else 0
                                rem_within = 1 if side[x] == side[y] else 0
                                score = add_cross * 2 + rem_within
                                candidates.append((score, u, v, x, y))
        
        # Sort by score descending, try top candidates
        candidates.sort(key=lambda c: -c[0])
        
        for score, u, v, x, y in candidates:
            if time.time() > deadline:
                break
            # Apply swap
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[x] &= ~(1 << y)
            adj_mask[y] &= ~(1 << x)
            deg[u] += 1; deg[v] += 1
            deg[x] -= 1; deg[y] -= 1
            
            new_score = score_graph(adj_mask, deg)
            
            # Undo
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[x] |= (1 << y)
            adj_mask[y] |= (1 << x)
            deg[u] -= 1; deg[v] -= 1
            deg[x] += 1; deg[y] += 1
            
            if new_score < best_score:
                best_adj = adj_mask[:]
                best_deg = deg[:]
                best_score = new_score
                return True
        
        return False

    def local_search(adj_mask, deg, best_score, best_adj, best_deg, max_iters=50):
        """Standard local search: try all single swaps, accept improving ones."""
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            found = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if (adj_mask[u] >> v) & 1:
                        continue
                    for x in range(n):
                        for y in range(x + 1, n):
                            if not (adj_mask[x] >> y) & 1:
                                continue
                            if (u == x and v == y) or (u == y and v == x):
                                continue
                            # Apply swap
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                            adj_mask[x] &= ~(1 << y)
                            adj_mask[y] &= ~(1 << x)
                            deg[u] += 1; deg[v] += 1
                            deg[x] -= 1; deg[y] -= 1
                            
                            new_score = score_graph(adj_mask, deg)
                            
                            # Undo
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                            adj_mask[x] |= (1 << y)
                            adj_mask[y] |= (1 << x)
                            deg[u] -= 1; deg[v] -= 1
                            deg[x] += 1; deg[y] += 1
                            
                            if new_score < best_score:
                                best_adj = adj_mask[:]
                                best_deg = deg[:]
                                best_score = new_score
                                found = True
                                improved = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        return improved, best_score, best_adj, best_deg

    # Main search loop
    adj_curr = best_adj[:]
    deg_curr = best_deg[:]
    
    while time.time() < deadline:
        # Try gradient-guided swaps first
        if gradient_guided_swap(adj_curr, deg_curr, best_score, best_adj, best_deg):
            adj_curr = best_adj[:]
            deg_curr = best_deg[:]
            continue
        
        # Then standard local search
        improved, best_score, best_adj, best_deg = local_search(adj_curr, deg_curr, best_score, best_adj, best_deg)
        adj_curr = best_adj[:]
        deg_curr = best_deg[:]
        
        if not improved:
            # Random restart with perturbation
            if time.time() > deadline:
                break
            adj_pert = best_adj[:]
            deg_pert = best_deg[:]
            for _ in range(3):
                # Pick random non-edge to add
                while True:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u < v and not (adj_pert[u] >> v) & 1:
                        break
                # Pick random edge to remove
                while True:
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x < y and (adj_pert[x] >> y) & 1:
                        break
                # Apply swap
                adj_pert[u] |= (1 << v)
                adj_pert[v] |= (1 << u)
                adj_pert[x] &= ~(1 << y)
                adj_pert[y] &= ~(1 << x)
                deg_pert[u] += 1; deg_pert[v] += 1
                deg_pert[x] -= 1; deg_pert[y] -= 1
            
            # Run local search on perturbed state
            adj_curr = adj_pert[:]
            deg_curr = deg_pert[:]
            improved, best_score, best_adj, best_deg = local_search(adj_curr, deg_curr, best_score, best_adj, best_deg, max_iters=30)
            adj_curr = best_adj[:]
            deg_curr = best_deg[:]

    best_edges = mask_to_edges(best_adj)
    return {'n': 16, 'edges': best_edges}
