import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65

    # ---------- fast numpy-based scoring ----------
    def score_graph(adj, deg):
        """Return max triangle degree-sum using numpy adjacency."""
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                # common neighbors w > v
                common = adj[u] & adj[v]
                common[:v + 1] = False
                ws = np.where(common)[0]
                if len(ws) == 0:
                    continue
                sums = deg[u] + deg[v] + deg[ws]
                local_max = sums.max()
                if local_max > max_sum:
                    max_sum = int(local_max)
        return max_sum

    def find_worst_triangle(adj, deg):
        """Return (max_sum, (u,v,w) or None)."""
        max_sum = 0
        worst = None
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                common = adj[u] & adj[v]
                common[:v + 1] = False
                ws = np.where(common)[0]
                if len(ws) == 0:
                    continue
                sums = deg[u] + deg[v] + deg[ws]
                local_max = sums.max()
                if local_max > max_sum:
                    max_sum = int(local_max)
                    idx = int(np.argmax(sums))
                    worst = (u, v, int(ws[idx]))
        return max_sum, worst

    def edges_to_adj(edges):
        adj = np.zeros((n, n), dtype=bool)
        deg = np.zeros(n, dtype=int)
        for u, v in edges:
            adj[u, v] = adj[v, u] = True
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def non_edges(adj):
        out = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    out.append((u, v))
        return out

    def edge_list(adj):
        out = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    out.append((u, v))
        return out

    # ---------- degree-ordered edge-replacement heuristic ----------
    def degree_ordered_search(adj, deg, edges, current_score, max_rounds=50000):
        """
        Iteratively swap a high-degree-sum edge (from the worst triangle) 
        with a low-degree-sum non-edge to reduce the max triangle degree sum.
        Prioritizes lowering the degree of the highest-degree vertex in the worst triangle.
        """
        rounds = 0
        while time.time() < deadline and rounds < max_rounds:
            rounds += 1
            max_sum, worst = find_worst_triangle(adj, deg)
            if worst is None:
                break
            
            u, v, w = worst
            # Sort vertices in the worst triangle by degree descending
            tri_verts = sorted([u, v, w], key=lambda x: -deg[x])
            highest = tri_verts[0]
            
            # Candidate removals: edges incident to the highest-degree vertex in the worst triangle
            # Also include the other two edges of the triangle as fallback
            cand_rems = []
            # First priority: edges from highest vertex to other triangle members
            for other in tri_verts[1:]:
                if adj[highest, other]:
                    a, b = (highest, other) if highest < other else (other, highest)
                    cand_rems.append((a, b))
            # Second priority: the remaining triangle edges
            for a, b in [(u, v), (u, w), (v, w)]:
                if a > b:
                    a, b = b, a
                if (a, b) not in cand_rems:
                    cand_rems.append((a, b))
            
            # Candidate adds: non-edges with minimal degree sum, 
            # but avoid adding edges that create new high-degree triangles
            ne = non_edges(adj)
            # Sort by degree sum, but penalize if it involves high-degree vertices
            def add_key(t):
                a, b = t
                base = deg[a] + deg[b]
                # Penalize if either endpoint is very high degree
                penalty = max(0, deg[a] - 8) + max(0, deg[b] - 8)
                return base + 0.5 * penalty
            
            ne_sorted = sorted(ne, key=add_key)
            
            best_gain = 0
            best_add = None
            best_rem = None
            best_new_score = current_score
            
            # Try each candidate removal with the top candidate additions
            for rem in cand_rems:
                x, y = rem
                for add in ne_sorted[:30]:
                    a, b = add
                    if (a == x and b == y) or (a == y and b == x):
                        continue
                    # Perform swap
                    adj[a, b] = adj[b, a] = True
                    adj[x, y] = adj[y, x] = False
                    deg[a] += 1; deg[b] += 1
                    deg[x] -= 1; deg[y] -= 1
                    
                    ns = score_graph(adj, deg)
                    
                    # Undo
                    adj[a, b] = adj[b, a] = False
                    adj[x, y] = adj[y, x] = True
                    deg[a] -= 1; deg[b] -= 1
                    deg[x] += 1; deg[y] += 1
                    
                    gain = current_score - ns
                    if gain > best_gain:
                        best_gain = gain
                        best_add = add
                        best_rem = rem
                        best_new_score = ns
                        if gain >= 3:
                            break
                if best_gain >= 3:
                    break
            
            if best_add is None:
                break
            
            # Apply the best swap
            a, b = best_add
            x, y = best_rem
            adj[a, b] = adj[b, a] = True
            adj[x, y] = adj[y, x] = False
            deg[a] += 1; deg[b] += 1
            deg[x] -= 1; deg[y] -= 1
            
            # Update edges list
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append([a, b])
            
            current_score = best_new_score
        
        return adj, deg, edges, current_score

    # ---------- general local search for refinement ----------
    def local_search(adj, deg, edges, current_score, max_rounds=30000):
        rounds = 0
        while time.time() < deadline and rounds < max_rounds:
            rounds += 1
            best_gain = 0
            best_add = None
            best_rem = None
            
            ne = non_edges(adj)
            el = edge_list(adj)
            
            ne_sorted = sorted(ne, key=lambda t: deg[t[0]] + deg[t[1]])
            el_sorted = sorted(el, key=lambda t: -(deg[t[0]] + deg[t[1]]))
            
            for add in ne_sorted[:20]:
                u, v = add
                for rem in el_sorted[:20]:
                    x, y = rem
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    adj[u, v] = adj[v, u] = True
                    adj[x, y] = adj[y, x] = False
                    deg[u] += 1; deg[v] += 1
                    deg[x] -= 1; deg[y] -= 1
                    
                    ns = score_graph(adj, deg)
                    
                    adj[u, v] = adj[v, u] = False
                    adj[x, y] = adj[y, x] = True
                    deg[u] -= 1; deg[v] -= 1
                    deg[x] += 1; deg[y] += 1
                    
                    gain = current_score - ns
                    if gain > best_gain:
                        best_gain = gain
                        best_add = add
                        best_rem = rem
                        if gain >= 2:
                            break
                if best_gain >= 2:
                    break
            
            if best_add is None:
                break
            
            u, v = best_add
            x, y = best_rem
            adj[u, v] = adj[v, u] = True
            adj[x, y] = adj[y, x] = False
            deg[u] += 1; deg[v] += 1
            deg[x] -= 1; deg[y] -= 1
            
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append([u, v])
            
            current_score -= best_gain
        
        return adj, deg, edges, current_score

    # ---------- initial state: K_{8,8} + one edge ----------
    init_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    adj, deg = edges_to_adj(init_edges)
    edges = [e[:] for e in init_edges]
    current_score = score_graph(adj, deg)

    best_score = current_score
    best_adj = adj.copy()
    best_deg = deg.copy()
    best_edges = [e[:] for e in edges]

    # Alternating degree-ordered search + general local search
    while time.time() < deadline:
        # Phase 1: Degree-ordered edge-replacement
        adj, deg, edges, current_score = degree_ordered_search(
            adj, deg, edges, current_score, max_rounds=20000)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_edges = [e[:] for e in edges]
        
        if time.time() > deadline:
            break
        
        # Phase 2: General local search for fine-tuning
        adj, deg, edges, current_score = local_search(
            adj, deg, edges, current_score, max_rounds=20000)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_edges = [e[:] for e in edges]
        
        # Random perturbation to escape local minima
        if time.time() > deadline:
            break
        
        adj_p = best_adj.copy()
        deg_p = best_deg.copy()
        edges_p = [e[:] for e in best_edges]
        for _ in range(3):
            ne = non_edges(adj_p)
            el = edge_list(adj_p)
            if not ne or not el:
                break
            add = random.choice(ne)
            rem = random.choice(el)
            if (add[0] == rem[0] and add[1] == rem[1]) or (add[0] == rem[1] and add[1] == rem[0]):
                continue
            adj_p[add[0], add[1]] = adj_p[add[1], add[0]] = True
            adj_p[rem[0], rem[1]] = adj_p[rem[1], rem[0]] = False
            deg_p[add[0]] += 1; deg_p[add[1]] += 1
            deg_p[rem[0]] -= 1; deg_p[rem[1]] -= 1
            for i, e in enumerate(edges_p):
                if (e[0] == rem[0] and e[1] == rem[1]) or (e[0] == rem[1] and e[1] == rem[0]):
                    edges_p.pop(i)
                    break
            edges_p.append(list(add))
        
        adj_p, deg_p, edges_p, sc_p = degree_ordered_search(
            adj_p, deg_p, edges_p, score_graph(adj_p, deg_p), max_rounds=10000)
        adj_p, deg_p, edges_p, sc_p = local_search(
            adj_p, deg_p, edges_p, sc_p, max_rounds=10000)
        if sc_p < best_score:
            best_score = sc_p
            best_adj = adj_p.copy()
            best_deg = deg_p.copy()
            best_edges = [e[:] for e in edges_p]

    # final validation
    final_adj, final_deg = edges_to_adj(best_edges)
    assert len(best_edges) == target_edges, f"edge count {len(best_edges)} != {target_edges}"
    return {'n': 16, 'edges': best_edges}
