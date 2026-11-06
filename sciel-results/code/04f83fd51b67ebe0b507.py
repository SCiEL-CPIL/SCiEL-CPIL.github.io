import random
import time

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65

    # ---------- fast bitmask-based scoring ----------
    def score_graph(masks, deg):
        """Return max triangle degree-sum using bitmask adjacency."""
        max_sum = 0
        for u in range(n):
            mu = masks[u]
            du = deg[u]
            for v in range(u + 1, n):
                if not (mu >> v) & 1:
                    continue
                # common neighbors w > v
                common = mu & masks[v] & (~((1 << (v + 1)) - 1))
                if common == 0:
                    continue
                dv = deg[v]
                # iterate set bits in common
                c = common
                while c:
                    lb = c & (-c)
                    w = lb.bit_length() - 1
                    s = du + dv + deg[w]
                    if s > max_sum:
                        max_sum = s
                    c ^= lb
        return max_sum

    def find_worst_triangle(masks, deg):
        """Return (max_sum, (u,v,w) or None)."""
        max_sum = 0
        worst = None
        for u in range(n):
            mu = masks[u]
            du = deg[u]
            for v in range(u + 1, n):
                if not (mu >> v) & 1:
                    continue
                common = mu & masks[v] & (~((1 << (v + 1)) - 1))
                if common == 0:
                    continue
                dv = deg[v]
                c = common
                while c:
                    lb = c & (-c)
                    w = lb.bit_length() - 1
                    s = du + dv + deg[w]
                    if s > max_sum:
                        max_sum = s
                        worst = (u, v, w)
                    c ^= lb
        return max_sum, worst

    def edges_to_masks(edges):
        masks = [0] * n
        deg = [0] * n
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return masks, deg

    def masks_to_edges(masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def non_edges(masks):
        out = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((masks[u] >> v) & 1):
                    out.append((u, v))
        return out

    def edge_list(masks):
        out = []
        for u in range(n):
            for v in range(u + 1, n):
                if (masks[u] >> v) & 1:
                    out.append((u, v))
        return out

    # ---------- degree-ordered edge-replacement heuristic ----------
    def degree_ordered_search(masks, deg, edges, current_score, max_rounds=50000):
        """
        Iteratively swap a high-degree-sum edge (from the worst triangle) 
        with a low-degree-sum non-edge to reduce the max triangle degree sum.
        Prioritizes lowering the degree of the highest-degree vertex in the worst triangle.
        """
        rounds = 0
        while time.time() < deadline and rounds < max_rounds:
            rounds += 1
            max_sum, worst = find_worst_triangle(masks, deg)
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
                a, b = (highest, other) if highest < other else (other, highest)
                if (masks[a] >> b) & 1:
                    cand_rems.append((a, b))
            # Second priority: the remaining triangle edges
            for a, b in [(u, v), (u, w), (v, w)]:
                if a > b:
                    a, b = b, a
                if (a, b) not in cand_rems:
                    cand_rems.append((a, b))
            
            # Candidate adds: non-edges with minimal degree sum, 
            # but avoid adding edges that create new high-degree triangles
            ne = non_edges(masks)
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
                for add in ne_sorted[:40]:
                    a, b = add
                    if (a == x and b == y) or (a == y and b == x):
                        continue
                    # Perform swap
                    masks[a] |= (1 << b)
                    masks[b] |= (1 << a)
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
                    deg[a] += 1; deg[b] += 1
                    deg[x] -= 1; deg[y] -= 1
                    
                    ns = score_graph(masks, deg)
                    
                    # Undo
                    masks[a] &= ~(1 << b)
                    masks[b] &= ~(1 << a)
                    masks[x] |= (1 << y)
                    masks[y] |= (1 << x)
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
            masks[a] |= (1 << b)
            masks[b] |= (1 << a)
            masks[x] &= ~(1 << y)
            masks[y] &= ~(1 << x)
            deg[a] += 1; deg[b] += 1
            deg[x] -= 1; deg[y] -= 1
            
            # Update edges list
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append([a, b])
            
            current_score = best_new_score
        
        return masks, deg, edges, current_score

    # ---------- general local search for refinement ----------
    def local_search(masks, deg, edges, current_score, max_rounds=30000):
        rounds = 0
        while time.time() < deadline and rounds < max_rounds:
            rounds += 1
            best_gain = 0
            best_add = None
            best_rem = None
            
            ne = non_edges(masks)
            el = edge_list(masks)
            
            ne_sorted = sorted(ne, key=lambda t: deg[t[0]] + deg[t[1]])
            el_sorted = sorted(el, key=lambda t: -(deg[t[0]] + deg[t[1]]))
            
            for add in ne_sorted[:25]:
                u, v = add
                for rem in el_sorted[:25]:
                    x, y = rem
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
                    deg[u] += 1; deg[v] += 1
                    deg[x] -= 1; deg[y] -= 1
                    
                    ns = score_graph(masks, deg)
                    
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    masks[x] |= (1 << y)
                    masks[y] |= (1 << x)
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
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            masks[x] &= ~(1 << y)
            masks[y] &= ~(1 << x)
            deg[u] += 1; deg[v] += 1
            deg[x] -= 1; deg[y] -= 1
            
            for i, e in enumerate(edges):
                if (e[0] == x and e[1] == y) or (e[0] == y and e[1] == x):
                    edges.pop(i)
                    break
            edges.append([u, v])
            
            current_score -= best_gain
        
        return masks, deg, edges, current_score

    # ---------- initial state: K_{8,8} + one edge, then break symmetry ----------
    # Start with K_{8,8} plus one internal edge, then do a few random swaps to break symmetry
    init_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    masks, deg = edges_to_masks(init_edges)
    edges = [e[:] for e in init_edges]
    
    # Do a few random swaps to break the K_{8,8} symmetry
    for _ in range(5):
        ne = non_edges(masks)
        el = edge_list(masks)
        if not ne or not el:
            break
        add = random.choice(ne)
        rem = random.choice(el)
        if (add[0] == rem[0] and add[1] == rem[1]) or (add[0] == rem[1] and add[1] == rem[0]):
            continue
        masks[add[0]] |= (1 << add[1])
        masks[add[1]] |= (1 << add[0])
        masks[rem[0]] &= ~(1 << rem[1])
        masks[rem[1]] &= ~(1 << rem[0])
        deg[add[0]] += 1; deg[add[1]] += 1
        deg[rem[0]] -= 1; deg[rem[1]] -= 1
        for i, e in enumerate(edges):
            if (e[0] == rem[0] and e[1] == rem[1]) or (e[0] == rem[1] and e[1] == rem[0]):
                edges.pop(i)
                break
        edges.append(list(add))
    
    current_score = score_graph(masks, deg)

    best_score = current_score
    best_masks = masks[:]
    best_deg = deg[:]
    best_edges = [e[:] for e in edges]

    # Alternating degree-ordered search + general local search
    while time.time() < deadline:
        # Phase 1: Degree-ordered edge-replacement
        masks, deg, edges, current_score = degree_ordered_search(
            masks, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_masks = masks[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        
        if time.time() > deadline:
            break
        
        # Phase 2: General local search for fine-tuning
        masks, deg, edges, current_score = local_search(
            masks, deg, edges, current_score, max_rounds=30000)
        if current_score < best_score:
            best_score = current_score
            best_masks = masks[:]
            best_deg = deg[:]
            best_edges = [e[:] for e in edges]
        
        # Random perturbation to escape local minima
        if time.time() > deadline:
            break
        
        masks_p = best_masks[:]
        deg_p = best_deg[:]
        edges_p = [e[:] for e in best_edges]
        for _ in range(4):
            ne = non_edges(masks_p)
            el = edge_list(masks_p)
            if not ne or not el:
                break
            add = random.choice(ne)
            rem = random.choice(el)
            if (add[0] == rem[0] and add[1] == rem[1]) or (add[0] == rem[1] and add[1] == rem[0]):
                continue
            masks_p[add[0]] |= (1 << add[1])
            masks_p[add[1]] |= (1 << add[0])
            masks_p[rem[0]] &= ~(1 << rem[1])
            masks_p[rem[1]] &= ~(1 << rem[0])
            deg_p[add[0]] += 1; deg_p[add[1]] += 1
            deg_p[rem[0]] -= 1; deg_p[rem[1]] -= 1
            for i, e in enumerate(edges_p):
                if (e[0] == rem[0] and e[1] == rem[1]) or (e[0] == rem[1] and e[1] == rem[0]):
                    edges_p.pop(i)
                    break
            edges_p.append(list(add))
        
        masks_p, deg_p, edges_p, sc_p = degree_ordered_search(
            masks_p, deg_p, edges_p, score_graph(masks_p, deg_p), max_rounds=15000)
        masks_p, deg_p, edges_p, sc_p = local_search(
            masks_p, deg_p, edges_p, sc_p, max_rounds=15000)
        if sc_p < best_score:
            best_score = sc_p
            best_masks = masks_p[:]
            best_deg = deg_p[:]
            best_edges = [e[:] for e in edges_p]

    # final validation
    final_masks, final_deg = edges_to_masks(best_edges)
    assert len(best_edges) == target_edges, f"edge count {len(best_edges)} != {target_edges}"
    return {'n': 16, 'edges': best_edges}
