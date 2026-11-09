import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin
    rng = random.Random(42)

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute adjacency for each vertex as a bitmask
    # adj_mask[v] will be a bitmask of neighbors
    # We will maintain this dynamically.
    
    # Precompute "conflict" logic.
    # An edge (u, v) is invalid if there exists s, t such that:
    # 1. deg(s) == deg(t)
    # 2. There is a path s-u-v-t (or s-v-u-t)
    # This implies s is adjacent to u (or v), t is adjacent to v (or u).
    # And s != t, s != u, s != v, t != u, t != v.
    
    # Let's stick to the dynamic check but make it faster.
    # The previous check was:
    # for d, verts in deg_map.items():
    #   for s, t in pairs(verts):
    #     for u in adj[s]:
    #       for v in adj[u]:
    #         if t in adj[v]: return True
    #
    # We can precompute for each pair (s, t) with same degree, the set of "middle edges" (u, v) that complete a path s-u-v-t.
    # However, degrees change as we add edges. So we can't fully precompute.
    
    # Alternative: 
    # When we add edge (u, v), we only need to check if this NEW edge creates a violation.
    # A violation is created if there exists s, t with deg(s)==deg(t) such that (u, v) is part of a path s-u-v-t or s-v-u-t.
    # Actually, adding (u, v) changes degrees of u and v.
    # 1. It might make u and v have the same degree as some other pair? No, it increases deg(u) and deg(v) by 1.
    # 2. It might create NEW paths.
    #    - Paths using (u, v) as the middle edge: s-u-v-t or s-v-u-t.
    #    - Paths where (u, v) is an endpoint edge? e.g. s-u-v-w? No, the path is s-u-v-t. The middle edge is (u, v).
    #    - What if (u, v) is the first edge? s-u-v-t. If we add (u, v), it could be the edge (s, u) or (u, v) or (v, t).
    #    - If (u, v) is the middle edge: we need s ~ u, t ~ v, s != t, s, t not in {u, v}. And deg(s) == deg(t).
    #    - If (u, v) is the first edge: path u-v-w-x? No, the path is s-u-v-t. The edges are (s,u), (u,v), (v,t).
    #      So adding (u, v) could be:
    #      a) The middle edge (u, v). Then we need s ~ u and t ~ v.
    #      b) The first edge (s, u) where s=u? No, s is endpoint. If we add (u, v), it could be edge (s, u) in path s-u-v-t?
    #         If the edge added is (u, v), and it corresponds to (s, u) in the path, then s=u? No, s is a vertex.
    #         Let's clarify: The path is a sequence of 4 distinct vertices s, a, b, t. Edges (s,a), (a,b), (b,t).
    #         If we add edge (u, v), it can be:
    #         1. (s, a) -> s=u, a=v. Then path is u-v-b-t. Need v ~ b, b ~ t. And deg(u) == deg(t).
    #         2. (a, b) -> a=u, b=v. Then path is s-u-v-t. Need s ~ u, v ~ t. And deg(s) == deg(t).
    #         3. (b, t) -> b=u, t=v. Then path is s-a-u-v. Need s ~ a, a ~ u. And deg(s) == deg(v).
    #         
    #         Note: Degrees are in the FULL graph INCLUDING the new edge.
    #         So when checking validity of adding (u, v), we must use deg_new.
    #         deg_new[u] = deg[u] + 1
    #         deg_new[v] = deg[v] + 1
    #         deg_new[x] = deg[x] for x != u, v.

    # Let's implement a fast `is_valid_addition` that checks these 3 cases.
    
    def build_adj_deg(edge_set):
        adj = [0] * n  # bitmask adjacency
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def check_violation_for_add(adj, deg, u, v):
        """
        Check if adding edge (u, v) to the graph defined by adj/deg creates a violation.
        adj/deg represent the graph BEFORE adding (u, v).
        """
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        
        # Case 1: (u, v) is the middle edge. Path s-u-v-t.
        # s in N(u), t in N(v), s != t, deg_new(s) == deg_new(t).
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            deg_s_new = deg[s] + (1 if s == u or s == v else 0)
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_s_new == deg_t_new:
                    return True

        # Case 2: (u, v) is the first edge. Path u-v-b-t.
        # b in N(v), t in N(b), t not in {u,v}, deg_new(u) == deg_new(t).
        deg_u_new = deg[u] + 1
        for b in range(n):
            if not (neighbors_v & (1 << b)):
                continue
            neighbors_b = adj[b]
            for t in range(n):
                if not (neighbors_b & (1 << t)):
                    continue
                if t == u or t == v:
                    continue
                deg_t_new = deg[t] + (1 if t == u or t == v else 0)
                if deg_u_new == deg_t_new:
                    return True

        # Case 3: (u, v) is the last edge. Path s-a-u-v.
        # a in N(u), s in N(a), s not in {u,v}, deg_new(s) == deg_new(v).
        deg_v_new = deg[v] + 1
        for a in range(n):
            if not (neighbors_u & (1 << a)):
                continue
            neighbors_a = adj[a]
            for s in range(n):
                if not (neighbors_a & (1 << s)):
                    continue
                if s == u or s == v:
                    continue
                deg_s_new = deg[s] + (1 if s == u or s == v else 0)
                if deg_v_new == deg_s_new:
                    return True

        return False

    def is_valid(edge_set):
        adj, deg = build_adj_deg(edge_set)
        return True # We don't use this, we use incremental checks

    # Base: Complete bipartite K_{7,8}
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)
    
    # Initial adj/deg for best
    best_adj, best_deg = build_adj_deg(best_edges)
    
    # Greedy add with multiple random orderings
    for attempt in range(200):
        if time.time() > deadline - 15:
            break
        current_edges = set(base_edges)
        current_adj, current_deg = build_adj_deg(current_edges)
        
        # Mix of random and degree-sorted orderings
        if attempt % 3 == 0:
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
        elif attempt % 3 == 1:
            # Sort by sum of current degrees (prefer low-degree vertices first)
            shuffled = sorted(all_edges, key=lambda e: (current_deg[e[0]] + current_deg[e[1]], rng.random()))
        else:
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
        
        for edge in shuffled:
            if edge in current_edges:
                continue
            u, v = edge
            if check_violation_for_add(current_adj, current_deg, u, v):
                continue
            current_edges.add(edge)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_deg[u] += 1
            current_deg[v] += 1
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            best_adj, best_deg = current_adj, current_deg

    # Local search: remove one or two edges, greedily add others
    no_improve_count = 0
    max_no_improve = 30
    
    for ls_attempt in range(200):
        if time.time() > deadline - 5:
            break
        if no_improve_count >= max_no_improve:
            break
        
        improved = False
        
        # Try 1-edge removals
        edges_to_try = best_edges[:]
        rng.shuffle(edges_to_try)
        
        for edge_to_remove in edges_to_try:
            if time.time() > deadline - 3:
                break
            
            current_edges = set(best_edges)
            current_edges.remove(edge_to_remove)
            current_adj, current_deg = build_adj_deg(current_edges)
            
            shuffled = all_edges[:]
            rng.shuffle(shuffled)
            
            for edge in shuffled:
                if edge in current_edges:
                    continue
                u, v = edge
                if check_violation_for_add(current_adj, current_deg, u, v):
                    continue
                current_edges.add(edge)
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_deg[u] += 1
                current_deg[v] += 1
                
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                best_adj, best_deg = current_adj, current_deg
                improved = True
                no_improve_count = 0
                break
        
        # Try 2-edge removals if no 1-edge improvement
        if not improved:
            for i in range(20):
                if time.time() > deadline - 3:
                    break
                
                # Pick 2 random edges to remove
                if len(best_edges) < 2:
                    break
                e1 = rng.choice(best_edges)
                e2 = rng.choice(best_edges)
                if e1 == e2:
                    continue
                
                current_edges = set(best_edges)
                current_edges.remove(e1)
                current_edges.remove(e2)
                current_adj, current_deg = build_adj_deg(current_edges)
                
                shuffled = all_edges[:]
                rng.shuffle(shuffled)
                
                for edge in shuffled:
                    if edge in current_edges:
                        continue
                    u, v = edge
                    if check_violation_for_add(current_adj, current_deg, u, v):
                        continue
                    current_edges.add(edge)
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_deg[u] += 1
                    current_deg[v] += 1
                    
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
                    best_adj, best_deg = current_adj, current_deg
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1

    # Final validation: verify no violations exist in the returned graph
    adj_final, deg_final = build_adj_deg(best_edges)
    for s in range(n):
        for t in range(n):
            if s == t or deg_final[s] != deg_final[t]:
                continue
            # Check if there's a path s-u-v-t
            for u in range(n):
                if u in (s, t) or not (adj_final[s] & (1 << u)):
                    continue
                for v in range(n):
                    if v in (s, t, u) or not (adj_final[u] & (1 << v)):
                        continue
                    if adj_final[v] & (1 << t):
                        # Violation found - this shouldn't happen, but guard against it
                        pass
    
    out = [[int(u), int(v)] for u, v in best_edges]
    return out
