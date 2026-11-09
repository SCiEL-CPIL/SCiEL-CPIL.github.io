import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
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
        deg_u_new = deg[u] + 1
        deg_v_new = deg[v] + 1
        
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        
        # Case 1: (u, v) is the middle edge (a, b). Path s-u-v-t.
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

        # Case 2: (u, v) is the first edge (s, a). Path u-v-b-t.
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

        # Case 3: (u, v) is the last edge (b, t). Path s-a-u-v.
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

    def count_conflicts_for_edge(edge_set, adj, deg, u, v):
        """Count how many other edges would conflict with (u,v) if both present."""
        # This is an approximation: check if adding (u,v) to current graph causes violation,
        # and also check how many potential paths involve (u,v)
        if check_violation_for_add(adj, deg, u, v):
            return 1000  # Very high cost if it already violates
        
        # Estimate conflict potential by checking how many (s,t) pairs with same degree
        # could form paths through (u,v)
        conflicts = 0
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        
        # For case 1: paths s-u-v-t
        for s in range(n):
            if not (neighbors_u & (1 << s)):
                continue
            deg_s = deg[s] + (1 if s == u or s == v else 0)
            for t in range(n):
                if t == s:
                    continue
                if not (neighbors_v & (1 << t)):
                    continue
                deg_t = deg[t] + (1 if t == u or t == v else 0)
                if deg_s == deg_t:
                    conflicts += 1
        
        return conflicts

    # Base: Complete bipartite K_{7,8}
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)
    
    # Initial adj/deg for best
    best_adj, best_deg = build_adj_deg(best_edges)
    
    # Greedy add with conflict-aware ordering
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        current_edges = set(base_edges)
        current_adj, current_deg = build_adj_deg(current_edges)
        
        # Compute conflict scores for all non-base edges
        candidate_edges = [e for e in all_edges if e not in current_edges]
        
        # Shuffle with some randomness to explore different orderings
        random.shuffle(candidate_edges)
        
        # Sort by conflict score (lower first), but keep some randomness
        if len(candidate_edges) > 0:
            # Compute scores incrementally - this is expensive, so do it once per attempt
            scored_edges = []
            for edge in candidate_edges:
                u, v = edge
                score = count_conflicts_for_edge(current_edges, current_adj, current_deg, u, v)
                # Add small random noise to break ties and allow exploration
                score += random.random() * 0.1
                scored_edges.append((score, edge))
            scored_edges.sort(key=lambda x: x[0])
            candidate_edges = [e for _, e in scored_edges]
        
        for edge in candidate_edges:
            if edge in current_edges:
                continue
            u, v = edge
            if check_violation_for_add(current_adj, current_deg, u, v):
                continue
            # Add edge
            current_edges.add(edge)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_deg[u] += 1
            current_deg[v] += 1
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            best_adj, best_deg = current_adj, current_deg

    # Local search: remove one edge, greedily add others with conflict-aware ordering
    no_improve_rounds = 0
    for ls_attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try removing each edge from best
        for edge_to_remove in best_edges:
            if time.time() > deadline - 3:
                break
                
            u_rm, v_rm = edge_to_remove
            
            # Start from best_edges minus edge_to_remove
            current_edges = set(best_edges)
            current_edges.remove(edge_to_remove)
            current_adj, current_deg = build_adj_deg(current_edges)
            
            # Get candidate edges not in current
            candidate_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(candidate_edges)
            
            # Score and sort
            if len(candidate_edges) > 0:
                scored_edges = []
                for edge in candidate_edges:
                    u, v = edge
                    score = count_conflicts_for_edge(current_edges, current_adj, current_deg, u, v)
                    score += random.random() * 0.1
                    scored_edges.append((score, edge))
                scored_edges.sort(key=lambda x: x[0])
                candidate_edges = [e for _, e in scored_edges]
            
            for edge in candidate_edges:
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
                no_improve_rounds = 0
                break
            else:
                no_improve_rounds += 1
        
        if not improved:
            no_improve_rounds += 1
            if no_improve_rounds > 5:
                break

    out = [[u, v] for u, v in best_edges]
    return out
