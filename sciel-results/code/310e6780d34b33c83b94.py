"""Construct a large triangle-free graph with no independent set of size 10."""
import time
import random
import sys

# Deadline guard: evaluator kills at 150s; we target finishing well before that.
_DEADLINE = 145.0  # seconds from start

def _time_left():
    return _DEADLINE - time.time()

def _is_triangle_free(adj, n):
    """Check if graph with adjacency list is triangle-free."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and (i in adj[k]):
                    return False
    return True

def _max_independent_set_size(adj, n):
    """Exact branch and bound for maximum independent set size."""
    # Use a simple branch and bound with coloring upper bound
    # For our purposes, we just need to check if alpha >= 10
    
    best = [0]
    
    # Sort vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    # Map from original vertex to position in order
    pos = [0] * n
    for idx, v in enumerate(order):
        pos[v] = idx
    
    # Build adjacency in order-index space
    adj_order = [set() for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            if pos[u] > pos[v]:
                adj_order[pos[v]].add(pos[u])
    
    # Greedy coloring for upper bound
    def color_bound(candidates):
        """Greedy coloring of the subgraph induced by candidates to get an upper bound on IS size."""
        if not candidates:
            return 0
        colors = {}
        # Order candidates by degree in the subgraph
        cand_set = set(candidates)
        deg_in_sub = {}
        for v in candidates:
            deg_in_sub[v] = len(adj_order[v] & cand_set)
        ordered = sorted(candidates, key=lambda v: deg_in_sub[v])
        
        color = 0
        for v in ordered:
            # Find smallest color not used by neighbors in the subgraph
            used_colors = set()
            for u in adj_order[v]:
                if u in cand_set and u in colors:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            color = max(color, c + 1)
        return color
    
    # Branch and bound
    def bb(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        if current_size + len(candidates) <= best[0]:
            return
        
        # Upper bound via coloring
        ub = color_bound(candidates)
        if current_size + ub <= best[0]:
            return
        
        # Pick a vertex with max degree in the candidate subgraph
        cand_set = set(candidates)
        degs = {}
        for v in candidates:
            degs[v] = len(adj_order[v] & cand_set)
        
        # Pick vertex with max degree
        v = max(candidates, key=lambda x: degs[x])
        
        # Branch 1: include v
        new_cands = [u for u in candidates if u != v and u not in adj_order[v]]
        bb(new_cands, current_size + 1)
        
        # Branch 2: exclude v
        new_cands = [u for u in candidates if u != v]
        bb(new_cands, current_size)
    
    bb(list(range(n)), 0)
    return best[0]

def _check_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if not _is_triangle_free(adj, n):
        return False
    if _max_independent_set_size(adj, n) >= 10:
        return False
    return True

def _andrasfai_9():
    """Return the Andrasfai graph A9 on 26 vertices."""
    n = 26
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 26 % 3 == 1:
                adj[i].add(j)
                adj[j].add(i)
    return adj, n

def construct():
    start_time = time.time()
    
    # Start with the Andrasfai graph A9 (26 vertices, triangle-free, alpha=9)
    adj, n = _andrasfai_9()
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to extend by adding vertices
    # Strategy: add vertices one at a time, choosing adjacency to maintain triangle-free and alpha < 10
    
    def try_extend(adj, n, max_additional=15):
        """Try to add vertices to the graph."""
        current_adj = [set(s) for s in adj]
        current_n = n
        
        for _ in range(max_additional):
            if _time_left() < 5.0:
                break
            
            # Try adding a new vertex connected to a subset of existing vertices
            # To keep triangle-free: if we connect new vertex to u and v, u and v must not be adjacent
            # To keep alpha < 10: the new vertex's non-neighbors (plus itself) form an independent set in the remaining graph
            
            # Strategy: connect to a maximal independent set's complement pattern
            # Actually, let's try: connect new vertex to a set S such that S is an independent set in G
            # (so no triangles), and the non-neighbors of new vertex have alpha <= 8 in G
            
            # Simpler approach: try random subsets
            found = False
            for attempt in range(200):
                if _time_left() < 3.0:
                    break
                
                # Choose a random independent set to be the neighbors of the new vertex
                # First, find a random independent set
                neighbors = set()
                remaining = list(range(current_n))
                random.shuffle(remaining)
                for v in remaining:
                    if all(v not in adj[u] for u in neighbors):
                        neighbors.add(v)
                
                # Check: non-neighbors of new vertex = all vertices not in neighbors
                # The new vertex is independent with all non-neighbors
                # So alpha of new graph = max(alpha of old graph, 1 + alpha of non-neighbor induced subgraph)
                non_neighbors = [v for v in range(current_n) if v not in neighbors]
                
                # Check triangle-free: neighbors must form an independent set (already ensured)
                # Check alpha: 1 + alpha(non_neighbors) < 10, i.e., alpha(non_neighbors) <= 8
                
                # Compute alpha of non-neighbor subgraph
                if not non_neighbors:
                    new_alpha = 1
                else:
                    # Build subgraph
                    nn_set = set(non_neighbors)
                    nn_adj = [set() for _ in range(len(non_neighbors))]
                    nn_index = {v: i for i, v in enumerate(non_neighbors)}
                    for i, v in enumerate(non_neighbors):
                        for u in current_adj[v]:
                            if u in nn_set:
                                nn_adj[i].add(nn_index[u])
                    
                    # Use a faster check: just verify alpha <= 8
                    # For small graphs, use the exact method
                    nn_n = len(non_neighbors)
                    if nn_n <= 20:
                        # Exact check
                        nn_best = [0]
                        def nn_bb(cands, cs):
                            if not cands:
                                if cs > nn_best[0]:
                                    nn_best[0] = cs
                                return
                            if cs + len(cands) <= nn_best[0]:
                                return
                            # Simple upper bound
                            v = cands[0]
                            # Include v
                            nn_bb([u for u in cands if u != v and u not in nn_adj[v]], cs + 1)
                            # Exclude v
                            nn_bb(cands[1:], cs)
                        nn_bb(list(range(nn_n)), 0)
                        if nn_best[0] >= 8:  # 1 + 8 = 9 < 10, so alpha <= 8 is OK
                            pass
                        else:
                            continue
                    else:
                        # Greedy check: if we can find independent set of size 9, it's bad
                        # Use greedy
                        greedy_alpha = 0
                        rem = list(range(nn_n))
                        while rem:
                            rem.sort(key=lambda x: len(nn_adj[x] & set(rem)), reverse=True)
                            # Pick vertex with min degree
                            v = min(rem, key=lambda x: len(nn_adj[x] & set(rem)))
                            greedy_alpha += 1
                            rem = [u for u in rem if u != v and u not in nn_adj[v]]
                        if greedy_alpha >= 9:
                            continue
                
                # Also need to ensure the whole graph's alpha < 10
                # The new graph's alpha = max(old_alpha, 1 + alpha(non_neighbors))
                # Old alpha is 9, so we need 1 + alpha(non_neighbors) < 10, i.e., alpha(non_neighbors) <= 8
                # Which we checked above
                
                # Add the vertex
                new_v = current_n
                new_adj = current_adj + [set()]
                for v in neighbors:
                    new_adj[v].add(new_v)
                    new_adj[new_v].add(v)
                
                current_adj = new_adj
                current_n += 1
                found = True
                break
            
            if not found:
                break
        
        return current_adj, current_n
    
    # Try multiple random extensions
    random.seed(42)
    for trial in range(50):
        if _time_left() < 10.0:
            break
        
        adj_try, n_try = try_extend(adj, n, max_additional=15)
        
        if n_try > best_n:
            # Verify
            if _check_valid(adj_try, n_try):
                best_n = n_try
                best_adj = [set(s) for s in adj_try]
                print(f"Found valid graph with n={n_try}", file=sys.stderr)
        
        # Also try different random seeds for the extension
        random.seed(trial * 12345 + 67890)
    
    # Final verification of best result
    if not _check_valid(best_adj, best_n):
        # Fall back to the original Andrasfai graph
        best_adj, best_n = _andrasfai_9()
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if j > i:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
