import random
import itertools
import time

def construct():
    n = 20
    all_edges = list(itertools.combinations(range(n), 2))
    edge_set = set(all_edges)
    
    # Precompute all pairs
    def is_valid_add(adj, u, v):
        # Check if adding (u,v) creates a K2,3
        # 1. Common neighbors of u and v must be < 3
        if len(adj[u] & adj[v]) >= 3:
            return False
        # 2. For any w in adj[u], common neighbors of v and w must be < 3
        #    (Note: u is a common neighbor of v and w now)
        for w in adj[u]:
            if w == v: continue
            # Common neighbors of v and w include u.
            # So old common neighbors of v and w must be < 2.
            if len(adj[v] & adj[w]) >= 2:
                return False
        # 3. For any w in adj[v], common neighbors of u and w must be < 3
        #    (Note: v is a common neighbor of u and w now)
        for w in adj[v]:
            if w == u: continue
            if len(adj[u] & adj[w]) >= 2:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(adj, u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def greedy_build():
        adj = [set() for _ in range(n)]
        edges = list(all_edges)
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def local_search(adj, max_iters=20000):
        current_edges = count_edges(adj)
        best_adj = [set(s) for s in adj]
        best_edges = current_edges
        
        for it in range(max_iters):
            # Try to add an edge
            improved = False
            current_edge_set = set()
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        current_edge_set.add((u, v))
            
            missing = list(edge_set - current_edge_set)
            if missing:
                random.shuffle(missing)
                # Try first few missing edges
                for u, v in missing[:5]:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        improved = True
                        break
            
            # If no edge was added, try a 2-opt move (remove 1, add 1)
            if not improved:
                if current_edges == 0:
                    break
                all_current = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            all_current.append((u, v))
                
                if not all_current:
                    break
                    
                # Try a few random removals
                for _ in range(3):
                    u, v = random.choice(all_current)
                    remove_edge(adj, u, v)
                    current_edges -= 1
                    
                    # Try to add as many edges as possible
                    current_edge_set = set()
                    for a in range(n):
                        for b in adj[a]:
                            if a < b:
                                current_edge_set.add((a, b))
                    
                    added_any = False
                    for eu, ev in all_edges:
                        if (eu, ev) in current_edge_set: continue
                        if (ev, eu) in current_edge_set: continue
                        if is_valid_add(adj, eu, ev):
                            add_edge(adj, eu, ev)
                            current_edges += 1
                            added_any = True
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [set(s) for s in adj]
                    
                    # Revert if not improved
                    if current_edges <= best_edges:
                        # Re-add the removed edge
                        add_edge(adj, u, v)
                        current_edges += 1
                        # We need to remove any edges that were added during the trial
                        # This is complex, so let's just accept the state if it's not worse
                        # Actually, for simplicity, let's just keep the state if it's not worse than before the removal
                        # But we don't have the previous state saved easily.
                        # Let's just revert to best_adj if we want strict improvement.
                        # For now, let's just keep the current state and continue.
                        # This might drift, but it's a heuristic.
                        # To be safe, let's revert to the state before the removal if no net gain.
                        # But we don't have it. So let's just accept the current state.
                        # This is a bit messy. Let's try a different approach.
                        # Let's just keep the state if it's not worse than the best so far.
                        # If it is worse, we revert to best_adj.
                        if current_edges < best_edges:
                            adj = [set(s) for s in best_adj]
                            current_edges = best_edges
                        break  # Stop trying more removals for this iteration
                
        return best_adj, best_edges

    best_edges_count = 0
    best_final_adj = None
    
    # Run multiple restarts
    start_time = time.time()
    restarts = 0
    while time.time() - start_time < 110:
        adj = greedy_build()
        adj, edges = local_search(adj)
        if edges > best_edges_count:
            best_edges_count = edges
            best_final_adj = adj
        restarts += 1
        
    # Convert best_final_adj to list of edges
    edges = []
    if best_final_adj:
        for u in range(n):
            for v in best_final_adj[u]:
                if u < v:
                    edges.append([u, v])
    else:
        # Fallback to simple greedy if local search failed
        adj = greedy_build()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])

    return edges
