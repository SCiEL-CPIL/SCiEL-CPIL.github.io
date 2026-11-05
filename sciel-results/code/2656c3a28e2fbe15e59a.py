import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 118.0  # Safety margin before 120s limit

    # Precompute all possible edges
    all_edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx += 1
    m_total = len(all_edges)  # 190

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def is_valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        nu_new = adj[u] | (1 << v)
        nv_new = adj[v] | (1 << u)
        
        for w in range(n):
            if w == u or w == v:
                continue
            nw = adj[w]
            common_uw = nu_new & nw
            if bin(common_uw).count('1') > 2:
                return False
            common_vw = nv_new & nw
            if bin(common_vw).count('1') > 2:
                return False
        
        common_uv = nu_new & nv_new
        if bin(common_uv).count('1') > 2:
            return False
        
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def try_add_all(adj, edges_count):
        """Greedily add all possible edges."""
        made_progress = True
        while made_progress:
            made_progress = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        edges_count += 1
                        made_progress = True
        return adj, edges_count

    def polish(adj, edges_count):
        """Final polish: try all possible edge additions in multiple passes."""
        for _ in range(3):  # Multiple passes
            made_progress = True
            while made_progress:
                made_progress = False
                for u, v in all_edges:
                    if not ((adj[u] >> v) & 1):
                        if is_valid_add(adj, u, v):
                            add_edge(adj, u, v)
                            edges_count += 1
                            made_progress = True
        return adj, edges_count

    def local_search(adj, edges_count, time_budget):
        """Local search with swaps, time-bounded."""
        current_adj = adj[:]
        current_edges = edges_count
        best_adj = current_adj[:]
        best_edges = current_edges
        
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.01
        
        while time.time() < time_budget:
            move_type = random.random()
            
            if move_type < 0.6:
                # Try to add a random non-existent edge
                u, v = random.choice(all_edges)
                if not ((current_adj[u] >> v) & 1):
                    if is_valid_add(current_adj, u, v):
                        add_edge(current_adj, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = current_adj[:]
            else:
                # Try a swap: remove an existing edge, add a different one
                # Collect existing edges
                existing = []
                for u, v in all_edges:
                    if (current_adj[u] >> v) & 1:
                        existing.append((u, v))
                
                if not existing:
                    continue
                    
                u, v = random.choice(existing)
                # Pick a random non-existent edge to add
                x, y = random.choice(all_edges)
                if (current_adj[x] >> y) & 1 or (x == u and y == v) or (x == v and y == u):
                    continue
                
                # Try the swap
                new_adj = current_adj[:]
                remove_edge(new_adj, u, v)
                if is_valid_add(new_adj, x, y):
                    add_edge(new_adj, x, y)
                    # Accept swap with some probability to explore
                    if random.random() < 0.25:
                        current_adj = new_adj
                        # current_edges unchanged
        
        return best_adj, best_edges

    def init_graph(seed):
        """Initialize with a random starting graph."""
        adj = [0] * n
        rng = random.Random(seed)
        
        # Start with some random edges
        for u, v in all_edges:
            if rng.random() < 0.15:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
        
        return adj

    # Best result across all restarts
    global_best_adj = None
    global_best_edges = 0

    # Multiple restarts with different seeds
    max_restarts = 30
    restart_time = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        seed = restart * 7919 + 12345  # Different seed per restart
        random.seed(seed)
        
        adj = init_graph(seed)
        adj, ec = try_add_all(adj, 0)
        
        # Local search with time budget
        search_budget = time.time() + restart_time * 0.8
        adj, ec = local_search(adj, ec, search_budget)
        
        # Final polish
        adj, ec = polish(adj, ec)
        
        if ec > global_best_edges:
            global_best_edges = ec
            global_best_adj = adj[:]

    # If no good result yet, try a structured initial graph
    if global_best_edges < 35:
        adj = [0] * n
        # Try 5 disjoint K4s as starting point
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    add_edge(adj, u, v)
        
        adj, ec = try_add_all(adj, 30)
        search_budget = time.time() + 5.0
        adj, ec = local_search(adj, ec, search_budget)
        adj, ec = polish(adj, ec)
        
        if ec > global_best_edges:
            global_best_edges = ec
            global_best_adj = adj[:]

    # Final polish on the best result
    if global_best_adj is not None:
        global_best_adj, global_best_edges = polish(global_best_adj, global_best_edges)

    # Convert to list of edges
    result = []
    for u, v in all_edges:
        if (global_best_adj[u] >> v) & 1:
            result.append([u, v])
    
    return result
