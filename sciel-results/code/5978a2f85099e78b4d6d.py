import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)  # 190
    
    # Precompute for each pair (u,v) the list of vertices w != u,v
    # and for each w, whether w is adjacent to u and v
    # Actually, we'll use bitmask representation for speed
    
    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2
    
    def is_valid(adj):
        """Check if graph is K2,3-free: no two vertices have 3+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() > 2:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
        
        # Adding edge (u,v) affects:
        # 1. Pair (u,v): common neighbors must be <= 2
        if (adj[u] & adj[v]).bit_count() > 2:
            return False
        
        # 2. For each w != u,v:
        #    If w~u, then pair (w,v) gains u as common neighbor
        #    If w~v, then pair (w,u) gains v as common neighbor
        common_uv = adj[u] & adj[v]
        
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[u] >> w) & 1:
                # w~u, so pair (w,v): common neighbors of w and v
                # Currently: adj[w] & adj[v]
                # After adding (u,v): u becomes common neighbor of w and v (since w~u and u~v)
                # So new common count = (adj[w] & adj[v]).bit_count() + 1 (if u not already common)
                # u is common neighbor of w and v iff w~u and u~v. u~v is being added, w~u is true.
                # But is u already in adj[w] & adj[v]? u~w is true, u~v is being added (was false).
                # So u was NOT a common neighbor before. Now it is.
                # New count = (adj[w] & adj[v]).bit_count() + 1
                if (adj[w] & adj[v]).bit_count() > 1:
                    return False
            
            if (adj[v] >> w) & 1:
                # w~v, so pair (w,u): common neighbors of w and u
                # v becomes common neighbor of w and u
                if (adj[w] & adj[u]).bit_count() > 1:
                    return False
        
        return True
    
    def can_remove_edge(adj, u, v):
        """Check if removing edge (u,v) keeps graph K2,3-free. Always true since removing edges only helps."""
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Simulated Annealing approach
    def simulated_annealing(initial_adj, time_limit):
        adj = initial_adj[:]
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        # Start temperature
        T = 2.0
        cooling_rate = 0.9995
        
        steps = 0
        while time.time() < time_limit:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random non-edge
                non_edges = []
                for u, v in all_pairs:
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
                
                if not non_edges:
                    continue
                
                u, v = random.choice(non_edges)
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    delta = 1
                    new_edges = current_edges + 1
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = adj[:]
                    current_edges = new_edges
                else:
                    # Can't add, try a different move
                    continue
                    
            elif move_type < 0.8:
                # Try to remove a random edge
                edges = []
                for u, v in all_pairs:
                    if (adj[u] >> v) & 1:
                        edges.append((u, v))
                
                if not edges:
                    continue
                
                u, v = random.choice(edges)
                remove_edge(adj, u, v)
                delta = -1
                new_edges = current_edges - 1
                current_edges = new_edges
                
            else:
                # Edge swap: remove one edge, add another
                edges = []
                for u, v in all_pairs:
                    if (adj[u] >> v) & 1:
                        edges.append((u, v))
                
                non_edges = []
                for u, v in all_pairs:
                    if not ((adj[u] >> v) & 1):
                        non_edges.append((u, v))
                
                if not edges or not non_edges:
                    continue
                
                rem_u, rem_v = random.choice(edges)
                add_u, add_v = random.choice(non_edges)
                
                # Remove first
                remove_edge(adj, rem_u, rem_v)
                
                # Try to add
                if can_add_edge(adj, add_u, add_v):
                    add_edge(adj, add_u, add_v)
                    # Net change: 0 edges (remove 1, add 1)
                    # But structure changed, might enable future improvements
                    # Accept with some probability
                    if random.random() < math.exp(0 / max(T, 1e-10)):
                        # Accept (neutral move)
                        pass
                    else:
                        # Reject: restore
                        remove_edge(adj, add_u, add_v)
                        add_edge(adj, rem_u, rem_v)
                else:
                    # Restore
                    add_edge(adj, rem_u, rem_v)
            
            T *= cooling_rate
            steps += 1
            
            # Check time periodically
            if steps % 1000 == 0:
                if time.time() > time_limit:
                    break
        
        return best_adj, best_edges
    
    # Multi-start with different initial configurations
    def random_initial():
        """Create a random sparse graph and saturate it."""
        adj = [0] * n
        # Add random edges sparsely
        for u, v in all_pairs:
            if random.random() < 0.15:
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
        # Saturate
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
        return adj
    
    def structured_initial():
        """Create a structured initial graph based on known constructions."""
        # Try a circulant-like structure
        adj = [0] * n
        # Add edges based on modular arithmetic
        for u in range(n):
            for d in [1, 2, 3]:
                v = (u + d) % n
                if u < v:
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
        # Saturate
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
        return adj
    
    def block_initial():
        """Try block-diagonal with some cross edges."""
        adj = [0] * n
        # 4 blocks of 5 vertices, complete within blocks
        for block in range(4):
            base = block * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
        # Add some cross-block edges
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and u < v:
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
        # Saturate
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        changed = True
        return adj
    
    # Main search loop
    best_adj = [0] * n
    best_edges = 0
    
    # Try different initializations
    initializers = [random_initial, structured_initial, block_initial]
    
    # First, try each initializer with a short SA run
    for init_func in initializers:
        if time.time() > deadline:
            break
        
        init_adj = init_func()
        init_edges = count_edges(init_adj)
        
        if init_edges > best_edges:
            best_edges = init_edges
            best_adj = init_adj[:]
        
        # Run SA from this start
        sa_adj, sa_edges = simulated_annealing(init_adj, min(time.time() + 15, deadline))
        
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj[:]
    
    # Continue with more SA restarts from the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Multiple restarts from best
        num_restarts = int(remaining_time / 10)
        num_restarts = max(num_restarts, 1)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            cand_adj = best_adj[:]
            
            # Remove 1-3 random edges
            num_removals = random.randint(1, 3)
            edges = []
            for u, v in all_pairs:
                if (cand_adj[u] >> v) & 1:
                    edges.append((u, v))
            
            for _ in range(min(num_removals, len(edges))):
                if edges:
                    u, v = random.choice(edges)
                    remove_edge(cand_adj, u, v)
                    edges = [e for e in edges if e != (u, v)]
            
            # Run SA from perturbed state
            time_for_sa = min(10, deadline - time.time())
            if time_for_sa < 2:
                break
            
            sa_adj, sa_edges = simulated_annealing(cand_adj, time.time() + time_for_sa)
            
            if sa_edges > best_edges:
                best_edges = sa_edges
                best_adj = sa_adj[:]
    
    # Final saturation of the best solution
    changed = True
    while changed:
        changed = False
        for u, v in all_pairs:
            if not ((best_adj[u] >> v) & 1):
                if can_add_edge(best_adj, u, v):
                    add_edge(best_adj, u, v)
                    changed = True
    
    final_edges = count_edges(best_adj)
    
    # Verify validity
    if not is_valid(best_adj):
        # Fallback to a known valid solution
        best_adj = [0] * n
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
    
    return get_edges_from_adj(best_adj)
