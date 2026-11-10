"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 20
    
    # Bitmask-based representation for fast feasibility checks
    # adj[i] is a bitmask of neighbors of vertex i
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
        
        # 2. For all w in N(v) \ {u}, pair (u,w) gains v as common neighbor
        #    So we need: for each w in N(v), popcount(adj[u] & adj[w]) < 2 (before adding v)
        #    After adding v, it becomes popcount(adj[u] & adj[w]) + 1 (if v was not already in adj[u]&adj[w])
        #    Actually, v is in N(v) so v is not in adj[v]. Let's think carefully.
        #    When we add edge (u,v):
        #    - u gains neighbor v, v gains neighbor u
        #    - For pair (u,w) where w in N(v): common neighbors gain v (since v in N(u) now and v in N(w) iff w in N(v))
        #      Wait, v is in N(u) after adding. v is in N(w) iff w is in N(v).
        #      So for w in N(v), v becomes a common neighbor of u and w.
        #      Before adding, common(u,w) = adj[u] & adj[w]. After, it's (adj[u]|bit_v) & adj[w] = (adj[u]&adj[w]) | (bit_v & adj[w])
        #      Since v in N(w) iff w in N(v), bit_v & adj[w] is set iff w in N(v).
        #      So for w in N(v): new common = old common + 1 (v is added)
        #      We need new common <= 2, so old common <= 1.
        #    - For pair (v,w) where w in N(u): similar, u becomes common neighbor of v and w.
        #      For w in N(u): new common(v,w) = old common(v,w) + 1 (u added)
        #      We need old common(v,w) <= 1.
        #    - For pair (u,v): handled above.
        #    - For other pairs (a,b) not involving u or v: no change.
        
        # Check pairs (u,w) for w in N(v):
        neighbors_v = adj[v]
        w = 0
        while w < n:
            if (neighbors_v >> w) & 1:
                if w != u:  # w != u since u not in N(v) yet
                    if popcount(adj[u] & adj[w]) > 1:
                        return False
            w += 1
        
        # Check pairs (v,w) for w in N(u):
        neighbors_u = adj[u]
        w = 0
        while w < n:
            if (neighbors_u >> w) & 1:
                if w != v:  # w != v since v not in N(u) yet
                    if popcount(adj[v] & adj[w]) > 1:
                        return False
            w += 1
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def count_edges(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def is_k23_free(adj):
        """Verify the graph is K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if popcount(common) > 2:
                    return False
        return True
    
    # Start with a random sparse graph
    random.seed(42)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.3:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Ensure initial graph is K2,3-free; if not, remove problematic edges
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if popcount(common) > 2:
                    # Remove a random edge from the common neighborhood
                    common_list = [w for w in range(n) if (common >> w) & 1]
                    w = random.choice(common_list)
                    if (adj[u] >> w) & 1:
                        remove_edge(adj, u, w)
                        changed = True
                    elif (adj[v] >> w) & 1:
                        remove_edge(adj, v, w)
                        changed = True
                    break
            if changed:
                break
    
    best_adj = adj[:]
    best_edges = count_edges(best_adj)
    
    # Simulated annealing
    temperature = 1.0
    cooling_rate = 0.9999
    min_temperature = 1e-4
    max_steps = 500000
    
    step = 0
    current_adj = adj[:]
    current_edges = count_edges(current_adj)
    
    while step < max_steps and temperature > min_temperature:
        if time.time() > deadline:
            break
        
        # Choose a random edge operation: add or remove
        if random.random() < 0.7:
            # Try to add an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (current_adj[u] >> v) & 1:
                continue
            
            new_adj = current_adj[:]
            add_edge(new_adj, u, v)
            
            # Check K2,3-free
            if not is_k23_free(new_adj):
                continue
            
            new_edges = count_edges(new_adj)
            delta_e = new_edges - current_edges
            
            if delta_e >= 0 or random.random() < min(1.0, temperature / max(1, delta_e)):
                current_adj = new_adj
                current_edges = new_edges
                if current_edges > best_edges:
                    best_adj = current_adj[:]
                    best_edges = current_edges
        else:
            # Try to remove an edge
            # Pick a random existing edge
            u = random.randint(0, n - 1)
            if current_adj[u] == 0:
                continue
            v_mask = current_adj[u]
            # Pick a random set bit
            v = 0
            tmp = v_mask
            while tmp:
                if (tmp >> v) & 1:
                    break
                v += 1
            else:
                continue
            
            new_adj = current_adj[:]
            remove_edge(new_adj, u, v)
            
            new_edges = count_edges(new_adj)
            delta_e = new_edges - current_edges
            
            if delta_e >= 0 or random.random() < min(1.0, temperature / max(1, -delta_e)):
                current_adj = new_adj
                current_edges = new_edges
        
        step += 1
        temperature *= cooling_rate
    
    # Try to add more edges greedily at the end
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if (best_adj[u] >> v) & 1:
                    continue
                new_adj = best_adj[:]
                add_edge(new_adj, u, v)
                if is_k23_free(new_adj):
                    best_adj = new_adj
                    best_edges = count_edges(best_adj)
                    improved = True
            if time.time() > deadline:
                break
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
