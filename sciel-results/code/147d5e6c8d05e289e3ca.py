import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_pairs = len(all_pairs)
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_common_count(adj, u, v):
        """Count common neighbors of u and v using bitmasks."""
        return popcount(adj[u] & adj[v])
    
    def is_valid_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # For each x in N(v) \ {u}: common[u][x] must be < 2
        # For each y in N(u) \ {v}: common[v][y] must be < 2
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v] & (1 << x):
                if get_common_count(adj, u, x) >= 2:
                    return False
        for y in range(n):
            if y == u or y == v:
                continue
            if adj[u] & (1 << y):
                if get_common_count(adj, v, y) >= 2:
                    return False
        return True
    
    def edge_count(adj):
        return sum(popcount(x) for x in adj) // 2
    
    def random_valid_graph():
        """Generate a random valid graph by greedily adding edges."""
        adj = [0] * n
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if is_valid_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    # Simulated annealing with remove-and-refill moves
    best_adj = random_valid_graph()
    best_edges = edge_count(best_adj)
    
    current_adj = best_adj[:]
    current_edges = best_edges
    
    t0 = 1.0
    t_end = 0.01
    cooling_rate = (t_end / t0) ** (1.0 / 50000)
    t = t0
    
    rng = random.Random(12345)
    
    while time.time() - start_time < deadline:
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.7:
            # Try to add a random non-edge
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                continue  # Already an edge
            if is_valid_edge(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            else:
                # Reject with probability based on temperature
                # Adding invalid edge: reject
                pass
        elif move_type < 0.9:
            # Remove a random edge
            existing_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current_adj[u] & (1 << v):
                        existing_edges.append((u, v))
            if not existing_edges:
                continue
            u, v = rng.choice(existing_edges)
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges -= 1
        else:
            # Remove-and-refill: remove 1-3 random edges, then greedily add back
            num_remove = rng.randint(1, 3)
            existing_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current_adj[u] & (1 << v):
                        existing_edges.append((u, v))
            if len(existing_edges) < num_remove:
                continue
            
            to_remove = rng.sample(existing_edges, num_remove)
            for u, v in to_remove:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            current_edges -= num_remove
            
            # Greedily add back edges
            pairs = all_pairs[:]
            rng.shuffle(pairs)
            added = 0
            for u, v in pairs:
                if not (current_adj[u] & (1 << v)):
                    if is_valid_edge(current_adj, u, v):
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        current_edges += 1
                        added += 1
                if added >= num_remove:
                    break
            
            # Accept if improved or with probability based on temperature
            if current_edges >= best_edges:
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            else:
                # Metropolis acceptance
                delta = best_edges - current_edges
                prob = math.exp(-delta / max(t, 0.001))
                if rng.random() > prob:
                    # Revert to best
                    current_adj = best_adj[:]
                    current_edges = best_edges
        
        # Cool down
        t *= cooling_rate
        if t < t_end:
            t = t_end
    
    # Final polish: greedy add all possible edges
    for u, v in all_pairs:
        if not (best_adj[u] & (1 << v)):
            if is_valid_edge(best_adj, u, v):
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                best_edges += 1
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
