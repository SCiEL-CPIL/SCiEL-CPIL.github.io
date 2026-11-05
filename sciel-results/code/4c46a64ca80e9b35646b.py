import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # 150s limit, leave 35s margin
    
    n = 16
    
    # Helper: Check 3-colorability
    def is_3_colorable(adj):
        # adj is list of 16 integers (bitmasks)
        colors = [-1] * n
        available = [(1 << 3) - 1] * n  # 0b111
        
        def backtrack(v):
            if v == n:
                return True
            # MRV heuristic: pick vertex with fewest available colors
            # But we are iterating in order. Let's just use order 0..15 for simplicity first.
            # For 16 vertices, simple backtracking is fast.
            avail = available[v]
            if avail == 0:
                return False
            
            # Try each color
            while avail:
                c_bit = avail & -avail
                c = c_bit.bit_length() - 1
                avail ^= c_bit
                
                # Check if color c is available for v
                # It is if no neighbor has color c
                # We can maintain this dynamically or check on the fly.
                # On the fly:
                valid = True
                for i in range(n):
                    if (adj[v] >> i) & 1:
                        if colors[i] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                colors[v] = c
                # Update available colors for neighbors
                for i in range(n):
                    if (adj[v] >> i) & 1:
                        available[i] &= ~c_bit
                        if available[i] == 0:
                            # Prune
                            break
                else:
                    if backtrack(v + 1):
                        return True
                
                # Undo
                colors[v] = -1
                for i in range(n):
                    if (adj[v] >> i) & 1:
                        available[i] |= c_bit
            
            return False
        
        return backtrack(0)

    # Helper: Get edges from adjacency bitmask
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Helper: Add edge if triangle-free
    def try_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # Check triangle
        if adj[u] & adj[v]:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    # Helper: Greedy expansion
    def expand(adj):
        # Make a copy
        adj_copy = adj[:]
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj_copy[u] >> v) & 1):
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Check if still addable (neighbors might have changed)
            if not ((adj_copy[u] >> v) & 1) and not (adj_copy[u] & adj_copy[v]):
                adj_copy[u] |= (1 << v)
                adj_copy[v] |= (1 << u)
        
        return adj_copy

    # Initialize best
    best_edge_count = 0
    best_adj = None

    # 1. Start with Clebsch Graph
    # Clebsch: dist(u,v) in 5-bit cube (quotient) is 1 or 4.
    # u,v in 0..15.
    clebsch_adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            diff = u ^ v
            dist = bin(diff).count("1")
            if dist == 1 or dist == 4:
                clebsch_adj[u] |= (1 << v)
                clebsch_adj[v] |= (1 << u)
    
    # Expand Clebsch
    while time.time() < deadline:
        expanded = expand(clebsch_adj)
        edge_count = sum(bin(x).count("1") for x in expanded) // 2
        if edge_count > best_edge_count:
            # Verify 3-colorability (should be non-3-colorable)
            if not is_3_colorable(expanded):
                best_edge_count = edge_count
                best_adj = expanded[:]
            else:
                # Should not happen, but if it does, we can't use it
                pass

        # Also try local search on the expanded graph
        if best_adj is not None:
            current_adj = best_adj[:]
            # Try removing a random edge and re-expanding
            edges_list = get_edges(current_adj)
            if edges_list:
                u, v = random.choice(edges_list)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                new_adj = expand(current_adj)
                new_count = sum(bin(x).count("1") for x in new_adj) // 2
                if new_count > best_edge_count and not is_3_colorable(new_adj):
                    best_edge_count = new_count
                    best_adj = new_adj[:]

    # 2. Random Maximal Triangle-Free Generation
    while time.time() < deadline:
        adj = [0] * n
        # Random order of edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not ((adj[u] >> v) & 1) and not (adj[u] & adj[v]):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        edge_count = sum(bin(x).count("1") for x in adj) // 2
        if edge_count > best_edge_count:
            if not is_3_colorable(adj):
                best_edge_count = edge_count
                best_adj = adj[:]

    if best_adj is None:
        # Fallback to Clebsch if something went wrong
        return get_edges(clebsch_adj)

    return get_edges(best_adj)

if __name__ == "__main__":
    construct()
