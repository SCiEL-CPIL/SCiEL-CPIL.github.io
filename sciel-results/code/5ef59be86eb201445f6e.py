"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s
    n = 40
    
    # Represent graph as list of bitmasks (one per vertex)
    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    def edge_count():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # Check if there's a path of length exactly 5 from u to v (simple path)
    # This means adding edge (u,v) would create a C6
    def creates_c6(u, v):
        """Check if there's a simple path of length 5 from u to v in current graph."""
        # BFS/DFS from u, tracking depth and visited
        # We need path u -> a -> b -> c -> d -> v, all distinct
        # So from u, go to depth 5, and v must be at depth 5
        # Use DFS with bitset for visited
        
        # Depth 0: {u}
        # We need to reach v at depth 5
        # DFS: (current_vertex, depth, visited_mask)
        
        # For efficiency, use iterative deepening or just BFS tracking paths
        # Since n=40, we can do a DFS
        
        stack = [(u, 0, 1 << u)]
        while stack:
            node, depth, visited = stack.pop()
            if depth == 5:
                if node == v:
                    return True
                continue
            if depth >= 5:
                continue
            # Get neighbors not in visited
            neighbors = adj[node] & ~visited
            # Also exclude v unless depth+1 == 5 (we can arrive at v at depth 5)
            # Actually we should allow arriving at v only at depth 5
            if depth + 1 < 5:
                neighbors &= ~(1 << v)  # don't visit v before depth 5
            while neighbors:
                lsb = neighbors & (-neighbors)
                nxt = lsb.bit_length() - 1
                neighbors ^= lsb
                stack.append((nxt, depth + 1, visited | lsb))
        return False
    
    # Full C6 check: does the graph contain any C6?
    def has_c6():
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(u, v):
                    if creates_c6(u, v):
                        return True
        return False
    
    # Faster: check if specific non-edge (u,v) would create a C6 if added
    # (same as creates_c6 above)
    
    # Initialize with a greedy construction
    random.seed(42)
    
    # Start with empty graph, greedily add edges
    for u in range(n):
        for v in range(u + 1, n):
            add_edge(u, v)
            if creates_c6(u, v):
                remove_edge(u, v)
    
    best_edges = edge_count()
    best_adj = adj[:]
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    current_adj = adj[:]
    current_edges = best_edges
    current_c6_free = True  # we started with C6-free
    
    # Verify current is C6-free
    # (it should be from greedy)
    
    while time.time() < deadline:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or has_edge(u, v):
                continue
            # Try adding
            add_edge(u, v)
            if creates_c6(u, v):
                # This creates a C6, reject
                remove_edge(u, v)
                # Delta: we lost 1 edge (but we didn't actually add it)
                # Actually the move is invalid, so reject
                continue
            else:
                # Accept (improvement)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
        elif move_type < 0.8:
            # Remove a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or not has_edge(u, v):
                continue
            remove_edge(u, v)
            current_edges -= 1
            # Always accept removal (it can't create a C6)
            # But it might be part of a future improvement
        else:
            # Swap: remove one edge, add another
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1 or not has_edge(u1, v1):
                continue
            remove_edge(u1, v1)
            current_edges -= 1
            
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or has_edge(u2, v2):
                # Can't add, revert
                add_edge(u1, v1)
                current_edges += 1
                continue
            add_edge(u2, v2)
            if creates_c6(u2, v2):
                # Revert both
                remove_edge(u2, v2)
                add_edge(u1, v1)
                current_edges += 1
                continue
            # Net change: 0 edges (removed one, added one)
            # Accept with probability based on... it's neutral, always accept
    
    # Return best found
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
    return result

# EVOLVE-BLOCK-END
