import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = []
    best_count = 0
    
    # Heuristic: Try multiple random restarts with different edge orderings
    # Or start with a specific structure and improve.
    
    # Let's try a greedy build from scratch with random edge order.
    # To be effective, we should try to add edges that are "likely" to be part of a dense C6-free graph.
    
    all_edges = [[u, v] for u in range(n) for v in range(u + 1, n)]
    
    def has_c6_simple_path(u, v, adj):
        # Check if there is a simple path of length 5 between u and v
        # adj is a list of sets or bitmasks
        # Using bitmasks for speed
        # adj[i] is an integer with bits set for neighbors of i
        
        # BFS/DFS to depth 5
        # We want a path u -> x1 -> x2 -> x3 -> x4 -> v
        # We can do a recursive search or iterative deepening.
        
        # Since n=40, we can use a visited mask.
        # But we only care about depth 5.
        
        # Let's use a stack: (current_node, depth, visited_mask)
        # visited_mask includes u and current_node.
        
        stack = [(u, 0, 1 << u)]
        while stack:
            node, depth, mask = stack.pop()
            if depth == 5:
                if node == v:
                    return True
                continue
            
            # Next nodes are neighbors of node not in mask
            # And we must reach v at depth 5.
            # Pruning: if depth == 4, node must be a neighbor of v.
            
            neighbors = adj[node] & ~mask
            if depth == 4:
                # Check if v is a neighbor
                if (1 << v) & neighbors:
                    return True
                # No need to push depth 5 nodes unless they are v, 
                # but we check at pop. 
                # Actually, if depth is 4, we are at x4. We need x4 -> v.
                # So we just check if v is in neighbors.
            else:
                # Add neighbors to stack
                # Optimization: only add if path length remaining is possible?
                # Max depth 5.
                while neighbors:
                    lsb = neighbors & -neighbors
                    next_node = lsb.bit_length() - 1
                    neighbors ^= lsb
                    stack.append((next_node, depth + 1, mask | lsb))
                    
        return False

    for restart in range(50): # Max 50 restarts, will stop earlier if time runs out
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        
        adj = [0] * n
        edges = []
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            
            # Check if u and v are already connected
            if (adj[u] >> v) & 1:
                continue
                
            # Check if adding (u, v) creates a C6
            # This is equivalent to checking if there is a path of length 5 between u and v.
            if has_c6_simple_path(u, v, adj):
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return best_edges
