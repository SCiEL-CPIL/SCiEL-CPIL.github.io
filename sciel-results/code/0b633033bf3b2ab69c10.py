import time
import random
import sys

# Set recursion limit just in case, though depth is small
sys.setrecursionlimit(2000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 40
    
    # --- Helper Functions ---
    
    def init_adj():
        return [0] * n

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def has_edge(adj, u, v):
        return (adj[u] >> v) & 1

    def count_edges(adj):
        # Sum of bits in all rows, divide by 2
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    def get_neighbors(adj, u):
        # Return list of neighbors
        neighbors = []
        mask = adj[u]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            neighbors.append(v)
            mask ^= lsb
        return neighbors

    def has_path_len_5(u, v, adj):
        """
        Returns True if there is a simple path of length 5 between u and v.
        Uses DFS.
        """
        if u == v:
            return False # Path length 0
        
        visited = 1 << u
        
        def dfs(current, target, depth):
            if depth == 5:
                return current == target
            if depth > 5:
                return False
            
            # Pruning: if remaining steps > max possible? 
            # Max distance in graph is n-1. 
            # If 5 - depth > n - 1, impossible. (Not useful here)
            
            neighbors = get_neighbors(adj, current)
            for nb in neighbors:
                if not (visited >> nb) & 1:
                    if dfs(nb, target, depth + 1):
                        return True
            return False
        
        return dfs(u, v, 1)

    def find_c6_pair(adj):
        """
        Finds a pair (u, v) such that adding edge (u, v) would create a C6.
        i.e., there is a path of length 5 between u and v.
        Returns (u, v) or None.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(adj, u, v):
                    if has_path_len_5(u, v, adj):
                        return (u, v)
        return None

    def is_c6_free(adj):
        return find_c6_pair(adj) is None

    def greedy_fill(adj):
        """
        Try to add all possible edges that do not create a C6.
        Returns the number of edges added.
        """
        added = 0
        # Iterate over all pairs
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(adj, u, v):
                    if not has_path_len_5(u, v, adj):
                        add_edge(adj, u, v)
                        added += 1
        return added

    # --- Main Logic ---
    
    # 1. Initialize
    # Start with a random graph
    p = 0.2
    adj = init_adj()
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                add_edge(adj, u, v)
    
    # 2. Clean: Remove edges from C6s until none exist
    # We might need to do this multiple times if removing one edge creates another?
    # No, removing an edge cannot create a new cycle. It can only break cycles.
    # So one pass is enough if we remove edges until no C6 is found.
    while True:
        pair = find_c6_pair(adj)
        if pair is None:
            break
        u, v = pair
        # Remove a random edge from the path? 
        # We don't have the path, just the pair.
        # Any edge on the path will break the C6.
        # But we don't know the path.
        # Let's just remove a random edge from the graph? 
        # No, that's inefficient.
        # Let's find the path.
        
        # Modify has_path_len_5 to return the path?
        # Or just pick a random neighbor?
        # Actually, if there is a path of length 5, removing ANY edge from that path breaks it.
        # We can find the path using the same DFS.
        
        # For now, let's just remove a random edge from the graph.
        # This is safe but might remove too many edges.
        # Better: Find the path.
        
        # Let's implement a helper to get the path
        def get_path_len_5(u, v, adj):
            visited = 1 << u
            path = [u]
            
            def dfs(current, target, depth):
                if depth == 5:
                    if current == target:
                        return True
                    return False
                
                neighbors = get_neighbors(adj, current)
                random.shuffle(neighbors)
                for nb in neighbors:
                    if not (visited >> nb) & 1:
                        visited |= (1 << nb)
                        path.append(nb)
                        if dfs(nb, target, depth + 1):
                            return True
                        path.pop()
                        visited &= ~(1 << nb)
                return False
            
            if dfs(u, v, 1):
                return path
            return None
        
        path = get_path_len_5(u, v, adj)
        if path:
            # Remove a random edge from the path
            edge_idx = random.randint(0, 4)
            e_u = path[edge_idx]
            e_v = path[edge_idx + 1]
            remove_edge(adj, e_u, e_v)
        else:
            # Should not happen
            break

    # 3. Greedy Fill
    greedy_fill(adj)
    best_edges = count_edges(adj)
    best_adj = adj[:] # Copy
    
    # 4. Local Search
    while time.time() < deadline:
        # Try to add more edges
        added = greedy_fill(adj)
        if added > 0:
            best_edges = count_edges(adj)
            best_adj = adj[:]
            continue
        
        # If no edges could be added, try a move
        # Pick a random edge to remove
        edges_list = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges_list.append((u, v))
                mask ^= lsb
        
        if not edges_list:
            break
            
        u, v = random.choice(edges_list)
        
        # Save state
        old_adj = adj[:]
        
        # Remove edge
        remove_edge(adj, u, v)
        
        # Try to fill
        added = greedy_fill(adj)
        
        new_edges = count_edges(adj)
        
        if new_edges > best_edges:
            best_edges = new_edges
            best_adj = adj[:]
        else:
            # Restore
            adj = old_adj
            
        # Small pause to check time
        if time.time() > deadline - 10:
            break

    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                result_edges.append([u, v])
            mask ^= lsb
            
    return result_edges

