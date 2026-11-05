import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit
    
    n = 24
    rng = random.Random(614)
    
    # Generate the shuffled edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    
    # Map edge to its 1-based position
    pos = {}
    for i, (u, v) in enumerate(edges):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1
    
    # Pre-build prefix adjacency structures incrementally
    # For each K, we need adj[u] = set of v such that edge (u,v) is in first K edges
    
    # We'll use binary search on K
    # Lower bound: n (need at least n edges for a cycle)
    # Upper bound: n*(n-1)//2 = 276
    
    lo = n
    hi = n * (n - 1) // 2
    
    # First, let's do a quick check: find the smallest K where min degree >= 2
    # This gives us a better lower bound
    deg = [0] * n
    for i in range(n):
        u, v = edges[i]
        deg[u] += 1
        deg[v] += 1
    
    # Find min K where all degrees >= 2
    min_deg_K = n
    for K in range(n, hi + 1):
        if min(deg) >= 2:
            min_deg_K = K
            break
        if K < hi:
            u, v = edges[K]
            deg[u] += 1
            deg[v] += 1
    
    lo = max(lo, min_deg_K)
    
    best_cycle = None
    best_K = float('inf')
    
    # Binary search for the minimum K with a Hamilton cycle
    # But we also need to verify by actually finding the cycle
    
    def find_hamilton_cycle(K, time_limit):
        """Try to find a Hamilton cycle in G_K (first K edges).
        Returns the cycle as a list of vertices, or None if not found."""
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for i in range(K):
            u, v = edges[i]
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree >= 2
        for u in range(n):
            if len(adj[u]) < 2:
                return None
        
        # Check connectivity
        visited = [False] * n
        visited[0] = True
        stack = [0]
        count = 1
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return None
        
        # Iterative DFS with Warnsdorff's heuristic
        # We'll try multiple strategies
        
        for strategy in range(20):
            if time.time() > time_limit:
                break
            
            if strategy == 0:
                # Standard Warnsdorff: prefer neighbors with fewest unvisited neighbors
                start_node = 0
                shuffle_seed = 0
            elif strategy == 1:
                # Reverse Warnsdorff: prefer neighbors with most unvisited neighbors
                start_node = 0
                shuffle_seed = 1
            elif strategy == 2:
                # Start from a random node
                start_node = strategy - 2
                shuffle_seed = strategy
            elif strategy == 3:
                # Start from node with min degree
                start_node = min(range(n), key=lambda x: len(adj[x]))
                shuffle_seed = 3
            elif strategy == 4:
                # Start from node with max degree
                start_node = max(range(n), key=lambda x: len(adj[x]))
                shuffle_seed = 4
            else:
                # Various random starts
                r = random.Random(strategy * 1000 + 7)
                start_node = r.randint(0, n - 1)
                shuffle_seed = strategy
            
            # Iterative DFS
            visited = [False] * n
            path = [start_node]
            visited[start_node] = True
            # Stack for backtracking: (node, iterator over neighbors)
            # We'll use a simpler approach: explicit stack with state
            
            # Use a stack-based DFS
            # Each entry: (current_node, next_neighbor_index)
            # But we need to track the order of neighbors tried
            
            # Simpler: recursive-style with explicit stack
            # State: (node, list_of_untried_neighbors, path_so_far)
            
            # Let's use a different approach: 
            # Maintain path and a stack of (node, choices_list)
            
            path = [start_node]
            visited[start_node] = True
            # choices_stack[i] = list of neighbors to try from path[i]
            choices_stack = []
            
            # Get neighbors of start_node, sorted by heuristic
            def get_neighbors_sorted(u, visited_arr, reverse=False):
                unvisited = [v for v in adj[u] if not visited_arr[v]]
                if not unvisited:
                    return []
                
                def warnsdorff_deg(v):
                    return sum(1 for w in adj[v] if not visited_arr[w])
                
                if reverse:
                    unvisited.sort(key=warnsdorff_deg, reverse=True)
                else:
                    unvisited.sort(key=warnsdorff_deg)
                return unvisited
            
            reverse_heuristic = (strategy == 1)
            
            # Initialize
            choices_stack.append(get_neighbors_sorted(start_node, visited, reverse_heuristic))
            
            found = False
            max_depth = n
            
            while choices_stack:
                if time.time() > time_limit:
                    break
                
                u = path[-1]
                choices = choices_stack[-1]
                
                if len(path) == n:
                    # Check if we can close the cycle
                    if start_node in adj[u]:
                        found = True
                        break
                    # Can't close, backtrack
                    choices_stack.pop()
                    visited[u] = False
                    path.pop()
                    continue
                
                if not choices:
                    # No more choices, backtrack
                    choices_stack.pop()
                    visited[u] = False
                    path.pop()
                    continue
                
                # Try next choice
                v = choices.pop(0)  # Pop from front (most preferred)
                
                # Put remaining choices back
                choices_stack[-1] = choices
                
                visited[v] = True
                path.append(v)
                choices_stack.append(get_neighbors_sorted(v, visited, reverse_heuristic))
            
            if found:
                return path[:]
        
        return None
    
    # Binary search
    # First, let's check if there's a Hamilton cycle at the upper bound
    # to make sure we have a valid answer
    
    # Actually, let's just do a linear search from lo upward, but with a 
    # smarter approach: try each K and stop when we find one.
    # Since we're minimizing, and the answer is likely small, linear is fine.
    # But binary search is better for the search space.
    
    # Let's do: first verify hi works, then binary search
    # But for now, let's do linear from lo, since the answer is likely close to lo
    
    for K in range(lo, hi + 1):
        if time.time() > deadline - 5.0:
            break
        
        cycle = find_hamilton_cycle(K, deadline - 5.0)
        if cycle is not None:
            best_cycle = cycle
            best_K = K
            break
    
    if best_cycle is None:
        # Fallback: try larger K values
        for K in range(lo, hi + 1):
            if time.time() > deadline - 2.0:
                break
            cycle = find_hamilton_cycle(K, deadline - 2.0)
            if cycle is not None:
                best_cycle = cycle
                best_K = K
                break
    
    if best_cycle is None:
        # Last resort: use the full graph
        # Find any Hamilton cycle in K_n
        # Simple: 0, 1, 2, ..., 23
        best_cycle = list(range(n))
    
    return best_cycle

if __name__ == "__main__":
    print(construct())
