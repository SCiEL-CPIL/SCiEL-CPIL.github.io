"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    """
    Decompose three fixed random graphs into a minimum number of cycles and edges.
    
    Strategy:
    1. Generate the three instance graphs.
    2. For each graph, use a heuristic to find a decomposition into cycles and edges.
       - First, try to find a large collection of edge-disjoint cycles using a randomized
         greedy approach with backtracking-like local search.
       - Then, the remaining edges become singleton pieces.
    3. Return the three lists of pieces.
    
    We aim to minimize the total number of pieces. A cycle of length k covers k edges
    with 1 piece, while k single edges cover k edges with k pieces. So longer cycles are
    better. We want to maximize the number of edges covered by cycles, and among those,
    prefer longer cycles.
    
    Algorithm for one graph:
    - Use a randomized greedy algorithm: repeatedly find a cycle in the remaining graph
      that is "good" (e.g., long), remove its edges, and continue.
    - To find a good cycle, we can use a heuristic: find a cycle that maximizes some
      criterion (e.g., length, or that uses vertices with high degree).
    - We can also use local search: start with a random cycle cover or a greedy decomposition,
      then try to merge edges into cycles or swap cycles to reduce the piece count.
    
    Given the time limit, we'll use a randomized greedy with multiple restarts and
    local improvement.
    """
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges, n, time_limit=40):
        """
        Decompose the edge set into cycles and single edges.
        Returns a list of pieces. Each piece is either [u, v] (an edge) or a list of
        at least 3 distinct vertices forming a cycle.
        """
        start_time = time.time()
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # We'll maintain the current edge set and try to find cycles.
        # Greedy approach: repeatedly find a cycle, remove its edges.
        # To find a good cycle, we can use a randomized DFS to find a cycle,
        # and among several random trials, pick the longest one.
        
        def find_longest_cycle(adj, edge_set, max_trials=50, max_depth=200):
            """
            Try to find a long cycle using randomized DFS.
            Returns a list of vertices in the cycle, or None if no cycle found.
            """
            best_cycle = None
            best_len = 0
            
            # Get a list of vertices
            vertices = list(adj.keys())
            if not vertices:
                return None
            
            for trial in range(max_trials):
                # Pick a random start vertex
                start = random.choice(vertices)
                # DFS to find a cycle
                # We'll use a stack-based DFS with randomization
                # To find a cycle, we need to find a path that returns to start.
                # We'll do a DFS and when we see an edge to the start vertex, we have a cycle.
                
                # To make it more effective, we can try to find a cycle that is long.
                # We'll do a randomized DFS with a depth limit.
                
                visited = {start}
                path = [start]
                stack = [(start, iter(sorted(adj[start], key=lambda x: random.random())))]
                found_cycle = None
                
                # We'll use a simple randomized DFS
                # Since the graph is small, this should be fast.
                
                # Actually, for a cycle, we need to find a simple cycle.
                # A DFS that finds a back edge to the start gives a cycle.
                
                # Let's do a proper randomized DFS
                def dfs(u, path, visited):
                    for v in list(adj[u]):
                        if v == start and len(path) >= 3:
                            # We found a cycle
                            cycle = path[:]
                            return cycle
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            result = dfs(v, path, visited)
                            if result is not None:
                                return result
                            path.pop()
                            visited.remove(v)
                    return None
                
                # To avoid deep recursion, we'll use an iterative approach
                # But for small graphs, recursion should be fine.
                
                # Let's try a few different random orderings
                cycle = dfs(start, [start], {start})
                if cycle is not None:
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
            
            return best_cycle
        
        # Better approach: use a randomized greedy that tries to find long cycles
        # We'll use a different strategy: 
        # 1. Find a cycle using a randomized algorithm
        # 2. Remove its edges
        # 3. Repeat until no more cycles can be found
        # 4. The remaining edges are singletons
        
        # To find a good cycle, we can use the following:
        # - Find a cycle in the graph
        # - Among all cycles found by random DFS, pick the longest
        
        # Let's implement a more robust cycle finder
        def find_cycle_randomized(adj, edge_set):
            """
            Find a random cycle in the graph.
            Returns a list of vertices in the cycle, or None.
            """
            vertices = list(adj.keys())
            if not vertices:
                return None
            
            # Try multiple random starts
            for _ in range(20):
                start = random.choice(vertices)
                # Do a randomized DFS to find a cycle back to start
                visited = {start}
                path = [start]
                stack = [(start, list(adj[start]))]
                random.shuffle(stack[0][1])
                
                while stack:
                    u, neighbors = stack[-1]
                    if not neighbors:
                        stack.pop()
                        if len(stack) > 0:
                            visited.discard(u)
                            path.pop()
                        continue
                    
                    v = neighbors.pop()
                    if v == start and len(path) >= 3:
                        # Found a cycle
                        return path[:]
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        stack.append((v, list(adj[v])))
                        random.shuffle(stack[-1][1])
            
            return None
        
        # Greedy decomposition
        pieces = []
        current_edges = set(edge_set)
        current_adj = defaultdict(set)
        for u, v in current_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
        
        # Try to find cycles
        while current_edges:
            if time.time() - start_time > time_limit:
                break
            
            # Find a cycle
            cycle = None
            for _ in range(30):  # Try multiple times to find a good cycle
                c = find_cycle_randomized(current_adj, current_edges)
                if c is not None:
                    if cycle is None or len(c) > len(cycle):
                        cycle = c
                        if len(cycle) >= 8:  # Good enough
                            break
            
            if cycle is None:
                # No more cycles found; the rest are single edges
                break
            
            # Remove the cycle's edges from current_edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in current_edges:
                    current_edges.remove(e)
                    current_adj[u].discard(v)
                    current_adj[v].discard(u)
                    if not current_adj[u]:
                        del current_adj[u]
                    if not current_adj[v]:
                        del current_adj[v]
            
            pieces.append(cycle[:])
        
        # Remaining edges are singletons
        for e in current_edges:
            pieces.append(list(e))
        
        return pieces
    
    # Generate the three graphs
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
