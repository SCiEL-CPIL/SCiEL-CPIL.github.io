import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 150s limit, keep 20s margin
    
    # Start with the known 32-vertex construction: 8 disjoint K4's
    # This gives n=32, alpha=8, C5-free
    n0 = 32
    edges0 = []
    for b in range(8):
        s = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges0.append([s + i, s + j])
    
    # Try to add vertices and edges to get n=33, 34, 35
    # Strategy: Start from the 32-vertex graph, try to add one vertex at a time
    # For each new vertex, try all possible edge connections to existing vertices
    # Check if the resulting graph is C5-free and has alpha < 9
    
    def is_c5_free(n, adj):
        """Check if graph with adjacency matrix adj (n x n) is C5-free."""
        # A 5-cycle exists if there are vertices v0-v1-v2-v3-v4-v0
        # with all edges present and v0 != v1 != v2 != v3 != v4 != v0
        # We check all 5-tuples of distinct vertices
        # For efficiency, we can use the fact that in a C5, each vertex has degree 2 in the cycle
        # Brute force: for each pair (v0, v1) that is an edge, 
        # for each v2 adjacent to v1 but not v0,
        # for each v3 adjacent to v2 but not v1 or v0,
        # check if v3 is adjacent to some v4 adjacent to v0, with v4 != v3
        
        # More efficient: iterate over all 5-cycles
        # A 5-cycle can be represented by its vertices in order
        # We fix v0 and v1 (edge), then v2 (adj to v1, not v0), v3 (adj to v2, not v0,v1),
        # then v4 must be adj to v3 and v0, and v4 not in {v0,v1,v2,v3}
        
        for v0 in range(n):
            for v1 in range(n):
                if v1 <= v0:
                    continue
                if not adj[v0][v1]:
                    continue
                for v2 in range(n):
                    if v2 in (v0, v1):
                        continue
                    if not adj[v1][v2]:
                        continue
                    for v3 in range(n):
                        if v3 in (v0, v1, v2):
                            continue
                        if not adj[v2][v3]:
                            continue
                        for v4 in range(n):
                            if v4 in (v0, v1, v2, v3):
                                continue
                            if adj[v3][v4] and adj[v4][v0]:
                                return False
        return True
    
    def has_independent_set_of_size(n, adj, size=9):
        """Check if graph has an independent set of given size using branch and bound."""
        # Use a simple backtracking with pruning
        # For n up to 35, this can be fast if we use good ordering
        
        # Convert to adjacency matrix (0/1)
        # adj[i][j] = 1 if edge exists
        
        # Greedy lower bound on independence number
        def greedy_is_size(n, adj):
            remaining = set(range(n))
            count = 0
            while remaining:
                # Pick vertex with minimum degree in remaining
                min_deg = float('inf')
                min_v = -1
                for v in remaining:
                    deg = sum(adj[v][u] for u in remaining if u != v)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                count += 1
                remaining.discard(min_v)
                # Remove all neighbors of min_v
                for u in list(remaining):
                    if adj[min_v][u]:
                        remaining.discard(u)
            return count
        
        # Upper bound: use greedy coloring of complement or just simple bound
        # For our purpose, we just need to check if alpha >= 9
        
        # Backtracking search
        def backtrack(remaining, current_size):
            if current_size >= size:
                return True
            if not remaining:
                return False
            # Upper bound: current_size + len(remaining)
            if current_size + len(remaining) < size:
                return False
            # Pick a vertex from remaining
            v = min(remaining)
            remaining.remove(v)
            # Include v
            if backtrack(remaining - {u for u in remaining if adj[v][u]}, current_size + 1):
                return True
            remaining.add(v)
            # Exclude v
            if backtrack(remaining, current_size):
                return True
            return False
        
        return backtrack(set(range(n)), 0)
    
    # Try to build larger graphs
    best_n = 32
    best_edges = edges0
    
    # Try adding vertices one by one
    for target_n in range(33, 36):
        if time.time() > deadline - 5:
            break
        
        # Start from current best and try to add (target_n - best_n) vertices
        # For simplicity, try adding one vertex at a time
        current_n = best_n
        current_edges = list(best_edges)
        
        for _ in range(target_n - best_n):
            if time.time() > deadline - 5:
                break
            
            found = False
            new_vertex = current_n
            
            # Try all possible edge sets for the new vertex
            # The new vertex can connect to any subset of existing vertices
            # But to maintain C5-freeness and low alpha, we need to be careful
            
            # Heuristic: try connecting to vertices in different blocks first
            # Or try minimal connections
            
            # For each possible adjacency pattern (2^current_n possibilities, too many)
            # Instead, try a few random patterns and some structured ones
            
            # Pattern 1: No edges (isolated vertex) - increases alpha by 1, bad
            # Pattern 2: Connect to one vertex from each block
            # Pattern 3: Connect to two vertices from one block
            
            # Let's try: connect to exactly k vertices, for k in range(1, current_n+1)
            # But that's still too many
            
            # Better approach: try to find a vertex position that maintains properties
            # Use local search: start with no edges, add edges one by one if it doesn't create C5
            
            # For now, try a few random edge sets
            random.seed(42 + target_n)
            for attempt in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Random subset of edges from new vertex
                new_edges = []
                for v in range(current_n):
                    if random.random() < 0.3:  # 30% chance to connect
                        new_edges.append([new_vertex, v])
                
                candidate_edges = current_edges + new_edges
                
                # Build adjacency matrix
                n = current_n + 1
                adj = [[0] * n for _ in range(n)]
                for u, v in candidate_edges:
                    adj[u][v] = 1
                    adj[v][u] = 1
                
                # Check C5-free
                if not is_c5_free(n, adj):
                    continue
                
                # Check alpha < 9
                if has_independent_set_of_size(n, adj, 9):
                    continue
                
                # Valid!
                best_n = n
                best_edges = candidate_edges
                current_n = n
                current_edges = candidate_edges
                found = True
                break
            
            if not found:
                # Try more systematic approach
                # Connect new vertex to one vertex from each of the 8 blocks
                # This should keep alpha controlled
                new_edges = []
                for b in range(8):
                    v = b * 4  # Connect to first vertex of each block
                    new_edges.append([new_vertex, v])
                
                candidate_edges = current_edges + new_edges
                
                n = current_n + 1
                adj = [[0] * n for _ in range(n)]
                for u, v in candidate_edges:
                    adj[u][v] = 1
                    adj[v][u] = 1
                
                if is_c5_free(n, adj) and not has_independent_set_of_size(n, adj, 9):
                    best_n = n
                    best_edges = candidate_edges
                    current_n = n
                    current_edges = candidate_edges
                    found = True
                else:
                    # Try connecting to different vertices
                    for pattern in range(16):
                        new_edges = []
                        for b in range(8):
                            offset = (pattern >> b) & 1
                            v = b * 4 + offset
                            new_edges.append([new_vertex, v])
                        
                        candidate_edges = current_edges + new_edges
                        
                        n = current_n + 1
                        adj = [[0] * n for _ in range(n)]
                        for u, v in candidate_edges:
                            adj[u][v] = 1
                            adj[v][u] = 1
                        
                        if is_c5_free(n, adj) and not has_independent_set_of_size(n, adj, 9):
                            best_n = n
                            best_edges = candidate_edges
                            current_n = n
                            current_edges = candidate_edges
                            found = True
                            break
                        
                        if time.time() > deadline - 5:
                            break
                    break
        
        if time.time() > deadline - 5:
            break
    
    # If we couldn't improve, return the 32-vertex graph
    if best_n == 32:
        return {'n': 32, 'edges': edges0}
    
    return {'n': best_n, 'edges': best_edges}
