import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Try to construct graphs from n=35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline - 5.0:
            break
        result = constructive_search(n, deadline)
        if result is not None:
            return result
    
    # Fallback: Paley graph on 17 vertices
    return paley_17()

def paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    return {'n': 17, 'edges': edges}

def constructive_search(n, deadline):
    """
    Constructive backtracking search.
    Build graph vertex by vertex. For each new vertex, decide edges to previous vertices.
    Constraint: no K4, no independent 6-set.
    """
    
    def would_create_k4(adj, new_v, candidate_neighbors):
        """Check if adding new_v with edges to candidate_neighbors creates a K4."""
        for u in candidate_neighbors:
            for v in candidate_neighbors:
                if u >= v:
                    continue
                if adj[u] & (1 << v):
                    # u and v are adjacent. Check if both are adjacent to new_v (they are by construction)
                    # and if there's a common neighbor w of u and v that is also a neighbor of new_v
                    # Actually, K4 = {new_v, u, v, w} where w is adjacent to all three.
                    # w must be in candidate_neighbors and adjacent to u and v.
                    common = adj[u] & adj[v]
                    # w must be in candidate_neighbors
                    for w in candidate_neighbors:
                        if w != new_v and (common >> w) & 1:
                            return True
        return False

    def would_create_ind6(adj, new_v, non_neighbors):
        """Check if new_v with non-edges to non_neighbors creates an independent 6-set."""
        # new_v is independent from all vertices in non_neighbors.
        # We need to check if there exist 5 vertices among non_neighbors that are mutually independent
        # AND also independent from each other.
        # Actually, we need 5 vertices from non_neighbors that form an independent set in the existing graph.
        # So we need to check if the subgraph induced by non_neighbors has an independent set of size 5.
        
        def find_ind_k(subgraph_adj, k, vertices):
            """Find independent set of size k in induced subgraph."""
            if len(vertices) < k:
                return None
            if k == 0:
                return []
            if k == 1:
                return [vertices[0]]
            
            # Branch on first vertex
            v = vertices[0]
            # Option 1: include v
            remaining_without_v = [u for u in vertices[1:] if not (subgraph_adj[v] >> u) & 1]
            res = find_ind_k(subgraph_adj, k-1, remaining_without_v)
            if res is not None:
                return [v] + res
            # Option 2: exclude v
            res = find_ind_k(subgraph_adj, k, vertices[1:])
            if res is not None:
                return res
            return None
        
        # Check if non_neighbors has an independent set of size 5
        # Use the existing adj for the induced subgraph
        return find_ind_k(adj, 5, list(non_neighbors)) is not None

    def backtrack(adj, current_v, deadline):
        """
        adj: list of bitmasks for vertices 0..current_v-1
        current_v: next vertex to place
        Returns True if successfully placed all vertices, False otherwise.
        Modifies adj in place.
        """
        if time.time() > deadline:
            return False
        
        if current_v == n:
            return True
        
        # Try different edge assignments for vertex current_v
        # The edges to previous vertices are a subset of {0, ..., current_v-1}
        # We need to choose which of these to connect to.
        
        # Heuristic: try to connect to vertices that are "safe"
        # A vertex u is "safe to connect" if adding edge (current_v, u) doesn't immediately create a K4
        # A vertex u is "safe to not connect" if not adding edge doesn't immediately create an ind6
        
        # Generate candidate edge sets
        # For efficiency, try a few random orderings
        
        prev_vertices = list(range(current_v))
        
        # Try multiple random orderings
        for attempt in range(50):
            if time.time() > deadline:
                return False
            
            random.shuffle(prev_vertices)
            
            # Greedy assignment with backtracking
            assigned = {}  # vertex -> bool (edge exists)
            
            def assign_recursive(idx, assigned):
                if time.time() > deadline:
                    return False
                
                if idx == len(prev_vertices):
                    # All vertices assigned. Check global constraints.
                    # Build full adj for current_v
                    adj[current_v] = 0
                    for u in prev_vertices:
                        if assigned[u]:
                            adj[current_v] |= (1 << u)
                            adj[u] |= (1 << current_v)
                    
                    # Check no K4 involving current_v
                    if has_k4_with_vertex(adj, current_v, n):
                        # Undo
                        for u in prev_vertices:
                            if assigned[u]:
                                adj[u] &= ~(1 << current_v)
                        adj[current_v] = 0
                        return False
                    
                    # Check no independent 6-set involving current_v
                    if has_ind6_with_vertex(adj, current_v, n):
                        # Undo
                        for u in prev_vertices:
                            if assigned[u]:
                                adj[u] &= ~(1 << current_v)
                        adj[current_v] = 0
                        return False
                    
                    # Recurse
                    return backtrack(adj, current_v + 1, deadline)
                
                v = prev_vertices[idx]
                
                # Try adding edge
                assigned[v] = True
                if not would_create_k4_local(adj, current_v, v, assigned):
                    if assign_recursive(idx + 1, assigned):
                        return True
                
                # Try not adding edge
                assigned[v] = False
                if not would_create_ind6_local(adj, current_v, v, assigned):
                    if assign_recursive(idx + 1, assigned):
                        return True
                
                return False
            
            # Initialize adj for current_v
            adj.append(0)
            
            if assign_recursive(0, {}):
                return True
            
            # Undo
            adj.pop()
        
        return False

    def would_create_k4_local(adj, new_v, v, assigned):
        """Quick check: if we add edge (new_v, v), does it create a K4?"""
        # K4 would be {new_v, v, u, w} where u,w are previous vertices
        # u and w must be adjacent to each other, to v, and to new_v
        # u and w must be in assigned with value True
        for u in assigned:
            if not assigned[u]:
                continue
            if not (adj[u] & (1 << v)):
                continue
            for w in assigned:
                if w <= u or not assigned[w]:
                    continue
                if not (adj[w] & (1 << v)):
                    continue
                if not (adj[u] & (1 << w)):
                    continue
                # Check if u and w are both connected to new_v
                # They will be if assigned[u] and assigned[w] are True
                return True
        return False

    def would_create_ind6_local(adj, new_v, v, assigned):
        """Quick check: if we don't add edge (new_v, v), does it create an ind6?"""
        # This is hard to check locally. Skip for now, rely on global check.
        return False

    def has_k4_with_vertex(adj, v, n):
        """Check if vertex v is part of any K4."""
        neighbors = [u for u in range(n) if u != v and (adj[v] >> u) & 1]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, w = neighbors[i], neighbors[j]
                if (adj[u] >> w) & 1:
                    # u and w are adjacent. Check if there's a common neighbor of u and w that is also neighbor of v
                    common = adj[u] & adj[w] & adj[v]
                    # Remove v itself
                    common &= ~(1 << v)
                    if common:
                        return True
        return False

    def has_ind6_with_vertex(adj, v, n):
        """Check if vertex v is part of any independent 6-set."""
        non_neighbors = [u for u in range(n) if u != v and not ((adj[v] >> u) & 1)]
        
        def find_ind_k(k, vertices):
            if len(vertices) < k:
                return False
            if k == 0:
                return True
            if k == 1:
                return True
            x = vertices[0]
            # Include x: need k-1 from vertices that are non-neighbors of x
            remaining = [u for u in vertices[1:] if not ((adj[x] >> u) & 1)]
            if find_ind_k(k - 1, remaining):
                return True
            # Exclude x
            if find_ind_k(k, vertices[1:]):
                return True
            return False
        
        return find_ind_k(5, non_neighbors)

    # Initialize
    adj = [0] * n
    
    # Try multiple random restarts with different initial orderings
    for restart in range(20):
        if time.time() > deadline - 2.0:
            break
        
        # Reset
        adj = [0] * n
        
        # Randomize vertex ordering for construction
        order = list(range(n))
        random.shuffle(order)
        
        # Remap: new index i corresponds to old vertex order[i]
        # Actually, let's just try the backtrack with the natural ordering but random shuffles in the inner loop
        
        if backtrack(adj, 0, deadline):
            # Success! Extract edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    return None
